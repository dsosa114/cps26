try:
    import pyzed.sl as sl
except ImportError as details:
    print ("Could not import pyzed because of this error:", details)
    print ("zed_camera class will not be available")
    sl = None

try:
    from openni import openni2, nite2, utils
except ImportError as details:
    print ("Could not import openni because of this error:", details)
    print ("ps_camera class will not be available")
    openni2 = None

try:
    from openni import _openni2 as c_api
except ImportError as details:
    print ("Could not import openni because of this error:", details)
    print ("ps_camera class will not be available")
    c_api = None

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError as details:
    print ("Could not import ultralytics because of this error:", details)
    print (" YOLOv8_Detection class will not be available")
    c_api = None

import numpy as np
import requests
import cv2

class zed_camera:
    def __init__(self) -> None:
        self.zed = self.init_capture_device()
        self.image_offset = (0, 0)
        self.ROI = []
        self.activeROI = False
        self.transformer = None
        self.calibration = 'perspective'
        self.sdk_version = self.zed.get_sdk_version()

        if self.zed is None:
            print("NO CAMERA DETECTED. THE PROGRAM WILL CLOSE.")
            exit(1)

        if "4" in self.sdk_version:
            w = self.zed.get_camera_information().camera_configuration.resolution.width
            h = self.zed.get_camera_information().camera_configuration.resolution.height
        else:
            w = self.zed.get_camera_information().camera_resolution.width
            h = self.zed.get_camera_information().camera_resolution.height
        
        print("Resolution: {0}x{1}".format(w, h))
            # Create an RGBA sl.Mat object
        self.im_left_zed = sl.Mat(w, h, sl.MAT_TYPE.U8_C4)
        self.im_right_zed = sl.Mat(w, h, sl.MAT_TYPE.U8_C4)
        # Create a sl.Mat with float type (32-bit)
        self.depth_zed = sl.Mat(w, h, sl.MAT_TYPE.F32_C1)
        self.point_cloud = sl.Mat()
        self.normal_map = sl.Mat()
        ### ZED 1
        # self.k_left = np.array([[1401.9200, 0, 951.4500],
        #               [0, 1401.9200, 539.6150],
        #               [0,   0,   1]], dtype=np.float32) #For 1080p resolution
        # self.k_right = np.array([[1394.7700, 0, 917.6100],
        #               [0, 1394.7700, 575.8270],
        #               [0,   0,   1]], dtype=np.float32) #For 1080p resolution
        
        ### ZED 2
        self.k_left = np.array([[1068.9200, 0, 961.9400],
                    [0, 1068.6500, 512.3030],
                    [0,   0,   1]], dtype=np.float32) #For 1080p resolution
        self.k_right = np.array([[1070.7000, 0, 955.8200],
                    [0, 1070.5000, 554.9720],
                    [0,   0,   1]], dtype=np.float32) #For 1080p resolution
    
    def init_capture_device(self):
        zed = sl.Camera()

        # Set configuration parameters
        init_params = sl.InitParameters()
        #init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
        #init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        #init_params.camera_fps = 30  # Set fps at 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode
        init_params.camera_fps = 30  # Set fps at 30

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(err)
            return None
        # exit(1)
        return zed
    
    def close(self):
        self.zed.close()

    def get_images(self):
        '''
        Returns:
            left_im = image generated in the left camera
            right_im = image generated in the right camera
            depth = depth map 
            point_cloud = point cloud information
        '''
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS :
                # Retrieve the left image in sl.Mat
                self.zed.retrieve_image(self.im_left_zed, sl.VIEW.LEFT)
                #timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
                #print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(im_left_zed.get_width(), im_left_zed.get_height(), timestamp.get_milliseconds()))
                # Retrieve the right image in sl.Mat
                self.zed.retrieve_image(self.im_right_zed, sl.VIEW.RIGHT)
                #timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
                #print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(im_right_zed.get_width(), im_right_zed.get_height(), timestamp.get_milliseconds()))
                # Use get_data() to get the numpy array
                left_rgb = self.im_left_zed.get_data()
                right_rgb = self.im_right_zed.get_data()
                # Display the left image from the numpy array
                #cv2.imshow("Image", image_ocv)
                # Retrieve depth data (32-bit)
                self.zed.retrieve_measure(self.depth_zed, sl.MEASURE.DEPTH) # Retrieve depth matrix. Depth is aligned on the left RGB image
                # Load depth data into a numpy array
                depth_ocv = self.depth_zed.get_data()
                # Print the depth value at the center of the image
                #print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
                self.zed.retrieve_measure(self.normal_map, sl.MEASURE.NORMALS)
        
        return left_rgb, right_rgb, depth_ocv, [self.point_cloud, self.normal_map]
    
    def pix_to_real_coord(self, points:list, depth:float, error:float = 0.):
        objectives = []
        if self.calibration == 'perspective':
            for point in points:
                if len(point) < 3:
                    point.append(0.)
                elif len(point) > 3:
                    additional = point.pop(-1)
                pix_coord = np.array([point[0], point[1], 1])
                scaled_coord = np.linalg.inv(self.k_left)@pix_coord
                xyz_coord = ((depth + (error*depth**-2))/scaled_coord[2])*scaled_coord#*np.array([-1., -1., 1.])
                objectives.append(np.hstack((xyz_coord, np.array([0., 0., point[2]]))))
        elif self.calibration == 'homography':
            for point in points:
                objective = np.append(self.transformer.transform_points(point).squeeze(), depth)
                if len(point) > 2:
                    additional = point.pop(-1)
                else:
                    additional = 0.
                objectives.append(np.hstack((objective, np.array([0., 0., additional]))))
        return objectives
    #define the events for the
    # mouse_click.
    def mouse_click(self, event, x, y, 
                    flags, param):
        img = np.copy(param)
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     if len(self.ROI) < 4:
        #         self.ROI.append((x,y))
        #     else:
        #         self.ROI.pop(0)
        #         self.ROI.append((x,y))
        #     for point in self.ROI:
        #         cv2.circle(img, point, 5, (0,0,255), -1)
        #         cv2.imshow("Trajectory", img)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.ROI) < 4:
                self.ROI.append((x*2,y*2))
            else:
                self.ROI.pop(0)
                self.ROI.append((x*2,y*2))
            for point in self.ROI:
                cv2.circle(img, (int(point[0]/2), int(point[1]/2)), 3, (0,0,255), -1)
                cv2.imshow("ROI", img)

    def define_image_roi(self):
        if self.activeROI:
            a = input("A ROI has been alredy selected. Do you want to rewrite it? (y/n)")
            if a == 'y' or a == 'Y':
                self.activeROI = False
                try:
                    img, _, _, _ = self.get_images()
                    h, w, _ = img.shape

                    img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_LINEAR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    cv2.imshow("ROI",img)
                    #cv2.namedWindow("Trajectory")  
                    cv2.setMouseCallback("ROI", self.mouse_click, param=img) 
                    #cap.release()
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    self.activeROI = True
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error) # An exception occurred: division by zero
                    cv2.destroyAllWindows()
        else:
            try:
                img, _, _, _ = self.get_images()
                h, w, _ = img.shape

                img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                cv2.imshow("ROI",img)
                #cv2.namedWindow("Trajectory")  
                cv2.setMouseCallback("ROI", self.mouse_click, param=img) 
                #cap.release()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.activeROI = True
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error) # An exception occurred: division by zero
                cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()
    
    def homography(self, real_coord:np.array):
        self.homography_depth = np.amax(real_coord[:,-1])
        if self.activeROI:
            self.transformer = ViewTransformer(np.array(self.ROI, dtype=np.float32), real_coord[:,:-1])
            self.calibration = 'homography'

class ps_camera:
    def __init__(self, res:tuple =(480, 640)) -> None:
        self.width = res[1]
        self.height = res[0]
        self.fps = 30
        self.image_offset = (0, 0)
        self.ROI = []
        self.activeROI = False
        self.activeSearchArea = False
        self.searchArea = None
        self.dev = self.init_capture_device()
        self.transformer = None
        self.calibration = 'perspective'
        if res[1] > 600:
            self.fx = 558.2502053
            self.fy = 558.58893773
            self.c_u = 313.84869948
            self.c_v = 243.77689141
        else:
            self.fx = 366.92
            self.fy = 388.99
            self.c_u = 9.0621
            self.c_v = 181.46

        self.K = np.array([[self.fx, 0., self.c_u],
                           [0., self.fy, self.c_v],
                           [0., 0., 1.]])

        if self.dev is None:
            print("NO CAMERA DETECTED. THE PROGRAM WILL CLOSE.")
            exit(1)

        dev_name = self.dev.get_device_info().name.decode('UTF-8')
        print("Device Name: {}".format(dev_name))

        self.rgb_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()
        ##configure the depth_stream
        #print 'Get b4 video mode', depth_stream.get_video_mode()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
                                                            resolutionX=self.width, 
                                                            resolutionY=self.height, 
                                                            fps=self.fps))
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                        resolutionX=self.width,
                                                        resolutionY=self.height,
                                                        fps=self.fps))
        ## Check and configure the mirroring -- default is True
        # print 'Mirroring info1', depth_stream.get_mirroring_enabled()
        self.depth_stream.set_mirroring_enabled(False)
        self.rgb_stream.set_mirroring_enabled(False)

        ## synchronize the streams
        self.dev.set_depth_color_sync_enabled(True) # synchronize the streams

        ## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
        self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        ##help(dev.set_image_registration_mode)

        ## start the stream
        self.rgb_stream.start()
        self.depth_stream.start()

     
    def init_capture_device(self):
        openni2.initialize()
        return openni2.Device.open_any()

    def close_capture_device(self):
        openni2.unload()

    def get_rgb(self, rgb_stream):
        """
        Returns numpy 3L ndarray to represent the rgb image.
        """
        #bgr   = np.frombuffer(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
        frame_rgb = rgb_stream.read_frame()
        frame_rgb_data = frame_rgb.get_buffer_as_uint8()
        bgr   = np.ndarray((frame_rgb.height, frame_rgb.width, 3),dtype=np.uint8,buffer=frame_rgb_data)
        rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        return bgr, rgb
    #get_rgb

    def get_depth(self, depth_stream):
        """
        Returns numpy ndarrays representing the raw and ranged depth images.
        Outputs:
            dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
            d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
        Note1:
            fromstring is faster than asarray or frombuffer
        Note2:
            .reshape(120,160) #smaller image for faster response
                    OMAP/ARM default video configuration
            .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                    Requires .set_video_mode
        """
        #dmap = np.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)  # Works & It's FAST
        frame_depth = depth_stream.read_frame()
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        dmap = np.ndarray((frame_depth.height, frame_depth.width),dtype=np.uint16,buffer=frame_depth_data).astype(np.float32)
        d4d = np.uint8(dmap *255/ 2**12-1) # Correct the range. Depth images are 12bits
        d4d = 255 - cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
        depth_uint8 = dmap*0.05+0
        depth_uint8[depth_uint8>255] = 255
        depth_uint8[depth_uint8<0] = 0
        depth_uint8 = depth_uint8.astype('uint8')
        return dmap, d4d, depth_uint8
    #get_depth

    def mask_rgbd(self, d4d,rgb, th=0):
        """
        Overlays images and uses some blur to slightly smooth the mask
        (3L ndarray, 3L ndarray) -> 3L ndarray
        th:= threshold
        """
        mask = d4d.copy()
        #mask = cv2.GaussianBlur(mask, (5,5),0)
        idx =(mask>th)
        mask[idx] = rgb[idx]
        return mask
    
    def get_images(self):
        ## Streams
        #RGB
        _, rgb = self.get_rgb(self.rgb_stream)
        #resized_rgb = cv2.resize(rgb,(640, 480))
        #DEPTH
        _,d4d, _ = self.get_depth(self.depth_stream)
        #resized_depth = cv2.resize(d4d,(640, 480))
        # Overlay rgb over the depth stream
        rgbd  = self.mask_rgbd(d4d,rgb)

        if self.activeSearchArea:
            img = rgb[self.searchArea[1]:(self.searchArea[1]+self.searchArea[3]), self.searchArea[0]:(self.searchArea[0]+self.searchArea[2])]

        else:
            img = rgb

        return img, d4d, rgbd, self.image_offset
    
    def close(self):
        self.depth_stream.stop()
        self.rgb_stream.stop()
        self.close_capture_device()

    def pix_to_real_coord(self, points:list, depth:float, error:float = 0.):
        objectives = []
        if self.calibration == 'perspective':
            for point in points:
                if len(point) < 3:
                    point.append(0.)
                elif len(point) > 3:
                    additional = point.pop(-1)
                pix_coord = np.array([point[0], point[1], 1])
                scaled_coord = np.linalg.inv(self.K)@pix_coord
                xyz_coord = ((depth + (error*depth**-2))/scaled_coord[2])*scaled_coord#*np.array([-1., -1., 1.])
                objectives.append(np.hstack((xyz_coord, np.array([0., 0., point[2]]))))
        elif self.calibration == 'homography':
            for point in points:
                objective = np.append(self.transformer.transform_points(point).squeeze(), depth)
                # objective = self.transformer.transform_points(point).squeeze()
                if len(point) > 2:
                    additional = point.pop(-1)
                else: 
                    additional = 0.
                objectives.append(np.hstack((objective, np.array([0., 0., additional]))))
        return objectives
    #define the events for the
    # mouse_click.
    def mouse_click(self, event, x, y, 
                    flags, param):
        img = np.copy(param)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix = x
            self.iy = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.rectangle(img, (self.ix, self.iy), (x, y),(255, 0, 0), 2)
                cv2.imshow("ROI", img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
            self.searchArea = np.array([self.ix, self.iy, x-self.ix, y-self.iy], dtype=np.int32)
            self.image_offset = (self.ix, self.iy)
            self.activeSearchArea = True
            cv2.imshow("ROI", img)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.ROI) < 4:
                self.ROI.append((x,y))
            else:
                self.ROI.pop(0)
                self.ROI.append((x,y))
            for point in self.ROI:
                cv2.circle(img, point, 3, (0,0,255), -1)
                cv2.imshow("ROI", img)

    def define_image_roi(self):
        self.drawing = False
        self.ix = -1
        self.iy = -1
        if self.activeROI or self.activeSearchArea:
            a = input("A ROI has been alredy selected. Do you want to rewrite it? (y/n)")
            if a == 'y' or a == 'Y':
                self.activeROI = False
                self.activeSearchArea = False
                try:
                    img, _, _, _ = self.get_images()
                    cv2.imshow("ROI",img)
                    #cv2.namedWindow("Trajectory")  
                    cv2.setMouseCallback("ROI", self.mouse_click, param=img) 
                    #cap.release()
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    self.activeROI = True
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error) # An exception occurred: division by zero
                    cv2.destroyAllWindows()
        else:
            try:
                img, _, _, _ = self.get_images()
                cv2.imshow("ROI",img)
                #cv2.namedWindow("Trajectory")  
                cv2.setMouseCallback("ROI", self.mouse_click, param=img) 
                #cap.release()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.activeROI = True
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error) # An exception occurred: division by zero
                cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()
    
    def homography(self, real_coord:np.array):
        self.homography_depth = np.amax(real_coord[:,-1])
        if self.activeROI:
            self.transformer = ViewTransformer(np.array(self.ROI, dtype=np.float32), real_coord[:,:-1])
            self.calibration = 'homography'


class wrist_camera:
    def __init__(self, ip:str = '172.23.254.231') -> None:
        self.resp = None
        self.count = 0
        self.ip = ip
        self.K = np.array([[718.90118408, 0., 327.12475112],
                           [0., 713.69824219, 213.88042599],
                           [0., 0., 1.]])
        self.image_offset = (0, 0)
        self.ROI = None
        self.activeROI = False

        try:
            # http://<robot_ip_address>:4242/current.jpg?type=color
            # http://<robot_ip_address>:4242/current.jpg?type=edges
            # http://<robot_ip_address>:4242/current.jpg?type=magnitude
            # http://<robot_ip_address>:4242/current.jpg?type=annotations
            self.resp = requests.get(f"http://{ip}:4242/current.jpg?type=color").content
        except:
            pass

    def get_image(self):
        try:
            # resp = requests.get("http://172.23.254.231:4242/current.jpg?type=color").content
            self.resp = requests.get(f"http://{self.ip}:4242/current.jpg?type=color").content
        except Exception as error:
            print(f"Someting went wrong. {error}. Couldn't retrieve a valid image")
            return None, None
        #Check the response
        if self.resp == None:
            #If the response is empty display an error message
            print("Couldn't retrieve a valid image. Check camera connection.")
            return None, None
        else:
            #If the response is not empty format the data to get a
            #tkinter format image
            frame = np.asarray(bytearray(self.resp), dtype="uint8")
            image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if self.activeROI:
                img = image[self.ROI[1]:(self.ROI[1]+self.ROI[3]), self.ROI[0]:(self.ROI[0]+self.ROI[2])]

            else:
                img = image

            return img, self.image_offset
    
    def mouse_click(self, event, x, y, 
                    flags, param):
        img = np.copy(param)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.im_point) < 1:
                self.im_point.append([x+self.image_offset[0],y+self.image_offset[1]])
            else:
                self.im_point.pop(0)
                self.im_point.append([x+self.image_offset[0],y+self.image_offset[1]])
            for point in self.im_point:
                cv2.circle(img, (point[0]-self.image_offset[0], point[1]-self.image_offset[1]), 3, (0,0,255), -1)
                cv2.imshow("Image point", img)

    def get_point_from_image(self, depth, **args):
        self.im_point = []
        try:
            img, _= self.get_image()
            cv2.imshow("Image point",img)
            #cv2.namedWindow("Trajectory")  
            cv2.setMouseCallback("Image point", self.mouse_click, param=img) 
            #cap.release()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero
            cv2.destroyAllWindows()
        targets = self.pix_to_real_coord(self.im_point, depth=depth, **args)
        
        return targets
        
    def pix_to_real_coord(self, points:list, depth:float, error:float = 0.):
        objectives = []
        for point in points:
            if len(point) < 3:
                point.append(0.)
            elif len(point) > 3:
                additional = point.pop(-1)
            pix_coord = np.array([point[0], point[1], 1])
            scaled_coord = np.linalg.inv(self.K)@pix_coord
            xyz_coord = ((depth + (error*depth**-2))/scaled_coord[2])*scaled_coord*np.array([-1., -1., 1.])
            objectives.append(np.hstack((xyz_coord, np.array([0., 0., point[2]]))))
            
        return objectives
    
    def define_image_roi(self):
        if self.activeROI:
            a = input("A ROI has been alredy selected. Do you want to rewrite it? (y/n)")
            if a == 'y' or a == 'Y':
                self.activeROI = False
                try:
                    img, _ = self.get_image()
                    roi = cv2.selectROI(img)
                    print(f"Selected ROI (left, top, width, height): {roi}")
                    if int(roi[2] * roi[3]) > 0:
                        self.ROI = roi
                        self.image_offset = (roi[0], roi[1])
                        self.activeROI = True
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error) # An exception occurred: division by zero
                    cv2.destroyAllWindows()
        else:
            try:
                img, _ = self.get_image()
                roi = cv2.selectROI(img)
                print(f"Selected ROI (left, top, width, height): {roi}")
                if int(roi[2] * roi[3]) > 0:
                    self.ROI = roi
                    self.image_offset = (roi[0], roi[1])
                    self.activeROI = True
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error) # An exception occurred: division by zero
                cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()
        
    def world_space_roi(self, depth, **args):
        corners = []
        if self.activeROI:
            top, bottom, left, right = self.ROI[1], int(self.ROI[1]+self.ROI[3]), self.ROI[0], int(self.ROI[0]+self.ROI[2])
            points = [[right+25, bottom], [left, bottom], [left, top], [right+25, top]]
            targets = self.pix_to_real_coord(points, depth=depth, **args)
            for item in targets:
                corners.append(item[:3])
            return np.array(corners, dtype=np.float32), targets
        else:
            print("First is necessary to define a ROI in the image.")
            self.define_image_roi()
            top, bottom, left, right = self.ROI[1], int(self.ROI[1]+self.ROI[3]), self.ROI[0], int(self.ROI[0]+self.ROI[2])
            points = [[right, bottom], [left, bottom], [left, top], [right, top]]
            targets = self.pix_to_real_coord(points, depth=depth, **args)
            for item in targets:
                corners.append(item[0][:-1])
            return np.array(corners, dtype=np.float32), targets


class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

class YOLOv8_Detection():

    def __init__(self, model:str='/home/daniel/Codes/robodk/test_codes/runs/segment/train2/weights/best.pt') -> None:
        ultralytics.checks()
        self.model = YOLO(model)
        class_names = self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in class_names]

    def overlay(self, image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.
        
        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        # color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def predict(self, image, SOURCE_ROI):
        h, w, _ = image.shape
        centers = []
        results = self.model.predict(task='segment', model=self.model, conf=0.25, source=image)
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):

                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])

                if ((xmax > int(np.amax(SOURCE_ROI[:,0])) or 
                        xmax < int(np.amin(SOURCE_ROI[:,0]))) or
                        (ymax > int(np.amax(SOURCE_ROI[:,1])) or 
                        ymax < int(np.amin(SOURCE_ROI[:,1])))):
                    break

                seg = cv2.resize(seg, (w, h))
                image = self.overlay(image, seg, self.colors[int(box.cls)], 0.4)
                    
                
                xcenter = int(xmin + ((xmax-xmin)/2))
                ycenter = int(ymax - ((ymax-ymin)/2))
                centers.append(np.array([xcenter, ycenter]))   
                # plot_one_box([xmin, ymin, xmax, ymax], left, color=colors[int(box.cls)], label=f'{class_names[int(box.cls)]} {float(box.conf):.3}')
                self.plot_one_box([xmin, ymin, xmax, ymax],image, color=self.colors[int(box.cls)], label=None)

        return centers, image