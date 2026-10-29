import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

class PositionChangeDetector:
    """
    Detects significant changes in a 3D position compared to the last
    accepted position based on a specified threshold.
    """
    def __init__(self, initial_position, threshold, high_threshold=1.0):
        """
        Initializes the detector.

        Args:
            initial_position (np.ndarray or list/tuple): The starting 3D position.
                                                        Example: np.array([x, y, z])
            threshold (float): The minimum distance (e.g., Euclidean) that
                               constitutes a significant change.
        """
        if not isinstance(initial_position, np.ndarray):
            initial_position = np.array(initial_position, dtype=float)
        if initial_position.shape != (3,):
            raise ValueError("initial_position must be a 3D array or convertible to one.")

        self.last_accepted_position = initial_position
        self.threshold = threshold
        self.high_threshold = high_threshold 
        self.sudden_move_detected = False
        self.first_run = True
        print(f"PositionChangeDetector initialized. Last accepted: {self.last_accepted_position}, Threshold: {self.threshold}")


    def update_position(self, new_position):
        """
        Processes a new position and determines if it's a significant change.

        Args:
            new_position (np.ndarray or list/tuple): The new 3D position.

        Returns:
            tuple: (has_changed, current_position_to_use)
                   - has_changed (bool): True if the new position was accepted
                                         as a significant change, False otherwise.
                   - current_position_to_use (np.ndarray): The new position if it
                                                           changed significantly, otherwise
                                                           the last accepted position.
        """
        if not isinstance(new_position, np.ndarray):
            new_position = np.array(new_position, dtype=float)
        if new_position.shape != (3,):
            raise ValueError("new_position must be a 3D array or convertible to one.")

        # Calculate Euclidean distance between the new and last accepted position
        distance = np.linalg.norm(new_position - self.last_accepted_position)

        if distance > self.threshold and distance < self.high_threshold:
            print(f"Significant change detected. \U0001F535 Distance: {distance:.4f} > Threshold: {self.threshold}")
            print(f"Old position: {self.last_accepted_position}, New position: {new_position}")
            self.last_accepted_position = new_position
            return True, self.last_accepted_position
        elif distance >= self.high_threshold:
            if self.first_run:
                print("First run, accepting initial position without threshold check.")
                self.last_accepted_position = new_position
                self.first_run = False
                return False, self.last_accepted_position
            print(f"\U0001F534 Sudden move detected. Distance: {distance:.4f} >= High Threshold: {self.high_threshold}. "
                  f"Position {new_position} accepted.")
            self.sudden_move_detected = True
            self.last_accepted_position = new_position
            return True, self.last_accepted_position
        else:
            # print(f"Insignificant change. Distance: {distance:.4f} <= Threshold: {self.threshold}. Position {new_position} ignored.")
            return False, self.last_accepted_position
        
        
class ObjectTracker:
    def __init__(self, target_id, occlusion_timeout=10, max_reassociation_distance=50.0):
        """
        Initializes the Object Tracker.

        Args:
            target_id (int or str): The initial ID of the object to track.
            occlusion_timeout (int): Number of frames to wait before declaring occlusion
                                     and sending a stop signal.
            max_reassociation_distance (float): Maximum distance between the last known
                                                position and a new detection for re-association.
        """
        self.initial_target_id = target_id
        self.current_tracking_id = target_id
        self.last_known_position = None  # (x, y)
        self.occlusion_timeout = occlusion_timeout
        self.max_reassociation_distance = max_reassociation_distance
        self.occlusion_counter = 0
        self.is_occluded = False
        self.stop_signal_sent = False

    def _calculate_distance(self, pos1, pos2):
        """Calculates Euclidean distance between two points."""
        if pos1 is None or pos2 is None:
            return float('inf')
        
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        # d = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
        print(f"Distance between {pos1} and {pos2}: {distance:.2f}")
        return  distance

    def _send_stop_signal(self):
        """Placeholder for sending a stop signal."""
        print(f"🛑 STOP SIGNAL: Object {self.current_tracking_id} is occluded or lost.")
        self.stop_signal_sent = True
        # In a real system, this would trigger an actual stop mechanism.

    def _reset_stop_signal(self):
        """Resets the stop signal flag."""
        if self.stop_signal_sent:
            print(f"🟢 RESUMING: Tracking re-established for ID {self.current_tracking_id}.")
            self.stop_signal_sent = False

    def process_detections(self, detections):
        """
        Processes new object detections for the current frame.

        Args:
            detections (list): A list of dictionaries, where each dictionary
                               represents a detected object and should have
                               'id' and 'position' (e.g., (x, y) tuple) keys.
                               Example: [{'id': 1, 'position': (10, 20)},
                                         {'id': 2, 'position': (30, 40)}]
        """
        found_current_id = False

        # 1. Search for the current_tracking_id
        for detection in detections:
            det_id = detection.get('id')
            det_pos = detection.get('position')

            # print(f"Processing detection: ID={det_id}, Position={det_pos}, ID_type={type(det_id)}, "
            #       f"Position_type={type(det_pos)}")
            # Validate detection format
            if det_id is None or (det_pos is None or np.all(np.isnan(det_pos))):
                print(f"Warning: Invalid detection format: {detection}")
                continue

            if det_id == self.current_tracking_id:
                # print(f"✅ Object {self.current_tracking_id} found at {det_pos}.")
                self.last_known_position = det_pos
                self.occlusion_counter = 0
                self.is_occluded = False
                self._reset_stop_signal()
                found_current_id = True
                return # Exit processing for this frame

        # 2. Handle Potential Occlusion if current_tracking_id not found
        if not found_current_id:
            self.occlusion_counter += 1
            print(f"⚠️ Object {self.current_tracking_id} not found. Occlusion counter: {self.occlusion_counter}")

            if self.occlusion_counter >= self.occlusion_timeout:
                if not self.is_occluded: # Send stop signal only once per occlusion event
                    self._send_stop_signal()
                self.is_occluded = True

                # 3. Attempt Re-association if occluded
                if self.last_known_position.all(): # Can only re-associate if we have a last known position
                    potential_candidates = []
                    for detection in detections:
                        new_det_id = detection.get('id')
                        new_det_pos = detection.get('position')

                        if det_id is None or (det_pos is None or np.all(np.isnan(det_pos))):
                            continue

                        distance = self._calculate_distance(self.last_known_position, new_det_pos)

                        # Check if the new ID is the old ID + 1 (or other custom logic)
                        # And if it's within the re-association distance
                        # You can make the ID change logic more flexible if needed
                        is_potential_reassociated_id = (
                            isinstance(self.current_tracking_id, (int, float)) and
                            isinstance(new_det_id, (int, float)) #and
                            # new_det_id == self.current_tracking_id + 1
                        )
                        # Fallback: or simply the closest if ID rule doesn't match but still very close
                        # is_fallback_candidate = True # (you could add more conditions here)

                        if distance < self.max_reassociation_distance:
                            # Prioritize the ID + 1 rule
                            if is_potential_reassociated_id:
                                print(f"🔄 Re-associating ID {self.current_tracking_id} with new ID {new_det_id} "
                                      f"at {new_det_pos} (distance: {distance:.2f}, ID incremented).")
                                self.current_tracking_id = new_det_id
                                self.last_known_position = new_det_pos
                                self.occlusion_counter = 0
                                self.is_occluded = False
                                self._reset_stop_signal()
                                return # Re-associated
                            else:
                                potential_candidates.append({'id': new_det_id, 'position': new_det_pos, 'distance': distance})

                    # If ID+1 rule didn't match, check other close candidates
                    if self.is_occluded and potential_candidates: # Still occluded and have candidates
                        # Sort by distance to find the closest one
                        potential_candidates.sort(key=lambda x: x['distance'])
                        closest_candidate = potential_candidates[0]
                        print(f"🔄 Considering re-association for occluded ID {self.current_tracking_id} "
                              f"with closest new ID {closest_candidate['id']} at {closest_candidate['position']} "
                              f"(distance: {closest_candidate['distance']:.2f}).")
                        # Add a confirmation step or make it automatic:
                        # For now, let's automatically re-associate if it's the only close one
                        # or based on stricter criteria if many are close.
                        # This example takes the closest one if the ID+1 rule failed.
                        self.current_tracking_id = closest_candidate['id']
                        self.last_known_position = closest_candidate['position']
                        self.occlusion_counter = 0
                        self.is_occluded = False
                        self._reset_stop_signal()
                        print(f"Re-associated with closest candidate.")
                        return

                if self.is_occluded and not self.stop_signal_sent: # Ensure stop signal if still occluded and no re-association
                     self._send_stop_signal()

            else: # Temporarily missing, but not yet timed out
                print(f"⏳ Object {self.current_tracking_id} temporarily missing. "
                      f"Waiting... ({self.occlusion_counter}/{self.occlusion_timeout})")
                self.is_occluded = False # Not officially occluded yet



def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def read_and_save_image(src, dst = None, format=cv2.COLOR_BGR2RGB):
    print("This function 'read_and_save_image' by default converts the input file to RGB color space if format is not specified")
    image = cv2.imread(src)
    image = cv2.cvtColor(image, format)
    if isnotebook():
        plt.imshow(image)

    #saving image
    if dst is None:
        cv2.imwrite(src, image)
    else:
        cv2.imwrite(dst, image)

    return image

def read_image(src, format=cv2.COLOR_BGR2RGB):
    ''' Some colorspaces codes
        cv::ColorConversionCodes {
            cv::COLOR_BGR2BGRA = 0,
            cv::COLOR_RGB2RGBA = COLOR_BGR2BGRA,
            cv::COLOR_BGRA2BGR = 1,
            cv::COLOR_RGBA2RGB = COLOR_BGRA2BGR,
            cv::COLOR_BGR2RGBA = 2,
            cv::COLOR_RGB2BGRA = COLOR_BGR2RGBA,
            cv::COLOR_RGBA2BGR = 3,
            cv::COLOR_BGRA2RGB = COLOR_RGBA2BGR,
            cv::COLOR_BGR2RGB = 4,
            cv::COLOR_BGR2GRAY = 6,
            cv::COLOR_RGB2GRAY = 7,
            cv::COLOR_GRAY2BGR = 8,
            cv::COLOR_GRAY2RGB = COLOR_GRAY2BGR,
            cv::COLOR_GRAY2BGRA = 9,
            cv::COLOR_BGR2YCrCb = 36,
            cv::COLOR_RGB2YCrCb = 37,
            cv::COLOR_YCrCb2BGR = 38,
            cv::COLOR_YCrCb2RGB = 39,
            cv::COLOR_HSV2BGR_FULL = 70,
            cv::COLOR_HSV2RGB_FULL = 71,
            cv::COLOR_BGR2YUV = 82,
            cv::COLOR_RGB2YUV = 83,
            cv::COLOR_YUV2BGR = 84,
            cv::COLOR_YUV2RGB = 85,
        }
        *find more in https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html
    '''
    # print("This function 'read_image' by default converts the input file to RGB color space if format is not specified")
    image = cv2.imread(src)
    if isnotebook():
        image = cv2.cvtColor(image, format)

    return image

def save_image(image, dst):
    #Only for arrays
    cv2.imwrite(dst, image)

def resize_image_from_file(src, size, method=cv2.INTER_LINEAR):
    #reading the image
    image = cv2.imread(src)
    #converting image to size (100,100,3)
    smaller_image = cv2.resize(image, size, interpolation=method)

    if isnotebook():
        #plot the resized image
        plt.imshow(smaller_image)

    return smaller_image

def resize_image(image, size, method=cv2.INTER_LINEAR):
    ''' cv::InterpolationFlags {
            cv::INTER_NEAREST = 0,
            cv::INTER_LINEAR = 1,
            cv::INTER_CUBIC = 2,
            cv::INTER_AREA = 3,
            cv::INTER_LANCZOS4 = 4,
            cv::INTER_LINEAR_EXACT = 5,
            cv::INTER_NEAREST_EXACT = 6,
            cv::INTER_MAX = 7,
            cv::WARP_FILL_OUTLIERS = 8,
            cv::WARP_INVERSE_MAP = 16
        }'''
    #converting image to size
    smaller_image = cv2.resize(image, size, interpolation=method)
    if isnotebook():
        #plot the resized image
        plt.imshow(smaller_image)

    return smaller_image

def image_contours(image, minVal=127, maxVal=255,minArea=500):
    #converting RGB image to Binary
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,minVal,maxVal,cv2.THRESH_BINARY_INV)
    #calculate the contours from binary image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # with_contours = cv2.drawContours(image,contours,-1,(0,255,0),3)

    return thresh, [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]

def sift_keypoints(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #create sift object
    sift  = cv2.xfeatures2d.SIFT_create()
    #calculate keypoints and their orientation
    keypoints,descriptors = sift.detectAndCompute(gray,None)
    #plot keypoints on the image
    with_keypoints = cv2.drawKeypoints(gray,keypoints)
    if isnotebook():
        #plot the image
        plt.imshow(with_keypoints)

    return with_keypoints, keypoints, descriptors

def surf_keypoints(image):
    #convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #instantiate surf object
    surf  = cv2.xfeatures2d.SURF_create(400)
    #calculate keypoints and their orientation
    keypoints,descriptors = surf.detectAndCompute(gray,None)

    with_keypoints = cv2.drawKeypoints(gray,keypoints)
    if isnotebook():
        #plot the image
        plt.imshow(with_keypoints)

    return with_keypoints, keypoints, descriptors

def show_image(image):
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image)
    plt.show()

def hough_lines(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #In canny you can modify thres1 and thres2
    edges = cv2.Canny(gray,150,250,apertureSize = 3)
    #show_image(edges)
    lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=120, minLineLength=30, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    return image

def hough_circles(image, radius=(30,60), param1=200, param2=50, dp=20, thresh=(150,255), mask=None, offset=(0,0), **args):
    min, max = radius
    keypoints = []
    #img = cv2.medianBlur(image,5)
    #img = cv2.GaussianBlur(image,(5,5),1)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if mask is None:
        img = binarize(image, thresh=thresh)
    else: 
        img = mask
    #img = blobs(img)
    #In canny you can modify thres1 and thres2
    edges = cv2.Canny(img,150,250,apertureSize = 3)
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1.2,dp,
                            param1=param1,param2=param2,minRadius=min,maxRadius=max)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
            keypoints.append([i[0] + int(offset[0]), i[1] + int(offset[1]), 0.0, i[2]])
    return image, keypoints

def basic_object_detection(img, minArea=1100, image_offset:tuple=(0,0), mask:list = [True, True ,True],**args):
    '''
    Parameters:\n
    \tminArea: Min. area in pixels of the blobs
    \tmask: [squares, circles, masses], defaults to True
    '''
    
    info = {'square':[], 'mass':[], 'circle':[]}

    if mask[0]:
        img, k_squares,_,_ = contours(img, minArea=minArea, offset=image_offset, **args)
        if len(k_squares) > 0:
            info['square'].extend(k_squares) 
    # mask, info = cvbf.square_blobs(img, minArea=1100)
    # if len(k_rect) > 0:
    #     print(k_rect[0].pt, k_rect[0].size)
    if mask[1]:
        img, k_cir = hough_circles(img, offset=image_offset,**args)
        if len(k_cir) > 0:
            info['circle'].extend(k_cir)  
    
    if mask[2]:
        img, masses, _ = mass_center(img, minArea=minArea, offset=image_offset, **args)
        if len(masses) > 0:
            info['mass'].extend(masses)  
    
    return info, img

def extended_object_detection(img, minArea=1100, image_offset:tuple=(0,0),mask:list = [True, True ,True],**args):
    '''
    Parameters:\n
    \tminArea: Min. area in pixels of the blobs
    \tmask: [squares, circles, masses], defaults to True
    '''
    info = {'square':[], 'ext_square':[], 'mass':[], 'ext_mass':[], 'circle':[]}

    if mask[0]:
        img, k_squares, _, e_squares = contours(img, minArea=minArea, offset=image_offset, **args)
        if len(k_squares) > 0:
            info['square'].extend(k_squares)
            info['ext_square'].extend(e_squares) 
    # mask, info = cvbf.square_blobs(img, minArea=1100)
    # if len(k_rect) > 0:
    #     print(k_rect[0].pt, k_rect[0].size)
    if mask[1]:
        img, k_cir = hough_circles(img, offset=image_offset,**args)
        if len(k_cir) > 0:
            info['circle'].extend(k_cir)  
    
    if mask[2]:
        img, masses, e_masses = mass_center(img, minArea=minArea, offset=image_offset, **args)
        if len(masses) > 0:
            info['mass'].extend(masses)
            info['ext_mass'].extend(e_masses)    
    
    return info, img

def mass_center(image, offset=(0,0),**args):
    img, keypoints = square_blobs(image, **args)
    info = []
    ext_info = []
    # edges = cv2.Canny(img,150,255,apertureSize = 3)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        for cnt in contours:
            if cnt.shape[0] > 5:
                M = cv2.moments(cnt)
                # print(M['m00'])
                cX = int(M["m10"] / M["m00"]) + int(offset[0])
                cY = int(M["m01"] / M["m00"]) + int(offset[1])
                cv2.circle(image, (cX-int(offset[0]), cY-int(offset[1])), 5, (255, 25, 200), -1)
                _, diags, angle = cv2.fitEllipse(cnt)
                info.append([cX, cY, angle])
                ext_info.append([(cX, cY), diags, angle])

    return image, info, ext_info

def contours(image, offset=(0,0),**args):
    info = []
    ext_info = []
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #In canny you can modify thres1 and thres2
    #img = binarize(image, thresh=(200,255))
    img, keypoints = square_blobs(image, **args)
    #edges = cv2.Canny(img,150,250,apertureSize = 3)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        for cnt in contours:
            epsilon = 0.08*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            if len(approx)>=4:
                x,y,w,h = cv2.boundingRect(approx)
                cx = int(x+w/2) + int(offset[0])
                cy = int(y+h/2) + int(offset[1])
                cv2.circle(image, (cx - int(offset[0]), cy - int(offset[1])), 5, (55, 255, 20), -1)
                if cnt.shape[0] >= 5:
                    _, diags, angle = cv2.fitEllipse(cnt)
                else:
                    _, diags, angle = (0, 0), (0, 0), 0
                info.append([cx, cy, angle])
                ext_info.append([(cx, cy), diags, angle])
                ar = w / float(h)
                if ar >= 0.95 and ar <= 1.05:
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(image,[box],0,(255,0,0),2)
                else: #or ((aspect_ratio_h/aspect_ratio_v)>=0.8):
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(image,[box],0,(255,0,255),2)
                #keypoints.append(box)
    return image, info, contours, ext_info

def square_blobs(image, minArea=1900, mask=None, **args):
    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    # Printing dictionary items
    img = binarize(image, thresh=(150,255))
    for key in args:
        if key == 'thresh':
            if mask is None:
                img = binarize(image, args[key])
            else: 
                img = mask
        else:
            if mask is not None:
                img = mask
    img = cv2.bitwise_not(img)
    new_mask = np.zeros(img.shape, np.uint8)
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = minArea
    for key in args:
        if key == 'maxArea':
            params.maxArea = args[key]

    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.9

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Get the number of blobs found
    blobCount = len(keypoints)
    if blobCount > 0:
        for keypoint in keypoints:
            #print(keypoint.pt[0], keypoint.pt[1], keypoint.size)
            x1 = int(keypoint.pt[0] - keypoint.size*1.5/2)
            x2 = int(keypoint.pt[0] + keypoint.size*1.5/2)
            y1 = int(keypoint.pt[1] - keypoint.size*1.5/2)
            y2 = int(keypoint.pt[1] + keypoint.size*1.5/2)
            #blobs = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
            new_mask = cv2.rectangle(new_mask,(x1,y1),(x2,y2),(255,255,255),-3)
            blobs = cv2.bitwise_and(new_mask, cv2.bitwise_not(img))
    else:
        blobs = new_mask

    return blobs, keypoints

def mouse_click(event, x, y, 
                flags, param):
    global ROI
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
        if len(ROI) < 4:
            ROI.append((x*2,y*2))
        else:
            ROI.pop(0)
            ROI.append((x*2,y*2))
        for point in ROI:
            cv2.circle(img, (int(point[0]/2), int(point[1]/2)), 3, (0,0,255), -1)
            cv2.imshow("ROI", img)

def define_image_roi(image):
    global ROI
    ROI = []
    try:
        img = np.copy(image)
        h, w, _ = img.shape

        img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imshow("ROI",img)
        #cv2.namedWindow("Trajectory")  
        cv2.setMouseCallback("ROI", mouse_click, param=img) 
        #cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error) # An exception occurred: division by zero
        cv2.destroyAllWindows()

    return ROI
#empty function
def doNothing(x):
    pass

def color_picker_from_image(frame)->dict:
    '''
    Returns:\n
        \t A dictionary with the min and max values of each color channel in BGR format.

    '''
    #creating a resizable window named Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    #creating track bars for gathering threshold values of red green and blue
    cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)

    while True:
        cv2.imshow('Image', frame)

        #converting into HSV color model
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV Image', hsv_image)
        #reading the trackbar values for thresholds
        min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
        min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
        min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

        max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
        max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
        max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

        #using inrange function to turn on the image pixels where object threshold is matched
        mask = cv2.inRange(hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #showing the mask image
        cv2.imshow('Mask Image', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return {'B':(min_blue, max_blue), 'G':(min_green, max_green), 'R':(min_red, max_red)}

def mask_color(image, color:dict):
    min_blue, max_blue = color['B']
    min_green, max_green = color['G']
    min_red, max_red = color['R']
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
    
    return mask


def circular_blobs(image):
    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    img = binarize(image, thresh=(200,255))
    img = cv2.bitwise_not(img)
    mask = np.zeros(img.shape, np.uint8)
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 1
    #params.maxArea = 1000

    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.78

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Get the number of blobs found
    # blobCount = len(keypoints)
    # if blobCount > 0:
    #     for keypoint in keypoints:
    #         #print(keypoint.pt[0], keypoint.pt[1], keypoint.size)
    #         x1 = int(keypoint.pt[0] - keypoint.size*1.5/2)
    #         x2 = int(keypoint.pt[0] + keypoint.size*1.5/2)
    #         y1 = int(keypoint.pt[1] - keypoint.size*1.5/2)
    #         y2 = int(keypoint.pt[1] + keypoint.size*1.5/2)
    #         #blobs = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
    #         mask = cv2.rectangle(mask,(x1,y1),(x2,y2),(255,255,255),-3)
    #         blobs = cv2.bitwise_and(mask, cv2.bitwise_not(img))
    # else:
    #     blobs = mask

    return blobs, keypoints

def binarize(image, thresh=(127,255)):
    min, max = thresh
    kernel = np.ones((5,5),np.uint8)
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, bwImage = cv2.threshold(gray, min, max, cv2.THRESH_BINARY)
    #bwImage = cv2.bitwise_not(bwImage)
    closing = cv2.morphologyEx(bwImage, cv2.MORPH_CLOSE, kernel)
    #opening = cv2.morphologyEx(bwImage, cv2.MORPH_OPEN, kernel)
    #cv2.bitwise_not(bwImage)
    return closing

def get_desired_object_template(image, windowName='ROI'):
    im = copy.deepcopy(image)
    roi = cv2.selectROI(windowName, im)
    img = im[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])]
    cv2.destroyAllWindows()
    # cross_corr_map = cv2.matchTemplate(image, img, method=cv2.TM_CCOEFF_NORMED)
    # _, max_val, _, _ = cv2.minMaxLoc(cross_corr_map)
    return img 

def locate_object_in_image(image, obstacle_image, confidence = 0.85, image_offset=(0,0), SOURCE_ROI=None):

    obstacle_location = []
    h, w, _ = obstacle_image.shape
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, method=cv2.TM_CCOEFF_NORMED)
    
    thresh = confidence

    maxVal = 1.0
    # img = np.copy(image)
    # loc = np.where( cross_corr_map >= thresh)
    # for pt in zip(*loc[::-1]):
    #     img = cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    #     center = [pt[0]+w//2+image_offset[0], pt[1]+h//2+image_offset[1]]
    #     img = cv2.circle(img, (pt[0]+w//2+1, pt[1]+h//2+1), 3, (0,0,255), -1)
    #     obstacle_location.append(center)
    try:
        while maxVal > thresh:

            _, maxVal, _, max_loc = cv2.minMaxLoc(cross_corr_map)
            
            if maxVal > thresh:
                cross_corr_map[max_loc[1]-h//2:max_loc[1]+h//2+1, max_loc[0]-w//2:max_loc[0]+w//2+1] = 0
                if SOURCE_ROI is not None:
                    if ((max_loc[0] > int(np.amax(SOURCE_ROI[:,0])) or 
                        max_loc[0] < int(np.amin(SOURCE_ROI[:,0]))) or
                        (max_loc[1] > int(np.amax(SOURCE_ROI[:,1])) or 
                        max_loc[1] < int(np.amin(SOURCE_ROI[:,1])))):
                        continue
                img = cv2.rectangle(image, (max_loc[0], max_loc[1]), (max_loc[0]+w+1, max_loc[1]+h+1), (0,255,0), 2)
                img = cv2.circle(img, (max_loc[0]+w//2+1, max_loc[1]+h//2+1), 3, (0,0,255), -1)
                center = [max_loc[0]+w//2+image_offset[0], max_loc[1]+h//2+image_offset[1]]
                obstacle_location.append(center)
            # threshold += 0.01
            else:
                img = image
    except Exception as error:
        print(error)
        return None
    return img, obstacle_location

def extract_features_sift(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    ### This code works for openCV4 >= 4.4.0
    ### For other versions it should be "sift = cv2.xfeatures2d.SIFT_create()"
    ### START CODE HERE ###
    sift = cv2.SIFT_create()
    kp = sift.detect(image,None)

    #to compute the descriptor of the keypoints found in with sift
    kp, des = sift.compute(image, kp)

    ### END CODE HERE ###

    return kp, des

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:def match_features_sift(des1, des2):
    """
    display = cv2.drawKeypoints(image, kp, None)
    if isnotebook():
        #plot the image
        plt.imshow(display)
    #plt.imshow(display)
    return display


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    ### START CODE HERE ###
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    match = flann.knnMatch(des1,des2,k=2)

    ### END CODE HERE ###

    return match

def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []

    ### START CODE HERE ###
    for m,n in match:
        if m.distance < dist_threshold*n.distance:
            filtered_match.append(m)
    ### END CODE HERE ###

    return filtered_match

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    if isnotebook():
        #plot the image
        plt.figure(figsize=(16, 6), dpi=100)
        plt.imshow(image_matches)
    return image_matches

def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix

    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system

    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    objectpoints = []

    ### START CODE HERE ###
    for m in match:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        if depth1 is not None:
            Z = depth1[int(y1), int(x1)]
            if Z < 3.4028235e+38:
                scaled_coord = np.linalg.inv(k)@np.array([x1, y1, 1])
                cali_coord = (float(Z)/scaled_coord[2])*scaled_coord
                objectpoints.append(cali_coord)
                image1_points.append([x1, y1])
                image2_points.append([x2, y2])
            else:
                continue
        else:
            image1_points.append([x1, y1])
            image2_points.append([x2, y2])

    if depth1 is not None:
        image1_points = np.float32(image1_points)
        image2_points = np.float32(image2_points)
        objectpoints = np.float32(objectpoints)

        _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, image2_points, k, None,  iterationsCount= 2000, flags=cv2.SOLVEPNP_EPNP)

        rmat, _ = cv2.Rodrigues(rvec)
        #print(rvec)
    else:
        image1_points = np.float64(image1_points)
        image2_points = np.float64(image2_points)
        E, mask =cv2.findEssentialMat(image1_points, image2_points, k, cv2.RANSAC, 0.999, 1.0)
        cv2.recoverPose(E, image1_points, image2_points, k, rmat, tvec, mask);
    ### END CODE HERE ###

    return rmat, tvec, image1_points, image2_points

if __name__ == '__main__':
    print('This will get executed only if')
    print('the module is invoked directly.')
    print('It will not run when this module is imported')
    #my_function()
