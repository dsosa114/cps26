### Import necessary libraries #####
import os
import numpy as np
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
plt.style.use('default')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':'12'})
rc('text', usetex=True)
plt.rcParams["mathtext.fontset"] = "cm"

import trimesh


class Arrow3D(FancyArrowPatch):
    """This class allow the user to create arrowprops on 3d-lines"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class PlotHelper():
    def __init__(self, home_dir:str = None, folder_name:str = None, supported_formats:list=['png']) -> None:
        """Some colors and line types preferred by Daniel Sosa."""
        ds_colors = {"clear blue":'#5B9BD5',
                "deep blue":'#4472C4', 
                "aqua":'#4DC58D', 
                "olive":'#70AD47', 
                "orange":'#ED7D31',
                "purple":'#7030a0', 
                "yellow":'#ffc000', 
                "red":'#c00000',
                'light green':'#00b050'}
        ds_linetype = {'dashdotdotted':(0, (3, 5, 1, 5, 1, 5)),
                    'densely dashdotdotted':(0, (3, 1, 1, 1, 1, 1)), 
                    'long dash':(5, (10, 3)), 
                    'densely dotted':(0, (1, 1))}  
        # Converting into list of tuple
        self.linetypes = [(k, v) for k, v in ds_linetype.items()]
        self.colors = [(k, v) for k, v in ds_colors.items()]
        self.colors_dict = ds_colors
        if home_dir is None:
            self.home = os.path.expanduser("~")
        else:
            self.home = home_dir
        self.folder = folder_name
        self.formats = supported_formats
        self.dir_dict = {}

    def hex_to_rgba(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 4], 16) for i in range(0, lv, lv // 4))
    
    def fexp(self, x,p):
        """a different kind of exponentiation for superquadric functions"""
        """ Adapted from Solina_1994_Volumetric Models in Computer Vision -- an Overview"""
        return (np.sign(x) * (np.abs(x)**p))

    def gen_sq(self, center=np.zeros(3), axes=np.ones(3), coeff=np.ones(3), R=np.eye(3)):
        """this module plots superquadratic surfaces with the given parameters"""
        """ Adapted from Solina_1994_Volumetric Models in Computer Vision -- an Overview"""
        # A, B, C = R.T@axes
        A, B, C = axes
        if coeff.size < 3:
            P, Q = coeff
        else:
            P, Q, _ = coeff

        phi, theta = np.mgrid[0:np.pi:80j, 0:2*np.pi:80j]
        x = A * (self.fexp(np.sin(phi),P)) * (self.fexp(np.cos(theta),Q)) #+ center[0]
        y = B * (self.fexp(np.sin(phi),P)) * (self.fexp(np.sin(theta),Q)) #+ center[1]
        z = C * (self.fexp(np.cos(phi),P)) #+ center[2]

        points = np.stack([x, y, z]).reshape(3, -1)
        points_transformed = R.T.dot(points) + np.expand_dims(center, axis=1)

        x_tr = points_transformed[0].reshape(phi.shape) #+ center[0]
        y_tr = points_transformed[1].reshape(phi.shape) #+ center[1]
        z_tr = points_transformed[2].reshape(phi.shape) #+ center[2]
        return x_tr , y_tr , z_tr
    
    @staticmethod
    def xstr(s):
        if s is None:
            return ''
        return str(s)

    def plot_create_directory_structure(self, folder_name:str, supported_formats:list=['png', 'svg', 'eps']):
        """Funtion to create directories for saving plots. All new directories are created in $HOME """
        self.folder = folder_name
        self.formats = supported_formats
        
        for f in self.formats:
            dir = os.path.join(self.home, self.folder, f)
            if not os.path.isdir(dir):
                print(f"{dir} not exists, proceeding to create it")
                try:
                    os.makedirs(dir)
                    print("Done. Directory created.")
                except Exception as e:
                    print(f"Something went wrong: {e}. No new directory was created.")
            else:
                print(f"The path already exists: {dir}")
            self.dir_dict[f] = dir
        
        return self.dir_dict
    
    # Convert points to mesh using trimesh
    def superquadric_to_mesh(self, filename='superquadric', save=True,**kwargs):
        # Generate the surface points
        x, y, z = self.gen_sq(**kwargs)
        # Flatten the points to create vertices
        vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        
        # Create faces for the mesh (assuming rectangular grid topology)
        faces = []
        rows, cols = x.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Each quad is divided into two triangles
                faces.append([i * cols + j, (i + 1) * cols + j, i * cols + (j + 1)])
                faces.append([(i + 1) * cols + j, (i + 1) * cols + (j + 1), i * cols + (j + 1)])
        
        faces = np.array(faces)
        
        # Create the trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if save:
            # Export the mesh to an STL file
            for format in self.formats:
                path = os.path.join(self.dir_dict[format], filename + '.' +format)
                mesh.export(path)
                print(f"Superquadric mesh saved to {path}")

        # Compute mass properties of the mesh
        inertia_tensor = mesh.moment_inertia
        center_of_mass = mesh.center_mass
        volume = mesh.volume
        mass = 1.0

        # Scale the inertia tensor by the mass (assuming uniform density)
        # The inertia tensor calculated by trimesh is unitless, so we multiply by mass
        inertia_tensor_scaled = inertia_tensor * (mass / volume)
        # utri = np.triu(inertia_tensor_scaled, 0)
        index = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
        inertia_vector = inertia_tensor_scaled[np.triu_indices_from(inertia_tensor_scaled)]
        inertia = dict(zip(index, inertia_vector))
        # Display results
        print(f"Inertia Tensor (scaled by mass):\n{inertia_tensor_scaled}")
        # print(f"Inertia Tensor upper triangle matrix:\n{utri}")
        print(f"Inertia Tensor values for urdf:\n{inertia}")

        print(f"Center of Mass: {center_of_mass}")
        print(f"Volume: {volume}")

        # generate_urdf(filename, mass, inertia)
        # # Number of faces in the mesh
        # num_faces = len(mesh.faces)

        # # Create an array of random colors (RGBA) for each face
        # face_colors = np.random.randint(0, 255, size=(num_faces, 4))
        
        # # Number of vertices in the mesh
        # num_vertices = len(mesh.vertices)

        # # Create an array of random colors (RGBA) for each vertex
        # vertex_colors = np.random.randint(0, 255, size=(num_vertices, 4))

        # # Assign colors to the vertices
        # mesh.visual.vertex_colors = vertex_colors

        # # Assign colors to the faces
        # mesh.visual.face_colors = face_colors
        # mesh.show()
        return mesh
    
    def save_plot(self, file_name:str, date:bool=False):
        if date:
            datestring = datetime.datetime.now().strftime("_%Y-%m-%d_%H%M%S")
            fname = file_name + datestring
        else:
            fname = file_name
        for f in self.formats:
            file = '.'.join((fname, f))
            # fig.tight_layout()
            plt.savefig(os.path.join(self.home, self.xstr(self.folder), f, file), format=f)
