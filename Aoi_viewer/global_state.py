# global_state.py
import numpy as np
from aoi_figure import create_initial_figure

class GlobalState:
    def __init__(self, default_size=1024):
        # Initialize with the default size when the app first boots
        self.set_camera_size(default_size)

    def set_camera_size(self, size):
        """Resets the figure and all arrays to the specified camera size."""
        # Now it dynamically builds the shape based on the 'size' argument
        self.fig = create_initial_figure(np.zeros((1, size, size)), 0, 2000, 7)
        self.loader = None
        self.image_g = np.zeros((1, size, size))
        self.image_r = np.zeros((1, size, size))
        self.image_b = np.zeros((1, size, size))
        self.image_datas = np.zeros((1, size, size))
        
        # Reset all lists and parameters so old data doesn't corrupt the new view
        self.coord_list = []
        self.blob_list = []
        self.rem_list = []
        self.rem_list_blob = []
        self.dr = 1
        self.org_size = 1
        self.blob_disable = True
        self.fret_g = []

global_state = GlobalState()
