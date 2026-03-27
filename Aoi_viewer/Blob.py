import numpy as np
import lmfit
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
import os

class Blob():
    
    def __init__(self, raw_blob = None, M = None, Mb = None):
        
        if np.any(raw_blob):
            self.org_y = int(raw_blob[0])
            self.org_x = int(raw_blob[1])
            self.r = 4

        self.coords = np.zeros((3, 2))

        self.M = M
        self.Mb = Mb
      

        self.shift = np.zeros((3, 2))

        params = lmfit.Parameters()
        params.add('centery', value = 4, min = 3, max = 5)
        params.add('centerx', value = 4, min = 3, max = 5)
        params.add('amplitude', value = 5000)
        params.add('sigmay', value = 3, min = 0, max = 6)
        params.add('sigmax', value = 3, min = 0, max = 6)
        self.params = params


        self.sum = np.zeros(3)
        self.redchi = np.zeros(3)
        self.rs = np.zeros(3)
        self.nfev = np.zeros(3)
        self.sigma = np.zeros((3, 2))
        self.center = np.zeros((3, 2))

        self.bnum = 0

        self.quality = 1

    def read_dict(self, kwargs):
        for key, value in kwargs.items():
            if isinstance(value, list): 
                value = np.array(value)        
            setattr(self, key, value)
       

    def affine(self, y, x, M, x_shift = 0):
        x1 = round(M[0][0] * x + M[0][1] * y + M[0][2]) + x_shift
        y1 = round(M[1][0] * x + M[1][1] * y + M[1][2])
        
        return [y1, x1]

    def map_coord(self, image_shape):
        """
        Dynamically maps coordinates based on the total width of the sensor.
        Works if image_shape is (H, W) or (Frames, H, W).
        """
        # Always get the very last dimension (Width)
        total_width = image_shape[-1]

        if total_width == 512:
            self.coords[0] = self.affine(self.org_y, self.org_x, self.M, x_shift = 0)
            self.coords[2] = self.affine(self.org_y, self.org_x, self.Mb, x_shift = 342)
            self.coords[1] = [round(self.org_y), round(self.org_x) + 171]
        elif total_width == 1024:
            self.coords[0] = self.affine(self.org_y, self.org_x, self.M, x_shift = 0)
            self.coords[2] = self.affine(self.org_y, self.org_x, self.Mb, x_shift = 684)
            self.coords[1] = [round(self.org_y), round(self.org_x) + 342]
            # Need change
            
        # old version
        # channel_w = total_width // 3
        
        # # Red Channel (Left)
        # self.coords[0] = self.affine(self.org_y, self.org_x, self.M, x_shift = 0)
        
        # # Blue Channel (Right) - Shift is exactly 2/3 of total width
        # self.coords[2] = self.affine(self.org_y, self.org_x, self.Mb, x_shift = channel_w * 2)
        
        # # Green Channel (Middle) - Shift is exactly 1/3 of total width
        # self.coords[1] = [round(self.org_y), round(self.org_x) + channel_w]


    def check_bound(self, image_shape):
        """
        image_shape is a tuple (height, width) e.g., (1024, 1024) or (512, 512)
        """
        h, w = image_shape[-2:]
        r = self.r
        if w == 512:
            # check original boundary
            if (self.org_x - r) < 1 or (self.org_x + r) > 170 or (self.org_y - r) < 1 or (self.org_y + r) > 511:
                self.quality = 0

            # check red
            if (self.coords[0][1] - r) < 2 or (self.coords[0][1] + r) > 170 or (self.coords[0][0] - r) < 2 or (self.coords[0][0] + r) > 510:
                self.quality = 0
            
            # check green
            if (self.coords[1][1] - r) < 173 or (self.coords[1][1] + r) > 340 or (self.coords[1][0] - r) < 2 or (self.coords[1][0] + r) > 510:
                self.quality = 0

            #check red
            if (self.coords[2][1] - r - 1) < 342 or (self.coords[2][1] + r + 1) > 511 or (self.coords[2][0] - r) < 2 or (self.coords[2][0] + r) > 510:
                self.quality = 0

        elif w == 1024:
            if (self.org_x - r) < 1 or (self.org_x + r) > 340 or (self.org_y - r) < 1 or (self.org_y + r) > 1023:
                self.quality = 0

            # check red
            if (self.coords[0][1] - r) < 2 or (self.coords[0][1] + r) > 340 or (self.coords[0][0] - r) < 2 or (self.coords[0][0] + r) > 1022:
                self.quality = 0
            
            # check green
            if (self.coords[1][1] - r) < 342 or (self.coords[1][1] + r) > 682 or (self.coords[1][0] - r) < 2 or (self.coords[1][0] + r) > 1022:
                self.quality = 0

            #check blue
            if (self.coords[2][1] - r - 1) < 684 or (self.coords[2][1] + r + 1) > 1023 or (self.coords[2][0] - r) < 2 or (self.coords[2][0] + r) > 1022:
                self.quality = 0
            #Maybe need change 

        # old version
        # channel_w = w // 3
        # # 1. Check Original/Red Channel Boundary (0 to channel_w)
        # if (self.org_x - r) < 1 or (self.org_x + r) > (channel_w - 1) or \
        #    (self.org_y - r) < 1 or (self.org_y + r) > (h - 1):
        #     self.quality = 0

        # # 2. Check Red Mapping (Coords[0])
        # if (self.coords[0][1] - r) < 2 or (self.coords[0][1] + r) > (channel_w - 1) or \
        #    (self.coords[0][0] - r) < 2 or (self.coords[0][0] + r) > (h - 2):
        #     self.quality = 0
        
        # # 3. Check Green Mapping (Coords[1]) (channel_w to 2*channel_w)
        # if (self.coords[1][1] - r) < (channel_w + 2) or (self.coords[1][1] + r) > (2 * channel_w - 2) or \
        #    (self.coords[1][0] - r) < 2 or (self.coords[1][0] + r) > (h - 2):
        #     self.quality = 0

        # # 4. Check Blue Mapping (Coords[2]) (2*channel_w to w)
        # if (self.coords[2][1] - r - 1) < (2 * channel_w + 2) or (self.coords[2][1] + r + 1) > (w - 1) or \
        #    (self.coords[2][0] - r) < 2 or (self.coords[2][0] + r) > (h - 2):
        #     self.quality = 0


    def check_max(self, dcombined_image, ratio_thres):
        if self.quality == 0:
            return None
        r = 3
        aoi = dcombined_image[self.org_y - r : self.org_y + r + 1, self.org_x  - r : self.org_x + r + 1]
        
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage import io, filters, measure, morphology

        def classify_blob_shape(image, ratio_thres = 1.25):
            """
            Classifies a white blob in 'image' as either 'elongated' or 'circular'
            based on the ratio of major_axis_length to minor_axis_length.
            
            Parameters:
                image (2D numpy array): Grayscale input image with a white blob on a dark background
                ratio_threshold (float): If the major/minor axis ratio is above this, we call it 'elongated'
            
            Returns:
                shape_type (str): 'elongated' or 'circular'
                aspect_ratio (float): The measured major/minor axis ratio
            """
            # Step 1: Threshold the image to get a binary mask
            # If your images are well-contrasted, you can use Otsu's method or a fixed threshold
            if image.max() == image.min():
                return "no blob", 10.0  # Image is perfectly flat, skip Otsu
            thresh_val = filters.threshold_otsu(image)
            binary = image > thresh_val
            
            # Step 2: Remove small noise (optional), e.g. with morphological opening or a remove_small_objects
            binary_clean = morphology.remove_small_objects(binary, min_size=5)
            
            # Step 3: Label and measure region properties
            labeled = measure.label(binary_clean)
            props = measure.regionprops(labeled)
            
            # Assuming there's only one large blob in the image, we take the largest one
            # or the first region if there's exactly one. Adjust logic if multiple blobs are present.
            if not props:
                return "no blob", 10.0

            # Find region with the maximum area
            region = max(props, key=lambda r: r.area)
            
            # Extract major and minor axes
            major_axis = region.major_axis_length
            minor_axis = region.minor_axis_length
            
            # Avoid division by zero if minor_axis is zero
            if minor_axis == 0:
                return "elongated", float('inf')
            
            aspect_ratio = major_axis / minor_axis
            
            # Step 4: Classify using the aspect ratio
            if aspect_ratio > ratio_thres:
                shape_type = "elongated"
            else:
                shape_type = "circular"
            
            return shape_type, aspect_ratio
                
        self.aspect_ratio = classify_blob_shape(aoi)[1]
        h = 60  # h parameter: adjust based on how deep the valley is between peaks
        h_max = h_maxima(aoi, h)
        c_num = np.sum(h_max)
        
        if  self.aspect_ratio > ratio_thres or c_num > 1:
            self.quality = 0
        
    def set_image(self, image, laser):
        if laser == 'red':
            self.dframe_r = image
        elif laser == 'green':
            self.dframe_g = image
        elif laser == 'blue':
            self.dframe_b = image


    def set_params(self, ch):
        channel_dict = {
            'red' : 0,
            'green' : 1,
            'blue' : 2
        }  

        ch = channel_dict[ch] 
        self.params['centery'].set(value = 4 + self.shift[ch][0], min = 4 + self.shift[ch][0] - 0.5, max = 4 + self.shift[ch][0] + 0.5)
        self.params['centerx'].set(value = 4 + self.shift[ch][1], min = 4 + self.shift[ch][1] - 0.5, max = 4 + self.shift[ch][1] + 0.5)
        self.params['amplitude'].set(value = self.sum[ch], vary = False)
        self.params['sigmay'].set(value = self.sigma[ch][0], vary = False)
        self.params['sigmax'].set(value = self.sigma[ch][1], vary = False)

        
    def gaussian_fit(self, ch, nfev = 150, laser = None):
        if self.quality == 0:
            return None
        
        channel_dict = {
            'red' : 0,
            'green' : 1,
            'blue' : 2
        }   
        
        laser_dict = {'red' :  self.dframe_r,
                    'green' :  self.dframe_g,
                    'blue' :  self.dframe_b}
        
        if laser == None:
            laser = ch
        ch = channel_dict[ch] 
        image = laser_dict[laser]
        thres = np.median(image)*81
        
        r = 4
        y = int(np.round(self.coords[ch][0]))
        x = int(np.round(self.coords[ch][1]))
        z = image[y-r:y+r+1,x-r:x+r+1].flatten()
        sum = np.sum(z)

        if sum > thres: 
            yr = np.arange(0, 9)
            xr = np.arange(0, 9)
            yr, xr = np.meshgrid(yr, xr, indexing='ij')
            yr = yr.flatten()
            xr = xr.flatten()
            
            model = lmfit.models.Gaussian2dModel()

            result = model.fit(z, y = yr, x = xr, params = self.params, max_nfev = nfev)
            self.redchi[ch] = result.redchi
            self.rs[ch] = result.rsquared
            self.sum[ch] = result.best_values['amplitude']
            self.nfev[ch] = result.nfev
            self.sigma[ch] = [result.best_values['sigmay'], result.best_values['sigmax']]
            self.center[ch] = [result.best_values['centery'], result.best_values['centerx']]

            if self.redchi[ch] > 1:
                self.coords[ch] = [y + result.best_values['centery'] - 4, x + result.best_values['centerx'] - 4]
                self.shift[ch] = self.coords[ch] - np.round(self.coords[ch])
                self.coords[ch] = np.round(self.coords[ch])
                

    def check_fit(self, ratio_thres):
        for ch in range(0, 3):
            min_sig = min(self.sigma[ch])
            # Prevent division by zero if the fit hit the absolute 0 lower bound
            if min_sig == 0:
                min_sig = 1e-9 
            #c1 = (self.redchi[ch] <  redchi_thres or (self.sigma[ch][0] < 3.6 and self.sigma[ch][1] < 3.6)) 
            #c2 = self.redchi[ch] <  redchi_thres * 2
            c3 = ((max(self.sigma[ch]) / min_sig) < (max(1, ratio_thres) / min(1, ratio_thres))) or (not np.any(self.sigma[ch]))

            if not(c3):
                self.quality = 0


    def plot_circle(self, cpath, dframe_g, dframe_b, i):
        r = 4
        yr = int(self.coords[0][0])
        xr = int(self.coords[0][1])
        yg = int(self.coords[1][0])
        xg = int(self.coords[1][1])
        yb = int(self.coords[2][0])
        xb = int(self.coords[2][1])

        fig, axes = plt.subplots(1,3)
        plt.axis('off')
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_yticks([])

        fig.suptitle(f"bnum: {self.bnum}", fontsize=12)

        axes[0].imshow(dframe_g[yr-r:yr+r+1, xr-r:xr+r+1], cmap='Greys_r', vmin=0, vmax=128)        
        axes[1].imshow(dframe_g[yg-r:yg+r+1, xg-r:xg+r+1], cmap='Greys_r', vmin=0, vmax=128)
        axes[2].imshow(dframe_b[yb-r:yb+r+1, xb-r:xb+r+1], cmap='Greys_r', vmin=0, vmax=160) 
        os.makedirs(cpath, exist_ok=True)
        plt.savefig(cpath+f'\\{i}.tif', dpi=50) 
        plt.close()

    def get_coord(self):
        c = list(self.coords.flatten())
        m = list(self.shift.flatten())
        return c + m
    
    def update_coord(self, coords):
        self.coords = np.array(coords[:len(self.coords.flatten())]).reshape(self.coords.shape)
        self.shift = np.array(coords[len(self.coords.flatten()):]).reshape(self.shift.shape)
