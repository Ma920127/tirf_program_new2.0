import h5py
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import bm3d
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.exposure import rescale_intensity
import os
import lmfit
from tqdm import tqdm
import scipy.ndimage
import cv2
import time
from PIL import Image
import statistics
import numpy as np


class Glimpse_mapping:
    
    def __init__(self, path):
        
        self.path = path
        self.path_g = path + r'\\g'
        self.path_r = path + r'\\r'
        self.path_b = path + r'\\b'
    
    def map(self, mode, seg, threhold = 0.25, circled_image = False):
        
        self.seg = seg
        t_aoi25 = []
        t_aoi49 = []
        path_g = self.path_g
        path_r = self.path_r
        path_b = self.path_b
        
        if mode == 'g':
            path = path_g
            sw = 1
        elif mode == 'r':
            path = path_r
            sw = 0
        else:
            path = path_b
            sw = 2
            
        gaussian_peaks2 = np.zeros((3,3,7,7),dtype=np.float32)
        for k in range (0,3):
            for l in range (0,3):     
              offy = -0.5*float(k)
              offx = -0.5*float(1)
              
              for i in range (0, 7): 
                for j in range (0,7):
                  dist = 0.3 * ((float(i)-3.0+offy)**2 + (float(j)-3.0+offx)**2)
                  gaussian_peaks2[k][l][i][j]= 2.0*np.exp(-dist)
                  
        circle = np.zeros((11, 11), dtype=np.int16)
        circle[0] = [ 0,0,0,0,0,0,0,0,0,0,0]
        circle[1] = [ 0,0,0,0,1,1,1,0,0,0,0]
        circle[2] = [ 0,0,0,1,0,0,0,1,0,0,0]
        circle[3] = [ 0,0,1,0,0,0,0,0,1,0,0]
        circle[4] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[5] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[6] = [ 0,1,0,0,0,0,0,0,0,1,0]
        circle[7] = [ 0,0,1,0,0,0,0,0,1,0,0]
        circle[8] = [ 0,0,0,1,0,0,0,1,0,0,0]
        circle[9] = [ 0,0,0,0,1,1,1,0,0,0,0]
        circle[10]= [ 0,0,0,0,0,0,0,0,0,0,0]    
              

        #g
        file = h5py.File(path+r'\header.mat','r')
        nframes=int(file[r'/vid/nframes'][0][0])
        # nframes = 40

        width=int(file[r'/vid/width/'][0][0])
        height=int(file[r'/vid/height/'][0][0])

        filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
        offset=file[r'/vid/offset'][:].flatten().astype('int')

        frame=np.zeros((height,width), dtype= np.int16)
        ave_arr = np.zeros((height,width), dtype= np.float32)

        nframes = 10 #?????(very important)
            
        gfilename = str(filenumber[0]) + '.glimpse'
        gfile_path = path+r'\\'+gfilename
        image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))

        try:
            gfilename = str(1) + '.glimpse'
            gfile_path = path+r'\\'+gfilename
            image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))
            image_g = np.concatenate((image_g,image_g_1))
        except:
            pass
        

        # image_g = image_g + 2**15 (some data too large than np.dtype('>i2'))
        image_g = image_g.astype(np.int32) + 2**15

        
        # average 10 frame intensity to one picture
        for j in range(self.seg * 10, self.seg * 10+nframes):
            ave_arr= ave_arr + image_g[j]
        ave_arr = ave_arr/(nframes)
        frame = ave_arr

        # 2. Calculate the 5th percentile for a 7x7 neighborhood around every pixel
        bac = scipy.ndimage.percentile_filter(frame, percentile=5, size=7)

        # 3. Subtract this local background from the original image
        frame_subtracted = frame - bac

        # 4. Clip negative values to 0 (and optionally convert back to your original dtype)
        frame = np.clip(frame_subtracted, a_min=0, a_max=None)

        # # remove background(very important)
        # bg_intensity = np.percentile(frame, 5)
        # frame = frame - bg_intensity
        # frame = np.clip(frame, a_min=0, a_max=None)

        # define maxf and minf and rescale intensity to 0~255(by np.ubyte)
        maxf = np.quantile(frame, 0.997)
        print(f'max intensity = {maxf}')
        minf = np.min(frame)
        print(f'min intensity = {minf}')
        frame = rescale_intensity(frame,in_range=(minf,maxf),out_range=np.ubyte)

        temp1=frame

        # cut image to r/g/b
        cut_size = int(width / 3)
        print(f'cut_size = {cut_size}')
        left_image  = temp1[0:height, cut_size * sw + sw : cut_size * sw + cut_size]
        left_image1  = frame[0:height, cut_size * sw + sw :  cut_size * sw + cut_size]

        # Detect blob
        blobs_dog = blob_dog(left_image, min_sigma=1.5/sqrt(2),max_sigma=3.5/sqrt(2), threshold = threhold ,overlap = 0, exclude_border = 3)
        if len(blobs_dog) == 0:
            print("No blobs detected!")
            # Return empty arrays/zeros to prevent downstream crashing
            return blobs_dog, 0, 0, 0, 0
        
        #blobs_dog = blob_dog(left_image)
        # print(len(blobs_dog))

        # Transfer standardeviation to radius by mutiply 2 square root
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        

        # remove the blob that too close to the right side
        blobs_dog = blobs_dog[blobs_dog[:,1].squeeze()<(cut_size - 5)]
        print(f'selected_blob ={len(blobs_dog)}')


        for blob in blobs_dog:
            y, x, r = blob
            y, x = round(y), round(x)
            
            # Check 5x5
            r = 2
            aoi25 = left_image[y-r:y+r+1, x-r:x+r+1]
            if aoi25.shape == (5, 5): 
                t_aoi25.append(np.sum(aoi25))
                
            # Check 7x7
            r = 3
            aoi49 = left_image[y-r:y+r+1, x-r:x+r+1]
            if aoi49.shape == (7, 7):
                t_aoi49.append(np.sum(aoi49))


        mean25 = np.mean(t_aoi25)
        sd25 = np.std(t_aoi25)
        mean49 = np.mean(t_aoi49)
        sd49 = np.std(t_aoi49)

        # plot aoi choose 
        fig = plt.figure(figsize = (cut_size/5, height/5))
        ax = fig.add_subplot()
        ax.imshow(left_image,cmap='Greys_r')
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red',linewidth=4, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        plt.tight_layout()
        
        # plt.show()

        cpath=os.path.join(path,r'circled')
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        
        # Can save aoi circle image to check threshold valid
        if circled_image == True:
            # plt.savefig(cpath+f'\\circled_{mode}_circled.tif', dpi=left_image.shape[0])
            plt.savefig(cpath+f'\\circled_{mode}_{seg}_circled.tif', dpi=100)

        # the other processed image
        im = Image.fromarray(left_image1)
        im.save(cpath+f'\\circled_{mode}.tif')
        self.left_image = left_image
        #plt.show()
        #plt.close()
        
        return blobs_dog, mean25, sd25, mean49, sd49
    
    def get_image(self):
        return self.left_image



# # test_np.ubyte image_show
# cv2.imshow('image_g', left_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# original
# left_image  = temp1[0:height, 170 * sw + sw : 170 * sw + 170]
# left_image1  = frame[0:height, 170 * sw + sw :  170 * sw + 170]

# denoise by bm3d (not used)
# dframe= bm3d.bm3d(frame, 6, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
#dframe = frame
#plt.imshow(np.concatenate((frame,dframe),axis=1),cmap='Greys_r',vmin=0,vmax=128)
#plt.savefig(path+r'\\ave.tif',dpi=300)
# plt.close()

#bac=frame-dframe
#bac=rescale_intensity(bac,in_range=np.ubyte,out_range=(minf,maxf))
# plt.imshow(bac,cmap='Greys_r')
# plt.savefig(path+r'\\back.tif',dpi=300)
# plt.close()


# # 2. Show the image using matplotlib
# plt.figure(figsize=(8, 8)) # Set the window size
# # Use cmap='gray' for standard black & white microscopy look.
# # You can also try cmap='magma' or cmap='viridis' for false color.
# img_plot = plt.imshow(frame, cmap='gray')
# plt.show()




# # conduct image smoothing througth 3*3 nearby pixel(bac not use)
# temp1 = frame 
# temp1 =scipy.ndimage.filters.uniform_filter(temp1,size=3,mode='nearest')



# # Divide image to 16*16 square and calculate each min_intensity and storage to aves(calculate background) (bac not use)
# aves = np.zeros((int(height/16),int(width/16)), dtype= np.float32)
# for i in range(8,height+1,16):
#     for j in range(8,width,16):
#         aves[int((i-8)/16)][int((j-8)/16)] = np.round(np.amin(temp1[i-8:i+8,j-8:j+8]),1)


# # use aves to calculate background intensity  (bac not use)
# aves = scipy.ndimage.zoom(aves, 16,order=1)
# aves = scipy.ndimage.filters.uniform_filter(aves,size=21,mode='nearest')
# bac=aves