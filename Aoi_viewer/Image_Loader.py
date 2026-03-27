import h5py
import numpy as np
import matplotlib.pyplot as plt
import bm3d
import time
from skimage.feature import blob_dog
from math import sqrt
from skimage.exposure import rescale_intensity
import os
from tqdm import tqdm
import scipy.ndimage
import cv2
from Blob import Blob
import os




class Image_Loader():
    
    def __init__(self, n_pro, thres, path, g_length, r_length, b_length, g_start, r_start, b_start, bac_mode, camera_size=1024):
        self.camera_size = camera_size # Save it to the class
        self.thres = thres
        self.path = path
        self.path_g = path+r'\g'
        self.path_b = path+r'\b'     
        self.path_r = path+r'\r'
        self.g_length = g_length 
        self.r_length = r_length 
        self.b_length = b_length 
        self.g_start = int(g_start)
        self.r_start = int(r_start)
        self.b_start = int(b_start)
        self.n_pro = n_pro
        self.bac_mode = bac_mode
        self.tic = None
        self.dcombined_image = None
        self.dframe = None
        self.image_g = None
        self.image_r = None
        self.image_b = None
        self.M = None
        self.Mb = None
        self.b_exists = None
        self.r_exists = None
        self.bac_b = None
        self.bac = None
        self.cpath = None
        
        
    def gaussian_peaks(self, offy, offx):
        gaussian_filter = np.zeros((7,7), dtype = np.float32)
        offy = np.round(offy, 2)
        offx = np.round(offx, 2)

        for i in range (-3, 4): 
                for j in range (-3, 4):
                    dist = 0.3 * ((i - offy)**2 + (j- offx)**2)
                    gaussian_filter[i+3][j+3] = np.exp(-dist)
        return gaussian_filter
    
    def cal_bac(self, image, nframes, q = 0.5):
        bac = np.zeros((nframes, image.shape[1], image.shape[2]))
        for bt in tqdm(range(0,nframes)):
                bac_temp = image[bt]
                bw = 16 
                aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
                
                for i in range(0,self.height, bw): #0~480
                    for j in range(0,self.width, bw): #0~480
                            aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw], 0.5),1)

                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac[bt] = aves

        return np.average(bac, axis=0)
    
    def cal_bac_med(self, image, size = 31, fsc = None, fsc_anchor = None, fsc_total = None):
        max = np.quantile(image, 0.9)
        min = np.min(image)
        image = (image - min) / (max - min) * 255
        image_8 = np.clip(image, 0, 255).astype(np.uint8)
        aves = np.zeros_like(image)
        for bt in tqdm(range(image.shape[0])):
            bac_temp = image_8[bt]
            ave = cv2.medianBlur(bac_temp, size)
            aves[bt] = ave /255 * (max-min) + min
            try:
                fsc.set("load_progress", str(fsc_anchor + (bt / image.shape[0] / fsc_total)))
            except:
                pass
        return aves


    def affine(self, x,y,M):
        x1 = M[0][0] * x + M[0][1] * y + M[0][2]
        y1 = M[1][0] * x + M[1][1] * y + M[1][2]
        
        return [y1, x1]
    
    def plot_circled(self, blobs_dog):
        fig = plt.figure()   
        ax = fig.add_subplot()
        ax.imshow(self.dcombined_image,cmap='Greys_r')
        
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='white', linewidth=0.5, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()

        plt.tight_layout()
        
        
        self.cpath = os.path.join(self.path,r'circled')
        os.makedirs(self.cpath, exist_ok = True)
        plt.savefig(self.cpath+f'\\circle_{self.n_pro}.tif',dpi=300)
        plt.close()

    def expand_bac(self, bac, length):
        bac = np.expand_dims(bac, 0)
        bac = np.repeat(bac, length, axis = 0)
        return bac
    
    def round_cord(self, coord):
        return np.concatenate(((coord[:6] + np.round(coord[6:])), (coord[6:] - np.round(coord[6:]))))
    

    def cal_time_g(self, path, start, length):
        file = h5py.File(path +r'\header.mat','r')
        time = np.array(file[r'/vid/ttb'][:]).reshape(-1).astype(float)
        time_n = np.cumsum(np.diff(np.concatenate(([time[0]], time))))*0.001
        time_n = time_n[start:start+length]

        return time_n, time[start]
    
    def cal_time(self, path, start, length, first):
        file = h5py.File(path +r'\header.mat','r')
        time = np.array(file[r'/vid/ttb'][:]).reshape(-1).astype(float)
        time_n = np.cumsum(np.diff(np.concatenate(([first], time))))*0.001
        time_n = time_n[start:start+length]

        return time_n   
    
    def load_image(self, fsc = None):
        
        self.tic = time.perf_counter()
        
        path = self.path
        path_g = self.path_g
        path_r = self.path_r
        path_b = self.path_b
        print(r'processing channel:')
        if os.path.exists(path_g) == True:
            g_exists = 1
            print(r'green')
        else:
            g_exists = 0

        if os.path.exists(path_b) == True:
            b_exists = 1
            print(r'blue')
        else:
            b_exists = 0

        if os.path.exists(path_r) == True:
            r_exists = 1
            print(r'red')
        else:
            r_exists = 0
            
        self.r_exists = r_exists
        self.b_exists = b_exists
        self.g_exists = g_exists
        r_finished = 0
        g_finished = 0
        b_finished = 0
        fsc_total = np.sum(r_exists + b_exists + g_exists)
        bac_mode = self.bac_mode 

        
        nframes_true = 0
        self.width = self.camera_size
        self.height = self.camera_size
        filenumber = [0]
        first = None

        #g_exist?  
        time_g = np.zeros(10)
 
        if  g_exists == 1:
            
            file = h5py.File(path_g+r'\header.mat','r')
            nframes_true = int(file[r'/vid/nframes'][0][0])
            self.width=int(file[r'/vid/width/'][0][0])
            self.height=int(file[r'/vid/height/'][0][0])
            filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
            gfilename = str(filenumber[0]) + '.glimpse'
            gfile_path = path_g+r'\\'+gfilename
            time_g, first = self.cal_time_g(path_g, self.g_start, self.g_length)
            image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height, self.width)))


            for n in tqdm(range(1, 10)):
                try:
                    gfilename = str(n) + '.glimpse'
                    gfile_path = path_g+r'\\'+gfilename
                    size = os.path.getsize(gfile_path)
                    if size>0:
                        image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                        image_g = np.concatenate((image_g, image_g_1))
                except:
                    pass
    
            image_g = image_g + 2**15
            print(f'Calculating g Backgrounds with mode {bac_mode}') 
            fsc_anchor = 0
            self.bac_g = self.cal_bac_med(image_g, 27, fsc, fsc_anchor, fsc_total) 
            

        
        #r_exist?
        time_r = np.zeros(10)
        
        if  r_exists == 1:
            file = h5py.File(path_r+r'\header.mat','r')
            rfilename = str(filenumber[0]) + '.glimpse'
            rfile_path = path_r+r'\\'+rfilename
            nframes_true = int(file[r'/vid/nframes'][0][0])
            if first == None:
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_r, first = self.cal_time_g(path_r, self.r_start, self.r_length)
            else:
                time_r = self.cal_time(path_r, self.r_start, self.r_length, first)
            image_r = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width))) 
            for n in range (1, 10):
                try:
                    rfilename = str(n) + '.glimpse'
                    rfile_path = path_r + r'\\'+rfilename
                    size = os.path.getsize(rfile_path)
                    if size>0:
                        image_r_1 = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                        image_r = np.concatenate((image_r, image_r_1))
                except:
                    pass
            image_r = image_r + 2**15

            print(f'Calculating r Backgrounds with mode {bac_mode}')
            fsc_anchor = (g_finished + r_finished) / (g_exists + r_exists)  
            self.bac_r = self.cal_bac_med(image_r, 27, fsc, fsc_anchor, fsc_total) #smaller
            try:
                fsc.set("load_progress", '0')
            except:
                pass
        
            
        #b_exist?    
        time_b = np.zeros(10)

        if  b_exists == 1:
            print(f'Calculating b Backgrounds with mode {bac_mode}')  
            file = h5py.File(path_b+r'\header.mat','r')
            bfilename = str(filenumber[0]) + '.glimpse'
            bfile_path = path_b+r'\\'+bfilename
            nframes_true = int(file[r'/vid/nframes'][0][0])
            if first == None:   
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_b, first = self.cal_time_g(path_b, self.b_start, self.b_length)
            else:
                time_b = self.cal_time(path_b, self.b_start, self.b_length, first)
            image_b = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height, self.width)))
            
            for n in range (1, 10):
                try:
                    bfilename = str(n) + '.glimpse'
                    bfile_path = path_b+r'\\'+bfilename
                    size = os.path.getsize(bfile_path)
                    if size>0:
                        image_b_1 = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                        image_b = np.concatenate((image_b, image_b_1))
                except:
                    pass


            image_b = image_b + 2**15   
            fsc_anchor = (g_finished + r_finished + b_finished) / (g_exists + r_exists + b_exists)  
            self.bac_b = self.cal_bac_med(image_b, 27, fsc, fsc_anchor, fsc_total) 
        ###



        #rescale image intensity and remove background

        self.image_g = image_g
        self.image_r = image_r
        self.image_b = image_b
        self.b_exists = b_exists
        self.g_exists = g_exists
        self.r_exists = r_exists
        self.time_g = time_g
        self.time_r = time_r
        self.time_b = time_b
        
        return  time_g, time_r, time_b, nframes_true
    
    
    
    def gen_dimg(self, anchor, mpath, maxf = 420, minf = 178, laser = 'green', average_frame = 20):
        
        if mpath == None:
            mpath = self.mpath
        self.mpath = mpath

        dframe_g = 0
        dframe_b = 0
        dframe_r = 0

        if  (self.r_exists == 1):
            end = min(self.image_r.shape[0], anchor + average_frame)
            start = max(0, end - average_frame)
            frame_r = np.average(self.image_r[start:end], axis = 0)
            frame_r = rescale_intensity(frame_r,in_range = (minf,maxf), out_range=np.ubyte)
            dframe_r = bm3d.bm3d(frame_r, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)
        
        if  (self.g_exists == 1):
            end = min(self.image_g.shape[0], anchor + average_frame)
            start = max(0, end - average_frame)
            frame_g = np.average(self.image_g[start:end], axis = 0)
            frame_g = rescale_intensity(frame_g, in_range = (minf,maxf), out_range = np.ubyte)
            dframe_g = bm3d.bm3d(frame_g, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)


        if  (self.b_exists == 1):
            end = min(self.image_b.shape[0], anchor + average_frame)
            start = max(0, end - average_frame)
            frame_b = np.average(self.image_b[start:end], axis = 0)
            frame_b = rescale_intensity(frame_b,in_range = (minf,maxf), out_range = np.ubyte)
            dframe_b = bm3d.bm3d(frame_b, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)



        laser_dict = {'green' :  dframe_g,
                    'blue' :  dframe_b,
                    'red' :  dframe_r
                    }

        print(laser)
        dframe = laser_dict[laser] 


        #combine two channel image
        self.M = np.load(mpath + r'\map_g_r.npy')
        self.Mb = np.load(mpath + r'\map_g_b.npy')

        print(dframe)

        split_1 = int(self.camera_size / 3)          # e.g., 170 for 512, or 341 for 1024
        split_2 = int((self.camera_size / 3) * 2)    # e.g., 341 for 512, or 682 for 1024
        # May be need change

        left_image  = dframe[0:self.height, 0:split_1]
        right_image = dframe[0:self.height, (split_1 + 1):split_2]
        blue_image  = dframe[0:self.height, (split_2 + 1):self.camera_size]
        

        rows, cols = right_image.shape
        
        left_image_trans = cv2.warpAffine(left_image, self.M, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
        blue_image_trans = cv2.warpAffine(blue_image, self.Mb, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
            
        
        dcombined_image = (right_image + left_image_trans + blue_image_trans)
        self.dcombined_image = dcombined_image

    
        self.dframe = dframe

        if self.b_exists:
            self.dframe_b = dframe_b
        else:
            self.dframe_b = dframe

        if self.g_exists:
            self.dframe_g = dframe_g
        else:
            self.dframe_g = dframe
        
        if self.r_exists:
            self.dframe_r = dframe_r
        else:
            self.dframe_r = dframe

        return dframe, dcombined_image


    
    
    def det_blob(self, plot = False, fsc = None, thres = None, r = 3, ratio_thres = 1.3):
        if thres != None:
            self.thres = thres

        print('Finding blobs')      
        blobs_dog = blob_dog(self.dcombined_image, min_sigma= (r-1) /sqrt(2), max_sigma = (r) /sqrt(2), threshold = self.thres, overlap = 0.8, exclude_border = 2)
        sorted_indices = np.lexsort((blobs_dog[:, 1], blobs_dog[:, 0]))
        blobs_dog = blobs_dog[sorted_indices]
        print(f'Found {blobs_dog.shape[0]} preliminary blobs')

    

        if plot == True:
            self.plot_circled(blobs_dog)

        blob_list = []



        try:
                fsc.set("cal_progress", str(0))
        except:
                pass
        
        self.cpath=os.path.join(self.path,r'circled')
        
        for i, raw_blob in enumerate(tqdm(blobs_dog)):

            try:
                fsc.set("progress", str(i / (blobs_dog.shape[0]-1)))
            except:
                pass
            
            b = Blob(raw_blob, self.M, self.Mb)
            b.map_coord(image_shape = self.image_g.shape)
            b.check_bound(image_shape = self.image_g.shape)

            b.set_image(self.dframe_r, laser = 'red')
            b.set_image(self.dframe_g, laser = 'green')
            b.set_image(self.dframe_b, laser = 'blue')

            b.check_max(self.dcombined_image, ratio_thres)
            np.save(self.path + r'\\dcombined_image.npy', self.dcombined_image)


            if self.r_exists or self.g_exists or self.b_exists:
                b.gaussian_fit(ch = 'red')
            elif self.g_exists:
                b.gaussian_fit(ch = 'red', laser = 'green')
            elif self.b_exists:
                b.gaussian_fit(ch = 'red', laser = 'blue')
            
            if self.g_exists:
                b.gaussian_fit(ch = 'green')
            elif self.b_exists:
                b.gaussian_fit(ch = 'green', laser = 'blue')
    
            if self.b_exists:
                b.gaussian_fit(ch = 'blue')

            #b.check_fit(ratio_thres)

            if b.quality == 1:
                #coord_list.append(b.get_coord())
                blob_list.append(b)
                if plot == True:
                    b.plot_circle(self.cpath, self.dframe_g, self.dframe_b, i)
            
        print(f'Found {len(blob_list)} filterd blobs')
        return blob_list
    
    
    def cal_drift(self, blob_list, laser, use_ch, n_slices = None, interval = None, anchors = None):
        
        
        tot_coord_list_drift = []

        length_dict = {
            'red' : self.r_length,
            'green' : self.g_length,
            'blue' : self.b_length
        }
        length = length_dict[laser]

        if n_slices == None:
            if interval == None:
                if anchors == None:
                    raise Exception("Please provide either n_slices, interval, or anchors")
                else:
                    if isinstance(anchors, list):
                        anchors = anchors
                    else:
                        anchors = [anchors]
            else:
                if not isinstance(interval, int):
                    raise Exception("Please provide an interger interval.")
                anchors = np.arange(0, length - 10, interval)
                
        else:
            if not isinstance(n_slices, int):
                raise Exception("Please provide interger slices.")
            anchors = np.linspace(0, length-10, n_slices).astype(int)
        
        if not np.any(anchors):
                    raise Exception("Length too short!")

        for anchor in tqdm(anchors):
            images = self.gen_dimg(anchor, mpath = None, maxf = 420, minf = 178, laser = laser, plot = False)
            blob_list_drift = []
            coord_list_drift = []
            if use_ch == 'all':
                use_ch = 'red'
                image = images[1]
            else:
                image = images[0]
            
            for i, b in enumerate(blob_list):
                b.set_image(image = image, laser = laser)
                b.set_params(use_ch)
                b.check_bound()
                b.gaussian_fit(ch = use_ch, nfev = 10, laser = laser)    
                blob_list_drift.append(b)
                coord_list_drift.append(b.get_coord())
            blob_list = blob_list_drift
            tot_coord_list_drift.append(coord_list_drift)
            
        
        cont_coord_list_g = np.zeros((len(blob_list), self.g_length, 12))
        cont_coord_list_b = np.zeros((len(blob_list), self.b_length, 12))
        cont_coord_list_r = np.zeros((len(blob_list), self.r_length, 12))

        time_dict = {
            'red' : self.time_r,
            'green' : self.time_g,    
            'blue' : self.time_b
        }
        
        
        for start in range(anchors.shape[0]-1):
            end = start + 1
            for b in range(len(blob_list)):
                delta_yx = np.array([(tot_coord_list_drift[end][b][0] + tot_coord_list_drift[end][b][6]- tot_coord_list_drift[start][b][0] - tot_coord_list_drift[start][b][6]),
                        (tot_coord_list_drift[end][b][1] + tot_coord_list_drift[end][b][7]- tot_coord_list_drift[start][b][1] - tot_coord_list_drift[start][b][7])])
                dt = time_dict[laser][anchors[end]] - time_dict[laser][anchors[start]]
                d_yx = delta_yx / dt
                d_yx_r = d_yx
                d_yx_g = np.matmul(self.M[:, :2], d_yx.T)
                d_yx_b = np.matmul(self.Mb[:, :2], d_yx.T)
                d_yx_tot = np.array([0, 0, 0, 0, 0, 0, *d_yx_r, *d_yx_g, *d_yx_b])

                t_start = time_dict[laser][anchors[start]]
                t_end = time_dict[laser][anchors[end]]

                #r
                if self.r_exists:
                    r_start = np.searchsorted(time_dict['red'], t_start)
                    r_end = np.searchsorted(time_dict['red'], t_end)
                    cont_coord_list_r[b][0] = tot_coord_list_drift[0][b]
                    for t in range(r_start, r_end+1):
                        dt =  time_dict['red'][t] - t_start
                        cont_coord_list_r[b][t] = cont_coord_list_r[b][r_start] + d_yx_tot * dt
                        cont_coord_list_r[b][t] = self.round_cord(cont_coord_list_r[b][t])
                    if start == anchors.shape[0]-2:
                        for t in range(r_end+1, self.r_length):
                            dt =  time_dict['red'][t] - t_start
                            cont_coord_list_r[b][t] = cont_coord_list_r[b][r_start] + d_yx_tot * dt
                            cont_coord_list_r[b][t] = self.round_cord(cont_coord_list_r[b][t])
                
                #g 
                if self.g_exists:
                    g_start = np.searchsorted(time_dict['green'], t_start)
                    g_end = np.searchsorted(time_dict['green'], t_end)
                    cont_coord_list_g[b][0] = tot_coord_list_drift[0][b]
                    for t in range(g_start, g_end+1):
                        dt =  time_dict['green'][t] - t_start
                        cont_coord_list_g[b][t] = cont_coord_list_g[b][g_start] + d_yx_tot * dt
                        cont_coord_list_g[b][t] = self.round_cord(cont_coord_list_g[b][t])
                    if start == anchors.shape[0]-2:
                        for t in range(g_end+1, self.g_length):
                            dt =  time_dict['green'][t] - t_start
                            cont_coord_list_g[b][t] = cont_coord_list_g[b][g_start] + d_yx_tot * dt
                            cont_coord_list_g[b][t] = self.round_cord(cont_coord_list_g[b][t])


                #b
                if self.b_exists:
                    b_start = np.searchsorted(time_dict['blue'], t_start)
                    b_end = np.searchsorted(time_dict['blue'], t_end)
                    cont_coord_list_b[b][0] = tot_coord_list_drift[0][b]
                    for t in range(b_start, b_end+1):
                        dt =  time_dict['blue'][t] - t_start
                        cont_coord_list_b[b][t] = cont_coord_list_b[b][b_start] + d_yx_tot * dt
                        cont_coord_list_b[b][t] = self.round_cord(cont_coord_list_b[b][t])

                    if start == anchors.shape[0]-2:
                        for t in range(b_end+1, self.b_length):
                            dt =  time_dict['blue'][t] - t_start
                            cont_coord_list_b[b][t] = cont_coord_list_b[b][b_start] + d_yx_tot * dt
                            cont_coord_list_b[b][t] = self.round_cord(cont_coord_list_b[b][t])


        coord_list = {'red' : cont_coord_list_r,
                      'green' : cont_coord_list_g,
                      'blue' : cont_coord_list_b
                      }
        return coord_list


    def cal_intensity(self, coord_list, maxf = 35000, minf = 32946, fsc = None):
        
        print('Calcultating Intensities')

        total_blobs = len(coord_list)
        trace_gg = np.zeros((total_blobs, int(self.g_length)))
        trace_gr = np.zeros((total_blobs, int(self.g_length)))
        trace_rr = np.zeros((total_blobs, int(self.r_length)))
        trace_bb = np.zeros((total_blobs, int(self.b_length)))
        trace_bg = np.zeros((total_blobs, int(self.b_length)))
        trace_br = np.zeros((total_blobs, int(self.b_length)))

        b_snap = np.zeros((total_blobs, 3, self.b_length, 9, 9))
        g_snap = np.zeros((total_blobs, 2, self.g_length, 9, 9))
        r_snap = np.zeros((total_blobs, 1, self.r_length, 9, 9))

        if self.g_exists == 1:
            bac_g = self.bac_g
            image_g = (self.image_g - bac_g).astype(np.float32)
        
        if self.r_exists == 1:
            bac_r = self.bac_r
            image_r = (self.image_r - bac_r).astype(np.float32)
        
        if self.b_exists == 1:
            bac_b = self.bac_b
            image_b = (self.image_b - bac_b).astype(np.float32)


        self.cpath=os.path.join(self.path,r'circled')
        os.makedirs(self.cpath+f'\\{self.n_pro}', exist_ok=True)
        for blob_count, blob in enumerate(tqdm(coord_list)):

            try:
                fsc.set("cal_progress", str(blob_count / (len(coord_list)-1)))
            except:
                pass

            yr, xr, yg, xg, yb, xb, ymr, xmr, ymg, xmg, ymb, xmb = blob
            r = 3

            
            yr = int(yr)
            xr = int(xr)
            yg = int(yg)
            xg = int(xg)
            yb = int(yb)
            xb = int(xb)


            if self.r_exists ==1:
                srr = 2 * self.gaussian_peaks(ymr, xmr)

                trace_rr[blob_count]  = np.einsum('tyx, yx -> t', image_r[:, yr-r:yr+r+1,xr-r:xr+r+1], srr, optimize = False) 
                
                r_snap[blob_count][0] = self.image_r[:, yr-4:yr+4+1,xr-4:xr+4+1]


            if self.g_exists ==1:
                sgg = 2 * self.gaussian_peaks(ymg, xmg)
                sgr = 2 * self.gaussian_peaks(ymr, xmr)

                trace_gg[blob_count]  = np.einsum('tyx, yx -> t', image_g[:, yg-r:yg+r+1,xg-r:xg+r+1], sgg, optimize = False)
                trace_gr[blob_count]  = np.einsum('tyx, yx -> t', image_g[:, yr-r:yr+r+1,xr-r:xr+r+1], sgr, optimize = False) 

                g_snap[blob_count][0] = self.image_g[:, yg-4:yg+4+1,xg-4:xg+4+1]
                g_snap[blob_count][1] = self.image_g[:, yr-4:yr+4+1,xr-4:xr+4+1]

                    
              
            if  self.b_exists == 1:
                sbb = 2 *self.gaussian_peaks(ymb, xmb)
                sbg = 2 *self.gaussian_peaks(ymg, xmg)
                sbr = 2 *self.gaussian_peaks(ymr, xmr)

                trace_bb[blob_count] = np.einsum('tyx, yx -> t', image_b[:, yb-r:yb+r+1,xb-r:xb+r+1], sbb, optimize = False)
                trace_bg[blob_count] = np.einsum('tyx, yx -> t', image_b[:, yg-r:yg+r+1,xg-r:xg+r+1], sbg, optimize = False)
                trace_br[blob_count] = np.einsum('tyx, yx -> t', image_b[:, yr-r:yr+r+1,xr-r:xr+r+1], sbr, optimize = False) 

                b_snap[blob_count][0] = self.image_b[:, yb-4:yb+4+1,xb-4:xb+4+1]
                b_snap[blob_count][1] = self.image_b[:, yg-4:yg+4+1,xg-4:xg+4+1]
                b_snap[blob_count][2] = self.image_b[:, yr-4:yr+4+1,xr-4:xr+4+1]
             
        
        np.savez(self.path + r'\blobs.npz', b = b_snap, g = g_snap, r = r_snap, minf = minf, maxf = maxf, yr = yr, xr = xr, yg = yg, xg = xg, yb = yb, xb = xb)
        
        return trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, blob_count+1
    
        

        
        
