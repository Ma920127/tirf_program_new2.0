import plotly.express as px
import h5py
from Image_Loader import Image_Loader 
from scipy.ndimage import uniform_filter1d as uniform_filter
import numpy as np
import os
from FRET_kernel import Fret_kernel
import json
from dash.exceptions import PreventUpdate
from Blob import Blob
from global_state import global_state as gs

def cal(path):
    g_start = 0
    path_g = path + r'\g'
    try:
        file = h5py.File(path_g + r'\header.mat', 'r')
        nframes = int(file[r'/vid/nframes'][0][0])
        print(f'total g frames : {nframes}')
        g_length = int((nframes - g_start))
    except:
        g_length = 0

    r_start = 0
    path_r = path + r'\r'
    try:
        file = h5py.File(path_r + r'\header.mat', 'r')
        nframes = int(file[r'/vid/nframes'][0][0])
        print(f'total r frames : {nframes}')
        r_length = int((nframes - r_start))
    except:
        r_length = 0

    b_start = 0
    path_b = path + r'\b'
    try:
        file = h5py.File(path_b + r'\header.mat', 'r')
        nframes = int(file[r'/vid/nframes'][0][0])
        print(f'total b frames : {nframes}')
        b_length = int((nframes - b_start))
    except:
        b_length = 0    
    return g_length, r_length, b_length, g_start, r_start, b_start

def draw_blobs(fig, coord_list, r, reverse):
    color = '#1f77b4' if int(reverse) == 0 else 'yellow'

    # New add selectedpoints=None,
    try:
        fig.update_traces(x=coord_list[:, 1], y=coord_list[:, 0],
                          marker=dict(size=2 * r + 1, line=dict(width=2), color=color),
                          selectedpoints=None,
                          selector=dict(name='blobs_r'))
        fig.update_traces(x=coord_list[:, 3], y=coord_list[:, 2],
                          marker=dict(size=2 * r + 1, line=dict(width=2), color=color),
                          selectedpoints=None,
                          selector=dict(name='blobs_g'))
        fig.update_traces(x=coord_list[:, 5], y=coord_list[:, 4],
                          marker=dict(size=2 * r + 1, line=dict(width=2), color=color),
                          selectedpoints=None,
                          selector=dict(name='blobs_b'))
    except:
        fig.update_traces(x=[], y=[], marker=dict(size=2 * r + 1, line=dict(width=2)), customdata= [], selector=dict(name='blobs_r'))
        fig.update_traces(x=[], y=[], marker=dict(size=2 * r + 1, line=dict(width=2)), customdata= [], selector=dict(name='blobs_g'))
        fig.update_traces(x=[], y=[], marker=dict(size=2 * r + 1, line=dict(width=2)), customdata= [], selector=dict(name='blobs_b'))
    return fig

def update_fret_labels(fig, frame):
    if np.any(gs.fret_g):
        fret =  np.round(gs.fret_g[:, int(frame)], 3)
    else:
        fret = None
    fig.update_traces(customdata= fret, selector=dict(name='blobs_r'))
    fig.update_traces(customdata= fret, selector=dict(name='blobs_g'))
    return fig

def update_blobs_coords(blob_list, coord_list):
    for i, b in enumerate(blob_list):
        b.update_coord(coord_list[i])

def move_blobs(coord_list, selector, step, changed_id):
    if 'up' in changed_id:
        if selector == 'channel_r':   
            for i in range(coord_list.shape[0]):
                coord_list[i][0] -= step
        elif selector == 'channel_g':   
            for i in range(coord_list.shape[0]):
                coord_list[i][2] -= step   
        else:
            for i in range(coord_list.shape[0]):
                coord_list[i][4] -= step
    if 'down' in changed_id:
        if selector == 'channel_r':   
            for i in range(coord_list.shape[0]):
                coord_list[i][0] += step
        elif selector == 'channel_g':   
            for i in range(coord_list.shape[0]):
                coord_list[i][2] += step  
        else:
            for i in range(coord_list.shape[0]):
                coord_list[i][4] += step
    if 'left' in changed_id:
        if selector == 'channel_r':   
            for i in range(coord_list.shape[0]):
                coord_list[i][1] -= step
        elif selector == 'channel_g':   
            for i in range(coord_list.shape[0]):
                coord_list[i][3] -= step  
        else:
            for i in range(coord_list.shape[0]):
                coord_list[i][5] -= step
    if 'right' in changed_id:
        if selector == 'channel_r':   
            for i in range(coord_list.shape[0]):
                coord_list[i][1] += step
        elif selector == 'channel_g':   
            for i in range(coord_list.shape[0]):
                coord_list[i][3] += step  
        else:
            for i in range(coord_list.shape[0]):
                coord_list[i][5] += step

    return coord_list

def load_path(thres, path, fsc, camera_size=1024):
    time_params = cal(path)
    loader = Image_Loader(0, thres, path, *time_params, 1, camera_size=camera_size)
    image_datas = loader.load_image(fsc)
    image_width = loader.width
    if image_width != camera_size:
        raise ValueError(f"SIZE MISMATCH: Expected {camera_size}, got {image_width}")
        
    image_g = loader.image_g
    image_r = loader.image_r
    image_b = loader.image_b
    fsc.set("load_progress", '1')
    return loader, image_g, image_r, image_b, image_datas

# 偵測blob intensity 到hel0檔
def cal_blob_intensity(loader, coord_list, path, image_datas, maxf, minf, fsc):
    coord_lists = [coord_list]
    trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i = loader.cal_intensity(coord_lists[0], maxf, minf, fsc)
    if not os.path.exists(path + r'\raw'):
        os.makedirs(path + r'\raw')
    time_g, time_r, time_b, nframes = image_datas
    np.savez(path + r'\raw' + f'\\hel0', nframes=nframes, cnt=i, trace_gg=trace_gg, trace_gr=trace_gr,
             trace_rr=trace_rr, trace_bb=trace_bb, trace_bg=trace_bg, trace_br=trace_br,
             time_g=time_g, time_r=time_r, time_b=time_b)
    print('finished')

def cal_FRET_utils(path, ps, ow, snap_time_g, snap_time_b, red, red_time, red_intensity,
                   green, green_time, green_intensity, leakage_g, leakage_b, f_lag, lag_b,
                   fit, fit_b, GFP_plot, GFP_hist, fsc):
    proc_config = {
        'leakage_g': leakage_g,
        'leakage_b': leakage_b,
        'lag_g': f_lag,
        'lag_b': lag_b,
        'ti': (0,1800000),
        'snap_time_g': snap_time_g,
        'snap_time_b': snap_time_b,
        'path': path,
        'red': red,
        'red_intensity': red_intensity,
        'red_time': red_time,
        'green': green,
        'green_intensity': green_intensity,
        'green_time': green_time,
        'fit_text': True,
        'preserve_selected': ps,
        'overwrite': ow,
    }
    kernel = Fret_kernel(proc_config)
    fret_g = kernel.auto_fret(plot=0, fit=fit, fit_b=fit_b, GFP_plot=GFP_plot, GFP_hist = GFP_hist, fsc=fsc)
    gs.fret_g = fret_g

    
def load_config(num, subfolder):
    # 1. Get the exact folder where aoi_utils.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the path strictly from that base directory
    file_path = os.path.join(base_dir, "configs", str(subfolder), f"{num}.json")
    
    # 3. Load the file
    if not os.path.exists(file_path):
        print(f"❌ Still missing! Looked at: {file_path}")
        return None
        
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Corrupted JSON: {e}")
        return None
    

def save_config(num, config_data, subfolder="1024"):
    folder_path = os.path.join("configs", str(subfolder))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(folder_path, f"{num}.json")
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=4)


def to_dict(b):
    # Make a copy of the dictionary so we don't destroy the live object
    b_dict = b.__dict__.copy()
    b_dict['dframe_r'] = None
    b_dict['dframe_b'] = None
    b_dict['dframe_g'] = None
    b_dict['dcombined_image'] = None
    b_dict['params'] = None
    
    for k in b_dict.keys():
        if isinstance(b_dict[k], np.ndarray):
            b_dict[k] = b_dict[k].tolist()
    return b_dict

def save_aoi_utils(data, file):
    with open(file, "w") as outfile:
        json.dump(data, outfile, default=to_dict)

def load_aoi_utils(file):
    with open(file, 'rb') as f:
        json_object = json.load(f)
    blob_list = []
    for b_dict in json_object:
        b = Blob()
        b.read_dict(b_dict) 
        blob_list.append(b)
    return blob_list




# old version
# def to_dict(b):
#     b.dframe_r = None
#     b.dframe_b = None
#     b.dframe_g = None
#     b.dcombined_image = None    
#     b.params = None        
#     b = b.__dict__
#     for k in b.keys():
#         if isinstance(b[k], np.ndarray):
#             b[k] = b[k].tolist()
#     return b

# def save_config(num, config_data):
#     os.makedirs(r'configs', exist_ok=True)
#     with open(f'configs\\{num}.json', 'w') as fp:
#         json.dump(config_data, fp)

# def load_config(num, init=False):
#     try:
#         import os
#         print("Current working directory:", os.getcwd())
#         with open(f'configs\\{num}.json', 'r') as fp:
#             config_data = json.load(fp)
#         #return list(config_data.values())
#         return dict(config_data)
#     except:
#         print('fail')
#         raise PreventUpdate
