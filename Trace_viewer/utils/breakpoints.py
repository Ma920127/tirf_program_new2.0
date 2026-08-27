import numpy as np
import os
import time as rtime
import shutil

import numpy as np
from .smoothing import uf, sa, mf, sg

def breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode, polyorder=2):
    # --- NEW SMOOTHING MAPPER ---
    if smooth_mode == 'moving':
        sm = lambda t, l: uf(t, l)
    elif smooth_mode == 'strided':
        sm = lambda t, l: sa(t, l)
    elif smooth_mode == 'median':
        sm = lambda t, l: mf(t, l)
    elif smooth_mode == 'savgol':
        sm = lambda t, l: sg(t, l, polyorder=polyorder)
    else:
        sm = lambda t, l: uf(t, l) # fallback
    # ----------------------------

    trans = {
        0: 'fret_g', 1: 'fret_b', 2: 'b', 3: 'b', 4: 'b', 5: 'b',
        6: 'g', 7: 'g', 8: 'g', 9: 'r', 10: 'fret_g', 11: 'fret_b', 12: 'b', 13: 'g', 14: 'r'
    }
    confirm_reset_show = False
    channel_error_show = False

    # Smooth the time array for the given channel (if channel is provided)

    if 'dtime' in changed_id:
        if channel is not None:
            if smooth_mode == 'strided':
                smoothed_time = sm(time[channel], smooth)
            else:
                smoothed_time = time[channel]
        else:
            print(1)
            channel_error_show = True
            return bkps, mode, confirm_reset_show, channel_error_show

        if mode == 'Add':
            # Use the first element of the smoothed time array
            bkps[channel][i].append((0, smoothed_time[0]))
            bkps[channel][i] = sorted(bkps[channel][i])
        elif mode == 'Remove':
            try:
                bkps[channel][i].pop(0)
                bkps[channel][i] = sorted(bkps[channel][i])
            except:
                pass

    if 'etime' in changed_id:
        if channel is not None:
            if smooth_mode == 'strided':
                smoothed_time = sm(time[channel], smooth)
            else:
                smoothed_time = time[channel]
        else:
            channel_error_show = True
            return bkps, mode, confirm_reset_show, channel_error_show
            
        if mode == 'Add':
            # Use the last element from the smoothed time array
            bkps[channel][i].append((time[channel].shape[0]-1, smoothed_time[-1]))
            bkps[channel][i] = sorted(bkps[channel][i])
        elif mode == 'Remove':
            bkps[channel][i].pop(-1)
            bkps[channel][i] = sorted(bkps[channel][i])

    if 'graph.clickData' in changed_id:
        if isinstance(clickData, dict):
            c_num = clickData["points"][0]["curveNumber"]
            channel = trans[c_num]
            # Re-smooth time for the new channel context
            if smooth_mode == 'strided':
                smoothed_time = sm(time[channel], smooth)
            else:
                smoothed_time = time[channel]
            if mode == 'Add':
                if c_num < 10:
                    idx = clickData["points"][0]["pointNumber"]
                    
                    # For strided, the clicked point index is downsampled, so we must map it back 
                    # correctly to the smoothed time array. (Plotly returns the index of the drawn point).
                    idx_t = smoothed_time[idx]
                    
                    # Find the closest true index in the raw time array
                    true_idx = np.abs(time[channel] - idx_t).argmin()
                    
                    bkps[channel][i].append((true_idx, idx_t))
                    bkps[channel][i] = sorted(bkps[channel][i])
            elif mode == 'Remove':
                if 10 <= c_num <= 14:
                    if len(bkps[channel][i]) > 0:
                        # Find the actual time (x-coordinate) we clicked
                        click_x = clickData["points"][0]["x"]
                        
                        # Find which breakpoint time (x[1]) is mathematically closest to our click
                        diff = [abs(x[1] - click_x) for x in bkps[channel][i]]
                        closest_idx = diff.index(min(diff))
                        
                        # Delete that specific breakpoint
                        bkps[channel][i].pop(closest_idx)
                        
            elif mode == 'Except':
                if 10 <= c_num <= 14:
                    if len(bkps[channel][i]) > 0:
                        click_x = clickData["points"][0]["x"]
                        
                        diff = [abs(x[1] - click_x) for x in bkps[channel][i]]
                        closest_idx = diff.index(min(diff))
                        
                        # Keep only the closest breakpoint
                        bkps[channel][i] = [bkps[channel][i][closest_idx]]
                    
    if mode == 'Clear':
        mode = "Add"
        if channel is not None:
            bkps[channel][i] = []
    if mode == 'Clear All':
        for channel in bkps:
            bkps[channel][i] = []
        mode = "Add"
    if mode == 'Set All':
        mode = "Add"
        if channel is not None:
            for key in bkps:
                bkps[key][i] = bkps[channel][i]
    if mode == 'Reset':
        mode = "Add"
        confirm_reset_show = True
    if 'confirm-reset' in changed_id:
        for channel in bkps:
            for j in range(len(bkps[channel])):
                bkps[channel][j] = []
                
    return bkps, mode, confirm_reset_show, channel_error_show


def sl_bkps(changed_id, path, bkps, mode):
    if ('save_bkps' in changed_id) or (mode == 'Clear All'):
        mode = 'Add'
        try:
            seconds = rtime.time()
            t = rtime.localtime(seconds)
            shutil.copy(path + r'/breakpoints.npz', path + f'/breakpoints_backup_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.npz')
        except:
            print('No existing save file found.')
        if not path:
            raise ValueError("No folder path loaded — cannot save breakpoints. Tell 🐎")
        for key in bkps:
            bkps[key] = np.array(bkps[key], dtype=object)
        np.savez(path + r'/breakpoints.npz', **bkps)
        print('file_saved')
    if 'load_bkps' in changed_id:
        try:
            loaded_bkps = dict(np.load(path + r'/breakpoints.npz', allow_pickle=True))
            # Convert the loaded arrays back into Python lists
            for key in loaded_bkps:
                bkps[key] = loaded_bkps[key].tolist()
        except Exception as e:
            print(f"breakpoints.npz not found or corrupted at '{path}': {e}. Tell 🐎")
    return bkps


def find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g, 
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, 
             chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode, polyorder=2): # Added polyorder
    """
    Find change points in the signal corresponding to a given channel.
    """
    channel_error_show = False
    
    # --- NEW SMOOTHING MAPPER ---
    if smooth_mode == 'moving':
        sm = lambda t, l: uf(t, l)
    elif smooth_mode == 'strided':
        sm = lambda t, l: sa(t, l)
    elif smooth_mode == 'median':
        sm = lambda t, l: mf(t, l)
    elif smooth_mode == 'savgol':
        sm = lambda t, l: sg(t, l, polyorder=polyorder)
    else:
        sm = lambda t, l: uf(t, l) # fallback
    # ----------------------------
    
    # Select parameters based on changed_id
    if 'chp_find_0' in changed_id:
        channel = chp_channel_0
        mode = chp_mode_0
        comp = chp_comp_0
        thres = chp_thres_0
        target_mode = chp_target_0
    elif 'chp_find_1' in changed_id:
        channel = chp_channel_1
        mode = chp_mode_1
        comp = chp_comp_1
        thres = chp_thres_1
        target_mode = chp_target_1
    else:
        return bkps, channel_error_show

    # Choose the signal based on the channel
    trans = {
        'fret_g' : fret_g,
        'fret_b' : fret_b,
        'r' : rr,
        'g' : gg,
        'b' : bb
    }

    try:
        signal = trans[channel]
    except:
        channel_error_show = True
        return bkps, channel_error_show
   
    # Determine which trace indices to process
    if target_mode == 'current trace':
        i_list = [i]
    elif target_mode == 'all traces':
        i_list = np.arange(0, signal.shape[0])
    elif target_mode == 'all good':
        i_list = np.arange(0, signal.shape[0])
        i_list = i_list[select_list_g == 1]
    else:
        i_list = []

    # Apply smoothing on the chosen signal AND time
    smoothed_signal = sm(signal, smooth)
    if smooth_mode == 'strided':
        smoothed_time = sm(time[channel], smooth)
    else:
        smoothed_time = time[channel]

    # Process each trace index in i_list. 
    for j in i_list:
        # Determine target indices based on the comparison operator and threshold.
        if comp == 'bigger':
            target_indices = np.where(smoothed_signal[j] > thres)[0]
        elif comp == 'smaller':
            target_indices = np.where(smoothed_signal[j] < thres)[0]
        else:
            target_indices = []

        # If no valid points are found, log the event and choose a default target.
        if len(target_indices) < 1:
            print(f'No valid points found for trace {j} in channel {channel}.')
            target_t = time[channel][0]  
            target_index = len(time[channel]) - 1  
            bkps[channel][j].append((target_index, target_t))
            continue

        # Select the target index based on the mode.
        if mode == 'first':
            target_index = target_indices[0]
        elif mode == 'second':
            target_index = target_indices[1] if len(target_indices) >= 2 else target_indices[0]
        elif mode == 'previous':
            target_index = target_indices[0] - 1
            if target_index < 0:
                print(f'The first point meets the threshold for trace {j} in channel {channel}.')
                target_index = 0
        else:
            target_index = target_indices[0]

        # Use the smoothed time array to get the accurate target time
        if target_index < len(smoothed_time):
            target_t = smoothed_time[target_index]
        else:
            target_t = smoothed_time[-1]

        # Find the true index in the raw, unsmoothed array that matches this time
        true_idx = np.abs(time[channel] - target_t).argmin()

        # Append the found change point (True Index, Target Time)
        bkps[channel][j].append((true_idx, target_t))
    
    return bkps, channel_error_show