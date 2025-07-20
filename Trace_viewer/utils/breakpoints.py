import numpy as np
import os
import time as rtime
import shutil

import numpy as np
from .smoothing import uf, sa

def breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode):
    # Select the smoothing function based on smooth_mode
    sm = uf if smooth_mode == 'moving' else sa

    trans = {
        0: 'fret_g', 1: 'fret_b', 2: 'b', 3: 'b', 4: 'b', 5: 'b',
        6: 'g', 7: 'g', 8: 'g', 9: 'r', 10: 'fret_g', 11: 'fret_b', 12: 'b', 13: 'g', 14: 'r'
    }
    confirm_reset_show = False
    channel_error_show = False

    # Smooth the time array for the given channel (if channel is provided)


    if 'dtime' in changed_id:
        if channel is not None:
            smoothed_time = sm(time[channel], smooth, 0)
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
            smoothed_time = sm(time[channel], smooth, 0)
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
            smoothed_time = sm(time[channel], smooth, 0)
            if mode == 'Add':
                if c_num < 10:
                    idx = clickData["points"][0]["pointNumber"]
                    # Use the first element as a placeholder (you can change this logic)
                    idx_t = smoothed_time[idx]
                    true_idx = np.abs(time[channel] - idx_t).argmin()
                    bkps[channel][i].append((true_idx, idx_t))
                    bkps[channel][i] = sorted(bkps[channel][i])
            elif mode == 'Remove':
                if 10 <= c_num <= 14:
                    rem_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i].pop(rem_idx)
            elif mode == 'Except':
                if 10 <= c_num <= 14:
                    exp_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i] = [bkps[channel][i][exp_idx]]
                    
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
        for key in bkps:
            bkps[key] = np.array(bkps[key], dtype=object)
        np.savez(path + r'/breakpoints.npz', **bkps)
        print('file_saved')
    if 'load_bkps' in changed_id:
        try:
            bkps = dict(np.load(path + r'/breakpoints.npz', allow_pickle=True))
            #print(bkps['fret_g'])
        except:
            print('File not found')
    return bkps


def find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g, 
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, 
             chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode):
    """
    Find change points in the signal corresponding to a given channel, using the provided
    parameters for mode, comparison, threshold, smoothing and target selection.
    
    Parameters:
      changed_id: A string indicating which change button was pressed.
      fret_g, fret_b, ...: Data arrays (e.g. signals) for different channels.
      i: Current trace index.
      time: A dictionary keyed by channel (e.g., 'g', 'b') that provides time arrays.
      select_list_g: A list (converted to an array) used for selection (unused in this version).
      chp_mode_X, chp_comp_X, chp_thres_X, chp_channel_X, chp_target_X: Parameters 
            (for X = 0,1) that define the mode (e.g., 'first', 'second', 'previous'),
            the comparison ('bigger' or 'smaller'), the threshold (numeric), the channel,
            and target specification (e.g., 'current trace', 'all traces', 'all good').
      bkps: A dictionary to store the results; it should be indexed as bkps[channel][j] (where j comes from i_list)
      smooth: A numeric value defining the smoothing window (moving average window or stride length).
      smooth_mode: Either 'moving' or 'strided'.
      
    Returns:
      Updated bkps with found change point indices and associated times.
    """
    # Convert the selection list (if needed)
    channel_error_show = False
    
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


    # Apply smoothing on the chosen signal
    if smooth_mode == 'moving':
        smoothed_signal = uf(signal, smooth)
    elif smooth_mode == 'strided':
        smoothed_signal = sa(signal, smooth)

        

    # Process each trace index in i_list. 
    # (In a real application, if each trace is separate, you might re-select or recompute "smoothed_signal" per trace.)
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
            # Use the first time value (as a default) and the last index from the original time array.
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

        # Obtain the corresponding time value.
        # Note: if smoothing changes the alignment, adjustments might be needed.
        if target_index < len(time[channel]):
            target_t = time[channel][target_index]
        else:
            target_t = time[channel][-1]

        # For strided smoothing, adjust the index using the smooth factor.
        if smooth_mode == 'strided':
            target_index = int(smooth * (target_index + 0.5))

        # Append the found change point to bkps for the current channel and trace.
        bkps[channel][j].append((target_index, target_t))
    
    return bkps, channel_error_show