from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def get_index_from_time_jit(t_array, t_val, smooth, est):
    """
    Optimized search for large arrays using a localized search window.
    Tries a fast local scan near the estimated index, then falls back to a 
    binary search on the broader (but narrowed) interval if needed.
    
    Parameters:
      t_array: Sorted 1D NumPy array.
      t_val  : The target value.
      smooth : The window size to ensure a valid slice.
      est    : An estimated index hint where t_val is expected.
    
    Returns:
      A valid index so that t_array[idx:idx+smooth] is in bounds.
    """
    n = t_array.shape[0]
    if n == 0:
        return 0

    # Clamp the estimated index.
    if est < 0:
        est = 0
    elif est >= n:
        est = n - 1

    # Define a small search window (delta) around the estimated index.
    # Tune this value based on the expected accuracy of 'est'.
    delta = 8  
    left_bound = est - delta if est - delta >= 0 else 0
    right_bound = est + delta if est + delta < n else n - 1

    idx = -1  # Use this flag to determine if a local search found a valid index.
    
    # Determine search direction based on t_val relative to t_array[est].
    if t_val <= t_array[est]:
        i = est
        while i >= left_bound:
            if t_array[i] < t_val:
                idx = i + 1
                break
            i -= 1
        # If not found locally, adjust search boundaries.
        if idx == -1:
            left_bound = 0
            right_bound = est
    else:
        i = est
        while i <= right_bound:
            if t_array[i] >= t_val:
                idx = i
                break
            i += 1
        if idx == -1:
            left_bound = est + 1
            right_bound = n

    # If found within the local window, ensure the index is clamped to a valid slice.
    if idx != -1:
        if idx > n - smooth:
            idx = n - smooth
        elif idx < 0:
            idx = 0
        return idx

    # Fallback: Binary search on the restricted interval.
    left = left_bound
    right = right_bound
    while left < right:
        mid = (left + right) >> 1
        if t_array[mid] < t_val:
            left = mid + 1
        else:
            right = mid

    idx = left
    if idx > n - smooth:
        idx = n - smooth
    elif idx < 0:
        idx = 0
    return idx


def show_blob(blobs, fig_blob, smooth, i, hoverData, time, aoi_max, strided):
    """
    Update the blob figure based on hover data and AOI settings.
    Uses the Numba-accelerated binary search for fast time indexing.
    """
    if blobs is None or hoverData is None:
        return fig_blob
    
    
    trans = {
        0: 'g', 1: 'b', 2: 'b', 3: 'b', 4: 'b', 5: 'b',
        6: 'g', 7: 'g', 8: 'g', 9: 'r', 10: 'g', 11: 'b', 12: 'b', 13: 'g', 14: 'r'
    }


    
    # Get the time value from hover data
    try:
        t = hoverData["points"][0]["x"]
        c_num = hoverData["points"][0]["curveNumber"]
        p_num = hoverData["points"][0]["pointNumber"]
        channel = trans[c_num]
    except:
        return fig_blob
    
    # Unpack channels and compute min/max for colormap scaling
    b = blobs['b']
    g = blobs['g']
    r = blobs['r']
    minf = min(int(blobs['minf']), int(aoi_max[0]))
    maxf = max(int(blobs['minf']), int(aoi_max[1]))
    
    # List to accumulate blob slices
    z_list = []

    # Process the blue channel:
    if np.any(b):
        if strided:
            hint = int(p_num * (time['b'].shape[0] / time[channel].shape[0])) * smooth
        else:
            hint = int(p_num * (time['b'].shape[0] / time[channel].shape[0]))
        
        x = get_index_from_time_jit(time['b'].astype(np.float32), t, smooth, hint)
        # Use np.mean instead of np.average for a small performance gain:
        z_list.append(np.mean(b[i, 0, int(x - smooth /2) : int(x + smooth/2)], axis=0))
        z_list.append(np.mean(b[i, 1, int(x - smooth /2) : int(x + smooth/2)], axis=0))
        z_list.append(np.mean(b[i, 2, int(x - smooth /2) : int(x + smooth/2)], axis=0))
    else:
        z_list.extend([np.zeros((9, 9)) for _ in range(3)])

    # Process the green channel:
    if np.any(g):
        if strided:
            hint = int(p_num * (time['g'].shape[0] / time[channel].shape[0])) * smooth
        else:
            hint = int(p_num * (time['g'].shape[0] / time[channel].shape[0]))

        x = get_index_from_time_jit(time['g'].astype(np.float32), t, smooth, hint)
        z_list.append(np.mean(g[i, 0, int(x - smooth /2) : int(x + smooth/2)], axis=0))
        z_list.append(np.mean(g[i, 1, int(x - smooth /2) : int(x + smooth/2)], axis=0))
    else:
        z_list.extend([np.zeros((9, 9)) for _ in range(2)])

    # Process the red channel:
    if np.any(r):
        if strided:
            hint = int(p_num * (time['r'].shape[0] / time[channel].shape[0])) * smooth
        else:
            hint = int(p_num * (time['r'].shape[0] / time[channel].shape[0]))

        x = get_index_from_time_jit(time['r'].astype(np.float32), t, smooth, hint)
        z_list.append(np.mean(r[i, 0, int(x - smooth /2) : int(x + smooth/2)], axis=0))
    else:
        z_list.append(np.zeros((9, 9)))

    # Concatenate the computed slices along the second axis:
    z = np.concatenate(z_list, axis=1)
    
    # Update the figure's layout with the new data and color scale
    fig_blob['layout']['coloraxis']['colorscale'] = 'viridis'
    fig_blob.update_traces(zmax=maxf, zmin=minf, selector=dict(type='heatmap'))
    fig_blob['data'][0]['z'] = z
    fig_blob['layout']['coloraxis']['cmax'] = maxf
    fig_blob['layout']['coloraxis']['cmin'] = minf

    return fig_blob
