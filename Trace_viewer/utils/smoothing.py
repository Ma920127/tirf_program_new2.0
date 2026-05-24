import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.signal import savgol_filter

def uf(t, lag, axis=-1, mode='nearest'):
    """
    Apply a uniform filter (Moving Average) on array t.
    Corresponds to 'Sliding Window Average'.
    """
    return uniform_filter1d(t, size=lag, mode=mode, axis=axis)

def sa(t, lag, axis=-1):
    """
    Apply strided averaging on array t.
    Corresponds to 'Fixed Window Average'. 
    Note: This reduces the array length by factor of 'lag'.
    """
    # Ensure we are working with a numpy array
    t = np.asanyarray(t)
    
    if t.ndim == 1:
        truncated_length = len(t) - (len(t) % lag)
        reshaped = t[:truncated_length].reshape(-1, lag)
        return np.mean(reshaped, axis=1)
    
    elif t.ndim == 2:
        # If processing the whole matrix (global fitting)
        truncated_length = t.shape[1] - (t.shape[1] % lag)
        reshaped = t[:, :truncated_length].reshape(t.shape[0], -1, lag)
        return np.mean(reshaped, axis=2)
    
    return t

def mf(t, lag, mode='nearest'):
    """
    Apply a median filter on array t.
    Handles both 1D arrays (e.g., time) and 2D arrays (e.g., stacked traces).
    """
    # If it's a 1D array (like your time array)
    if t.ndim == 1:
        return median_filter(t, size=lag, mode=mode)
    
    # If it's a 2D array (like your fret_g traces)
    elif t.ndim == 2:
        # size=(1, lag) ensures we only smooth horizontally along the time axis, 
        # preventing traces from bleeding into each other.
        return median_filter(t, size=(1, lag), mode=mode)
        
    else:
        return median_filter(t, size=lag, mode=mode)

def sg(t, lag, polyorder=2, mode='interp', axis=-1):
    """
    Apply Savitzky-Golay filter.
    Corresponds to 'Savitzky-Golay'.
    Note: lag (window_length) must be odd and greater than polyorder.
    """
    if lag <= 0:
        raise ValueError(f"Smoothing window 'lag' must be positive, got {lag}. Tell 🐎")
    # Ensure window_length is odd
    if lag % 2 == 0:
        lag += 1
        
    # Ensure window_length > polyorder
    if lag <= polyorder:
        lag = polyorder + 1
        if lag % 2 == 0: lag += 1

    return savgol_filter(t, window_length=lag, polyorder=polyorder, mode=mode, axis=axis)
