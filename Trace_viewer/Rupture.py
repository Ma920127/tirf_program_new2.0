import ruptures as rpt
import numpy as np

rup_config={'model': 'l1',
            'pen' : 1,
            'min_size' : 1, 
            'jump' : 1,
    }

class Rupture:
    def __init__(self, trace):
        self.trace = np.array(trace)
        self.config = dict(rup_config)   # copy so mutations don't affect the global default
        
    def det_bkps(self, pad_width=30):
        config = self.config
        model = config['model']
        min_size = config['min_size']
        pen = config['pen']
        
        # 1. Pad the beginning of the trace
        padded_trace = np.pad(self.trace, (pad_width, 0), mode='edge')
        
        # 2. Fit Binseg on the padded data
        raw_bkps = rpt.Binseg(model=model, min_size=min_size, jump=config['jump']).fit_predict(padded_trace, pen=pen)
        
        # 3. Strip padding away to get true indexes
        bkps = []
        for b in raw_bkps:
            true_b = b - pad_width
            if 0 < true_b < len(self.trace):
                bkps.append(true_b)
        return bkps

    def analyze_trace(self, sm_time, min_gap=0, direction='both', censor_mode='none', upper_thresh=0.8, lower_thresh=0.2):
        """Master function that finds breakpoints and applies all user filters."""
        raw_indices = self.det_bkps()
        
        # 1. MIN-GAP FILTER
        gap_filtered = []
        last_idx = -99999
        for idx in sorted(raw_indices):
            if (idx - last_idx) >= min_gap:
                gap_filtered.append(idx)
                last_idx = idx

        # 2. DIRECTION FILTER
        time_bkps = []
        for idx in gap_filtered:
            start = max(0, int(idx) - 5)
            end = min(len(self.trace), int(idx) + 5)
            
            before_slice = self.trace[start:int(idx)]
            after_slice = self.trace[int(idx):end]
            
            if len(before_slice) > 0 and len(after_slice) > 0:
                mean_before = np.mean(before_slice)
                mean_after = np.mean(after_slice)
                
                is_up = mean_after > mean_before
                is_down = mean_after < mean_before
                
                keep = False
                if direction == 'both': keep = True
                elif direction == 'up' and is_up: keep = True
                elif direction == 'down' and is_down: keep = True
                
                if keep:
                    safe_idx = min(max(0, int(idx)), len(sm_time) - 1)
                    time_bkps.append(sm_time[safe_idx])
                    
        # 3. CENSOR LOGIC
        if len(time_bkps) == 0 and censor_mode and censor_mode != 'none':
            mean_intensity = np.mean(self.trace)
            
            try: u_val = float(upper_thresh)
            except: u_val = float('inf')
            
            try: l_val = float(lower_thresh)
            except: l_val = -float('inf')
            
            if censor_mode in ['right', 'both'] and mean_intensity >= u_val:
                time_bkps.append(sm_time[-1])
            elif censor_mode in ['left', 'both'] and mean_intensity <= l_val:
                time_bkps.append(sm_time[0])
                
        return time_bkps
