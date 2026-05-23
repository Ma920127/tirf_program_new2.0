import os
import numpy as np
from pathlib import Path

class Loader :
    
    def __init__(self,path):
        self.path = path
    
    def load_traces(self):
        if not os.path.exists(self.path + r'\data.npz'):
            raise FileNotFoundError(f"data.npz not found at '{self.path}'. Tell 🐎")
        Q1 = np.load(self.path+r'\data.npz')

        fret_g = Q1['fret_g']
        fret_b = Q1['fret_b']
        rr = Q1['rr']
        gg = Q1['gg']
        gr = Q1['gr']
        bb = Q1['bb']
        bg = Q1['bg']
        br = Q1['br']
        time_g = Q1['time_g']
        time_b = Q1['time_b']
        time_r = Q1['time_r']
        tot_g = gg + gr
        tot_b = bb + bg + br
        N_traces = fret_g.shape[0]
        total_frame = fret_g.shape[1]
        fret_g_bkps = [[] for _ in range(N_traces)]
        fret_b_bkps = [[] for _ in range(N_traces)]
        b_bkps = [[] for _ in range(N_traces)]
        g_bkps = [[] for _ in range(N_traces)]
        r_bkps = [[] for _ in range(N_traces)]
        bkps = {
            'fret_b' : fret_b_bkps,
            'fret_g' : fret_g_bkps,
            'b' : b_bkps,
            'g' : g_bkps,
            'r' : r_bkps,
        }
        time = {
            'fret_b' : time_b,
            'fret_g' : time_g,
            'b' : time_b,
            'g' : time_g,
            'r' : time_r,
            }
        
        try:
            select_list_g = np.load(self.path + r'\selected_g.npy', allow_pickle=True)
        except:
            select_list_g = np.zeros(N_traces)

        try:
            colocalized_list = np.load(self.path + r'\colocalized_list.npy', allow_pickle=True)
        except:
            colocalized_list = np.zeros(N_traces)
        
        blob_path = (Path(self.path).parent.parent.absolute()) / 'blobs.npz'
        try:
            blobs = np.load(blob_path)
        except:
            blobs = None
        
        # Try new unified hmm.npz (same level as data.npz) first
        try:
            _hmm = np.load(self.path + r'\hmm.npz', allow_pickle=True)['data'].item()
            hmm_fret_g = _hmm.get('fret_g', {}).get('hd_states', np.zeros_like(fret_g))
        except Exception:
            # Fall back to legacy HMM_traces location
            try:
                _hmm = np.load(self.path + r'\HMM_traces\hmm.npz', allow_pickle=True)['data'].item()
                hmm_fret_g = _hmm.get('fret_g', {}).get('hd_states', np.zeros_like(fret_g))
            except Exception:
                try:
                    hmm_fret_g = np.load(self.path + r'\HMM_traces\hmm_fret_g.npz', allow_pickle=True)['hd_states']
                except Exception:
                    hmm_fret_g = np.full(fret_g.shape, None, dtype=object)


        ch_label = []
        if np.any(fret_b):
            ch_label.append('fret_b')
        if np.any(fret_g):
            ch_label.append('fret_g')
        if np.any(bb):
            ch_label.append('b')
        if np.any(gg):
            ch_label.append('g')
        if np.any(rr):
            ch_label.append('r')

        return fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, N_traces, total_frame, bkps, select_list_g, colocalized_list, ch_label, blobs, hmm_fret_g
    