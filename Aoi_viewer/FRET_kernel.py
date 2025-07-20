from processor import Processor
from GFP import GFP
from Gaussian_mixture.Gaussian_mixture_aoi import GMM
import numpy as np
from scipy.ndimage import uniform_filter1d as uf
import os
import subprocess
from win32com.client import Dispatch
import pythoncom

class Fret_kernel:
    
    def __init__(self, proc_config):
        self.path = proc_config['path']
        self.lag_g = proc_config['lag_g']
        self.lag_b = proc_config['lag_b']
        self.snap_time_g = proc_config['snap_time_g']
        self.snap_time_b = proc_config['snap_time_b']
        self.ow = proc_config['overwrite']
        self.proc_config = proc_config
    
    def is_path_open_in_explorer(self, path):
        path = os.path.abspath(path).lower()
        pythoncom.CoInitialize()
        shell = Dispatch("Shell.Application")
        for window in shell.Windows():
            try:
                folder = window.Document.Folder.Self.Path
                if folder.lower() == path:
                    return True
            except Exception:
                continue
        return False

        
    def auto_fret(self, plot=True, fit=True, fit_b = True, GFP_plot =True,  GFP_hist = True, fsc = None):
                
                n = self.ow    
                os.makedirs(self.path + r'/FRET', exist_ok = True)
                print(f'Saving to folder {n}') 

                procr = Processor(n, self.proc_config)
                fret_g, fret_b = procr.process_data()
                path = self.path + f'\\FRET\\{n}'

                snap_time_g = self.snap_time_g
                snap_time_b = self.snap_time_b


               

                fret_g = uf(fret_g, size = self.lag_g, mode = 'nearest')
                fret_b = uf(fret_b, size = self.lag_b, mode = 'nearest')

        
                if fit == True:   
                    print(f'Green : Snaping from {snap_time_g[0]} to {snap_time_g[1]}')       
                    fitr = GMM(path, fret_g[:, snap_time_g[0]:snap_time_g[1]], select = 1, channel = 'g')
                    fitr.fit(self.proc_config['fit_text'], fsc) 

                if fit_b == True:          
                    print(f'Blue : Snaping from {snap_time_b[0]} to {snap_time_b[1]}')
                    fitr = GMM(path, fret_b[:, snap_time_b[0]:snap_time_b[1]], select = 1, channel = 'b')
                    fitr.fit(self.proc_config['fit_text'], fsc) 

                if GFP_plot == True:
                    GFPr = GFP(path)
                    GFPr.plot(self.lag_g, snap_time_g)
                
                if GFP_hist == True:
                    GFPr = GFP(path)
                    GFPr.plot_hist(self.lag_g, snap_time_b)

                if self.is_path_open_in_explorer(path):
                    print(f"'{path}' already opened.")
                else:
                    subprocess.Popen(f'explorer "{path}"')
                

                try:
                    fsc.set("fret_progress", str(1))
                except:
                    pass

                return fret_g