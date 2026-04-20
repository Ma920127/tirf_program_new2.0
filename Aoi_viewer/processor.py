from scipy.ndimage import uniform_filter1d as uf
import numpy as np
import os
import matplotlib.pyplot as plt


class Processor:
    def __init__(self, n, proc_config):
        self.n = n
        self.leakage_g = proc_config['leakage_g']
        self.leakage_b = proc_config['leakage_b']
        self.gamma_g = 1
        self.gamma_b = 1
        self.direct_bg = 0.13
        self.path = proc_config['path']
        self.lag = 1
        self.ti = proc_config['ti']
        self.red = proc_config['red']
        self.red_intensity = proc_config['red_intensity']
        self.red_time = proc_config['red_time']
        self.green = proc_config['green']
        self.green_intensity = proc_config['green_intensity']
        self.green_time = proc_config['green_time']
        self.ps = proc_config['preserve_selected']

    def load_data_npz(self):
        file_path = os.path.join(self.path, 'raw', 'hel0.npz')
        
        # 1. Load the data exactly ONCE into memory
        try:
            data = np.load(file_path)
            self.N_traces = int(data['cnt'])
        except FileNotFoundError:
            print(f"Could not find raw data at {file_path}")
            return

        # Helper function to safely extract data or return zeros
        def get_trace(key):
            return data[key] if key in data else np.zeros(self.N_traces)

        raw_gg = get_trace('trace_gg')
        raw_gr = get_trace('trace_gr')
        time_g = get_trace('time_g')

        raw_rr = get_trace('trace_rr')
        time_r = get_trace('time_r')

        raw_bb = get_trace('trace_bb')
        raw_bg = get_trace('trace_bg')
        raw_br = get_trace('trace_br')
        time_b = get_trace('time_b')

        # 2. Vectorized assignment (No more for-loops!)
        if np.any(raw_gg):
            self.avg_gg = raw_gg.copy()
            self.avg_gr = raw_gr.copy()
        else:
            self.avg_gg = np.zeros((self.N_traces, 10))
            self.avg_gr = np.zeros((self.N_traces, 10))

        if np.any(raw_rr):
            self.avg_rr = raw_rr.copy()
        else:
            self.avg_rr = np.zeros((self.N_traces, 10))

        if np.any(raw_bb):
            self.avg_bb = raw_bb.copy()
            self.avg_bg = raw_bg.copy()
            self.avg_br = raw_br.copy()
        else:
            self.avg_bb = np.zeros((self.N_traces, 10))
            self.avg_bg = np.zeros((self.N_traces, 10))
            self.avg_br = np.zeros((self.N_traces, 10))

        self.avg_time_g = uf(time_g, size=self.lag, mode='nearest')
        self.avg_time_r = uf(time_r, size=self.lag, mode='nearest')
        self.avg_time_b = uf(time_b, size=self.lag, mode='nearest')

    def plot_intensity(self, data, title):
        plt.hist(data.reshape(-1), bins=np.arange(-2000, 40000, 2000), density=True, color='purple')
        plt.ylabel('Probability Density')
        plt.xlabel('Intensity')
        plt.title(title)
        plt.tight_layout()
        
        out_path = os.path.join(self.path, 'total_intensity')
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, f'{title}.tif'))
        plt.close()

    def process_data(self):
        self.load_data_npz()  

        avg_time_g = self.avg_time_g
        avg_time_b = self.avg_time_b
        avg_time_r = self.avg_time_r

        avg_gg = self.avg_gg
        avg_gr = self.avg_gr - self.leakage_g * self.avg_gg
        avg_bb = self.avg_bb
        
        if self.avg_gg.shape == self.avg_bg.shape:
            print(f'Using direct bg_leakage = {self.direct_bg}')
            avg_bg = self.avg_bg - self.leakage_b * self.avg_bb - self.direct_bg * self.avg_gg 
            avg_br = self.avg_br - self.leakage_g * self.avg_bg - self.direct_bg * self.avg_gr 
        else:
            avg_bg = self.avg_bg - self.leakage_b * self.avg_bb
            avg_br = self.avg_br - self.leakage_g * self.avg_bg
            
        avg_rr = self.avg_rr

        # Load selected arrays safely
        fret_dir = os.path.join(self.path, 'FRET', str(self.n))
        try:
            if self.ps == 1:
                self.selected_g = np.load(os.path.join(fret_dir, 'selected_g.npy'))
            else:
                self.selected_g = np.ones(self.N_traces)
        except FileNotFoundError:
            self.selected_g = np.ones(self.N_traces)

        try:
            if self.ps == 1:
                self.selected_b = np.load(os.path.join(fret_dir, 'selected_b.npy'))
            else:
                self.selected_b = np.ones(self.N_traces)
        except FileNotFoundError:
            self.selected_b = np.ones(self.N_traces)

        # 3. Safe FRET calculation with 1e-10 to prevent division by zero crashes
        if np.any(avg_gg):
            fret_g = (avg_gr / self.gamma_g) / (avg_gr / self.gamma_g + avg_gg + 1e-10)
        else:
            fret_g = np.zeros((self.N_traces, 10))

        if np.any(avg_bb):
            fret_b = (avg_br / self.gamma_g + avg_bg) / (avg_br / self.gamma_g + avg_bg + self.gamma_b * avg_bb + 1e-10)
        else:
            fret_b = np.zeros((self.N_traces, 10))

        # 4. Vectorized Thresholding (Instantaneous!)
        if self.red == 1:
            if np.any(avg_rr):
                # Calculate the mean across the chosen time slice for all traces at once
                red_means = np.mean(avg_rr[:, self.red_time[0]:self.red_time[1]], axis=1)
                self.selected_g[red_means <= self.red_intensity] = -1
            else:
                print('No red channel')

        if self.green == 1:
            if np.any(avg_bb):
                combined_b = avg_bb + avg_bg + avg_br
                green_means = np.mean(combined_b, axis=1) # Adjust if you meant a specific time slice!
                self.selected_b[green_means <= self.green_intensity] = -1
            else:
                print('No green channel')

        print(f"Total Green Traces: {(self.selected_g == 1).sum():.0f}")
        print(f"Total Blue Traces: {(self.selected_b == 1).sum():.0f}")
        
        os.makedirs(fret_dir, exist_ok=True)


        save_g_path = os.path.join(fret_dir, 'selected_g.npy')
        save_b_path = os.path.join(fret_dir, 'selected_b.npy')

        if self.ps == 0 or not os.path.exists(save_g_path):
            np.save(save_g_path, self.selected_g)
            
        if self.ps == 0 or not os.path.exists(save_b_path):
            np.save(save_b_path, self.selected_b)

        self.fret_g = fret_g
        self.fret_b = fret_b

        print('Plotting total intensities...')
        self.plot_intensity(avg_gg, 'avg_gg')
        self.plot_intensity(avg_gr, 'avg_gr')
        self.plot_intensity(avg_gg + avg_gr, 'total_g')

        self.plot_intensity(avg_bb, 'avg_bb')
        self.plot_intensity(avg_bg, 'avg_bg')
        self.plot_intensity(avg_br, 'avg_br')
        self.plot_intensity(avg_bb + avg_bg + avg_br, 'total_b')

        # Save the final data payload safely
        np.savez(os.path.join(fret_dir, "data.npz"), 
                gg=avg_gg, 
                gr=avg_gr, 
                bb=avg_bb, 
                bg=avg_bg, 
                br=avg_br, 
                rr=avg_rr, 
                fret_g=fret_g,
                fret_b=fret_b,
                time_g=avg_time_g,
                time_b=avg_time_b,
                time_r=avg_time_r
                ) 
        
        return fret_g, fret_b

