from hmmlearn.hmm import GaussianHMM
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d as uf

# Maps UI channel name → data keys, time key, breakpoint key, clipping, and plot labels
CHANNEL_CONFIG = {
    'fret_g': {'npz_key': 'fret_g', 'time_key': 'time_g', 'bkps_key': 'fret_g', 'clip': (0.01, 0.99), 'is_fret': True,  'ylabel': 'FRET'},
    'fret_b': {'npz_key': 'fret_b', 'time_key': 'time_b', 'bkps_key': 'fret_b', 'clip': (0.01, 0.99), 'is_fret': True,  'ylabel': 'FRET'},
    'bb':     {'npz_key': 'bb',     'time_key': 'time_b', 'bkps_key': 'b',      'clip': None,           'is_fret': False, 'ylabel': 'Intensity (BB)'},
    'gg':     {'npz_key': 'gg',     'time_key': 'time_g', 'bkps_key': 'g',      'clip': None,           'is_fret': False, 'ylabel': 'Intensity (GG)'},
    'rr':     {'npz_key': 'rr',     'time_key': 'time_r', 'bkps_key': 'r',      'clip': None,           'is_fret': False, 'ylabel': 'Intensity (RR)'},
}

class HMM_fitter:

    def __init__(self, path):
        self.path = path
        self.N_traces = 0
        self.hd_states = 0
        self.selected = []
        self.filt_Q = []
        self.means = []
        self.channel = 'fret_g'
        self.cfg = CHANNEL_CONFIG['fret_g']

    def load_traces(self, channel='fret_g'):
        cfg = CHANNEL_CONFIG[channel]
        self.channel = channel
        self.cfg = cfg

        data = np.load(self.path + r'\\data.npz')
        Q = data[cfg['npz_key']]
        self.N_traces = Q.shape[0]
        self.selected = np.load(self.path + r'\\selected_g.npy').astype(int)

        try:
            self.bkps = np.load(self.path + r'\\breakpoints.npz', allow_pickle=True)[cfg['bkps_key']]
        except:
            self.bkps = np.array([[] for _ in range(Q.shape[0])], dtype=object)

        self.time = data[cfg['time_key']]
        self.Q = Q
        return Q


    def process_Q(self, w=10):
        Q = uf(self.Q, w, mode='reflect', axis=1)
        if self.cfg['clip'] is not None:
            Q = np.clip(Q, *self.cfg['clip'])
        N_traces = Q.shape[0]
        pro_Q = np.zeros(0)
        length = []

        for i in range(0, N_traces):
            bkp = self.bkps[i]
            if self.selected[i] == 1:
                if len(bkp) > 0:
                    frag = Q[i][:bkp[0][0]]
                    pro_Q = np.concatenate([pro_Q, frag])
                    length.append(bkp[0][0])
                else:
                    pro_Q = np.concatenate([pro_Q, Q[i]])
                    length.append(Q.shape[1])
        pro_Q = pro_Q.reshape(-1, 1)
        self.pro_Q = pro_Q
        self.length = length

        return N_traces, length


    def fitHMM(self, r, w=10, means=None, fix_means=False, epoch=10, covariance_type='spherical', n_iter=20):
        N_traces, length = self.process_Q(w)
        if not np.any(means):
            means = [0.5]

        self.means = means
        means = np.array(means)

        k = means.shape[0]
        if fix_means:
            params = 'stc'
        else:
            params = 'stmc'

        if r:
            print('fitting')
            tic = time.perf_counter()
            models = []
            conv = []

            for e in tqdm(range(epoch)):
                print(f'\n Epoch {e}')
                model = GaussianHMM(n_components=k, n_iter=n_iter, verbose=True, min_covar=0, startprob_prior=np.ones(k)/k, means_prior=means, covars_prior=0.00001, covariance_type=covariance_type, transmat_prior=0.01, init_params='stc', params=params, implementation='scaling')
                model.means_ = means
                model.fit(self.pro_Q, length)
                models.append(model)
                conv.append(model.monitor_.history[-1])
            best = np.argmax(conv)
            print(f'Best likelihood {conv[best]}')
            model = models[best]
            toc = time.perf_counter()
            print(f"Finished in {toc - tic:0.4f} seconds")
            with open(self.path + r"\model.pkl", "wb") as file: pickle.dump(model, file)

        else:
            model = GaussianHMM(n_components=k, n_iter=10, verbose=True, min_covar=100, covariance_type=covariance_type, covars_prior=0.001, transmat_prior=0, init_params='stc', params=params)
            with open(self.path + r"\model.pkl", "rb") as file: model = pickle.load(file)

        mus = np.array(model.means_)
        self.mus = mus

        sigmas = np.array(model.covars_)
        P = np.array(model.transmat_)
        print(sigmas)
        print(mus)
        print(P)

        print('predicting')
        tic = time.perf_counter()
        hidden_states = model.predict(self.pro_Q, length)
        likelihood = model.predict(self.pro_Q, length)
        aic = model.aic(self.pro_Q, length)
        bic = model.bic(self.pro_Q, length)

        toc = time.perf_counter()
        print(f"Finished in {toc - tic:0.4f} seconds")
        print(f"Log Likelihood =  {np.sum(likelihood):.4f}")
        print(f"averaged Log Likelihood =  {np.average(likelihood):.4f}")
        print(f"AIC =  {aic:.4f}")
        print(f"aBIC =  {bic:.4f}")

        self.hidden_states = hidden_states
        return model, hidden_states

    def cal_states(self, plot, p_length=None, text=False, mode='tif'):
        plt.switch_backend('agg')
        font = {'family': 'Arial', 'size': 5}
        plt.rc('font', **font)
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        plt.rcParams["figure.figsize"] = (180/72, 120/72)
        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['ytick.major.width'] = 0.5

        N_traces = self.Q.shape[0]
        length = self.length
        start = 0
        time_smooth = uf(self.time, 10, mode='reflect', axis=0)
        hd_states = []
        time_arr = []
        print('plotting')
        path = os.path.join(self.path, 'HMM_traces')
        os.makedirs(path, exist_ok=True)

        j = 0

        for i in tqdm(np.arange(0, N_traces)):
            if self.selected[i] == 1:
                end = start + length[j]
                mus_frag = self.hidden_states[start:end]
                time_frag = time_smooth[:length[j]]
                time_arr.append(time_frag)
                trace_frag = self.pro_Q[start:end]
                hd_states_frag = self.mus[mus_frag]
                hd_states.append(hd_states_frag)

                if plot:
                    plt.plot(time_frag, trace_frag)
                    plt.plot(time_frag, hd_states_frag, linewidth=0.8)
                    if p_length is None:
                        p_length = np.max(time_frag)
                    for state, mean in enumerate(np.flip(np.sort(self.mus, axis=0))):
                        plt.hlines(mean, 0, np.max(time_frag), colors='skyblue', linestyles='dashed')
                        if text:
                            plt.text(p_length + 2, mean, f'{state}', color='red')
                    if self.cfg['is_fret']:
                        plt.ylim(0, 1)
                    plt.xlim(0, p_length)
                    plt.xlabel('time (s)')
                    plt.ylabel(self.cfg['ylabel'])
                    plt.tight_layout()
                    plt.savefig(path + f'\\{i}.{mode}', dpi=300, format=mode)
                    plt.close()
                start = end
                j = j + 1
            else:
                hd_states.append([])
        print(len(hd_states))
        print(len(self.length))

        save_name = f'hmm_{self.channel}.npz'
        np.savez(
            os.path.join(path, save_name),
            hd_states=np.array(hd_states, dtype=object),
            time=np.array(time_arr, dtype=object),
            means=self.mus,
            channel=self.channel
        )
