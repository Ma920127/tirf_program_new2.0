import numpy as np
import os



class DwellManager:
    def __init__(self, load_path=None):
        '''
        Initializes the DwellManager. 
        If a load_path is provided, it attempts to load existing data. 
        Otherwise, it starts with an empty template.
        '''
        # Start with the default empty structure
        self.data = self._initialize_empty_dict()
        self.events = self._initialize_empty_dict()
        
        # If a path was provided during creation, load the data immediately
        if load_path is not None:
            self.load(load_path)

    def _initialize_empty_dict(self):
        '''Internal method to return the base nested dictionary structure.'''
        return {
            'gg': {},
            'bb': {},
            'rr': {},
            'fret_g': {},
            'fret_b': {},
            'custom': {}
        }

    def reset(self):
        '''Wipes the current data and resets it to the empty structure.'''
        self.data = self._initialize_empty_dict()
        self.events = self._initialize_empty_dict()
        print("Dwell data has been reset to empty.")

    def save(self, save_path):
        '''Saves the current self.data dictionary to a .npz file.'''
        filepath = os.path.join(save_path, 'dwell.npz')
        try:
            np.savez_compressed(filepath, **self.data,events_dict = self.events)
            print(f"Successfully saved dwell data to: {filepath}")
        except Exception as e:
            print(f"Error saving dwell data: {e}")

    def load(self, load_path):
        '''Loads the nested dwell dictionary from a .npz file into self.data.'''
        filepath = os.path.join(load_path, 'dwell.npz')
        
        if not os.path.exists(filepath):
            print(f"Warning: '{filepath}' not found. Keeping current data.")
            return False
            
        try:
            # STEP 1: Load the .npz archive (DO NOT put .item() at the end of this line!)
            npz_file = np.load(filepath, allow_pickle=True)

            if 'events_dict' in npz_file.files:
                self.events = npz_file['events_dict'].item()
            else:
                self.events = self._initialize_empty_dict()
            
            # Extract standard data, ignoring the events_dict key
            self.data = {key: npz_file[key].item() for key in npz_file.files if key != 'events_dict'}
            print(f"Successfully loaded dwell data from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading dwell data from {filepath}: {e}")
            return False