# aoi_callbacks/callback_load_config.py
from dash_extensions.enrich import Output, Input, State
import numpy as np
from dash import callback_context, no_update
from dash.exceptions import PreventUpdate
from aoi_utils import save_config, load_config
from global_state import global_state as gs

# Define the exact order of keys to match your UI Outputs
config_keys = [
    'thres', 'mpath', 'average_frame', 'ratio_thres', 'minf', 'maxf',
    'channel', 'radius','min_distance','leakage_g', 'leakage_b', 'f_lag', 'lag_b',
    'snap_time_g', 'snap_time_b', 'red_intensity', 'red_time', 'red',
    'green_intensity', 'green_time', 'green', 'fit', 'fit_b',
    'gfp_plot', 'preserve_selected', 'overwrite'
]

# , 'preserve_selected', 'overwrite'

def register_load_config(app, fsc):
    @app.callback(
        # 1. Outputs List
        [Output(k, 'value') for k in config_keys] + [Output('configs', 'value')],
        # 2. Inputs List
        [
            Input('configs', 'value'),
            Input('savec', 'n_clicks'),
            Input('autoscale', 'n_clicks'),
            Input('graph-size-tabs', 'value')
        ],
        # 3. States List
        [State(k, 'value') for k in config_keys],
        prevent_initial_call=True
    )
    def load_config_callback(configs, savec, autoscale, graph_tabs, *config_values):
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Reconstruct the config_data dictionary from the *args
        config_data = {k: v for k, v in zip(config_keys, config_values)}

        # --- TAB OR RADIO CLICK ---
        if triggered_id in ["graph-size-tabs", "configs"]:
            target_num = configs
            
            loaded = load_config(num=target_num, subfolder=str(graph_tabs))
            
            if loaded:
                print(f"SUCCESS: Loaded configs/{graph_tabs}/{target_num}.json")
                return [loaded.get(k, no_update) for k in config_keys] + [int(target_num)]
            else:
                print(f"FAILED: configs/{graph_tabs}/{target_num}.json not found.")
                raise PreventUpdate

        # --- AUTOSCALE ---
        if triggered_id == "autoscale":
            if not gs.fig or not gs.fig["data"]:
                raise PreventUpdate
            z = gs.fig["data"][0]["z"]
            results = []
            for k in config_keys:
                if k == 'maxf': results.append(float(np.round(np.max(z))))
                elif k == 'minf': results.append(float(np.round(np.min(z))))
                else: results.append(config_data[k])
            return results + [no_update]

        # --- SAVE ---
        if triggered_id == "savec":
            save_config(num=configs, config_data=config_data, subfolder=str(graph_tabs))
            print(f"SAVED configs/{graph_tabs}/{configs}.json")
            raise PreventUpdate

        raise PreventUpdate

    return register_load_config
