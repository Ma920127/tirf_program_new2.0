# aoi_callbacks/callback_cal_FRET.py
from dash_extensions.enrich import Output, Input, State
from aoi_utils import cal_FRET_utils
from dash.exceptions import PreventUpdate


def register_cal_FRET(app, fsc):
    @app.callback(
        Output("FRET", "title"),
        Input("FRET", "n_clicks"),
        [
            State("path", "value"),
            State("preserve_selected", "value"),
            State("overwrite", "value"),
            State("leakage_g", "value"),
            State("leakage_b", "value"),
            State("f_lag", "value"),
            State("lag_b", "value"),
            State("red", "value"),
            State("green", "value"),
            State("fit", "value"),
            State("fit_b", "value"),
            State("gfp_plot", "value"),
            State("gfp_hist", "value"),
            State("snap_time_g", "value"),
            State("snap_time_b", "value"),
            State("red_time", "value"),
            State("green_time", "value"),
            State("red_intensity", "value"),
            State("green_intensity", "value")
        ]
    )
    def cal_FRET(n_clicks, path, ps, ow, leakage_g, leakage_b, f_lag, lag_b, red, green,
                 fit, fit_b, gfp_plot, gfp_hist, snap_time_g, snap_time_b, red_time, green_time,
                 red_intensity, green_intensity):
        if not n_clicks:  # Stops it from running on startup
            raise PreventUpdate
        
        fsc.set("fret_progress", 0)
        cal_FRET_utils(path, ps, ow, snap_time_g, snap_time_b, red, red_time, red_intensity,
                      green, green_time, green_intensity, leakage_g, leakage_b, f_lag, lag_b,
                      fit, fit_b, gfp_plot, gfp_hist, fsc)
        fsc.set("stage", "Idle")
        return None

    return register_cal_FRET
