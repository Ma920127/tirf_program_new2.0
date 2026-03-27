# aoi_callbacks/callback_log.py
from dash_extensions.enrich import Output, Trigger

def register_update_log(app, fsc):
    @app.callback(
        Output("log", "children"),
        Trigger("interval", "n_intervals")
    )
    def update_log(n):
        value = fsc.get("stage")
        if value is None:
            fsc.set("stage", "Idle")
        return value

    return register_update_log
