# aoi_callbacks/callback_progress.py
from dash_extensions.enrich import Output, Trigger

def register_update_progress(app, fsc):
    @app.callback(
        [
            Output("load_progress", "value"),
            Output("load_progress", "label"),
            Output("blob_progress", "value"),
            Output("blob_progress", "label"),
            Output("int_progress", "value"),
            Output("int_progress", "label"),
            Output("fret_progress", "value"),
            Output("fret_progress", "label")
        ],
        Trigger("interval", "n_intervals")
    )
    def update_progress(n):
        load_prog = fsc.get("load_progress")
        blob_prog = fsc.get("progress")
        cal_prog = fsc.get("cal_progress")
        fret_prog = fsc.get("fret_progress")
        if load_prog is None:
            fsc.set("load_progress", 0)
            load_prog = 0
        load_prog = float(load_prog) * 100
        load_prog_label = f"{load_prog:.0f} %" if load_prog > 5 else ""
        if blob_prog is None:
            fsc.set("progress", 0)
            blob_prog = 0
        blob_prog = float(blob_prog) * 100
        blob_prog_label = f"{blob_prog:.0f} %" if blob_prog > 5 else ""
        if cal_prog is None:
            fsc.set("cal_progress", 0)
            cal_prog = 0
        cal_prog = float(cal_prog) * 100
        cal_prog_label = f"{cal_prog:.0f} %" if cal_prog > 5 else ""
        if fret_prog is None:
            fsc.set("fret_progress", 0)
            fret_prog = 0
        fret_prog = float(fret_prog) * 100
        fret_prog_label = f"{fret_prog:.0f} %" if fret_prog > 5 else ""
        return load_prog, load_prog_label, blob_prog, blob_prog_label, cal_prog, cal_prog_label, fret_prog, fret_prog_label

    return register_update_progress
