from aoi_viewer import server, app
from waitress import serve

if __name__ == '__main__':
    print("🚀 Starting AOI Viewer Server on http://0.0.0.0:8042 ...")
    serve(server, host="0.0.0.0", port=8042, threads=12)






# test error
# from aoi_viewer import server, app
# from waitress import serve

# # --- BULLETPROOF ID CHECKER ---
# keys_to_check = [
#     'thres', 'mpath', 'average_frame', 'ratio_thres', 'minf', 'maxf',
#     'channel', 'radius', 'leakage_g', 'leakage_b', 'f_lag', 'lag_b',
#     'snap_time_g', 'snap_time_b', 'red_intensity', 'red_time', 'red',
#     'green_intensity', 'green_time', 'green', 'fit', 'fit_b',
#     'gfp_plot', 'preserve_selected', 'overwrite',
#     'savec', 'autoscale', 'configs', 'graph-size-tabs'
# ]

# # Convert the layout to a simple string to safely search it
# layout_str = str(app.layout)

# print("\n--- 🕵️ CHECKING LAYOUT IDs 🕵️ ---")
# missing_count = 0
# for key in keys_to_check:
#     # Check for both single and double quotes just in case
#     if f"id='{key}'" not in layout_str and f'id="{key}"' not in layout_str:
#         print(f"❌ CRITICAL ERROR: ID '{key}' is MISSING from aoi_layout.py!")
#         missing_count += 1

# if missing_count == 0:
#     print("✅ ALL IDs MATCH PERFECTLY!")
# print("----------------------------------\n")
# # ------------------------------

# if __name__ == '__main__':
#     print("🚀 Starting AOI Viewer Server on http://0.0.0.0:8042 ...")
#     serve(server, host="0.0.0.0", port=8042, threads=12)