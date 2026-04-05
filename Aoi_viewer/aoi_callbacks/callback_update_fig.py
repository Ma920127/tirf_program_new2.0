# aoi_callbacks/callback_update_fig.py
import numpy as np
import subprocess
import time
import logging
import os
from tqdm import tqdm
from dash import callback_context, no_update, Patch
from dash_extensions.enrich import Output, Input, State
from aoi_utils import draw_blobs, move_blobs, update_blobs_coords, load_path, cal_blob_intensity, save_aoi_utils, load_aoi_utils, update_fret_labels
from cal_drift import cal_drift
from global_state import global_state
from aoi_figure import create_initial_figure

def register_update_fig(app, fsc):
    @app.callback(
        [
            Output("graph", "figure"),
            Output("graph", "clickData"),
            Output("anchor", "value"),
            Output("blob", "disabled"),
            Output("cal_intensity", "disabled"),
            Output("frame_slider", "value"),
            Output("frame_slider", "max"),
            Output("snap_time_g", "max"),
            Output("red_time", "max"),
            Output("snap_time_b", "max"),
            Output("green_time", "max"),
            Output("aoi_mode", "value"),
            Output("aoi_num", "children"),
            Output("loadp", "title"),
            Output("FRET", "outline"),
            Output("auto", "n_clicks")
        ],
        [
            Input("graph", "clickData"),
            Input("graph", "relayoutData"),
            Input("blob", "n_clicks"),
            Input("up", "n_clicks"),
            Input("down", "n_clicks"),
            Input("left", "n_clicks"),
            Input("right", "n_clicks"),
            Input("fit_gauss", "n_clicks"),
            Input("frame_slider", "value"),
            Input("anchor", "value"),
            Input("average_frame", "value"),
            Input("loadp", "n_clicks"),
            Input("minf", "value"),
            Input("maxf", "value"),
            Input("reverse", "value"),
            Input("channel", "value"),
            Input("cal_drift", "n_clicks"),
            Input("load_drift", "n_clicks"),
            Input("cal_intensity", "n_clicks"),
            Input("openp", "n_clicks"),
            Input("configs", "value"),
            Input("aoi_mode", "value"),
            Input("graph-size-tabs", "value")
        ],
        [
            State("ratio_thres", "value"),
            State("radius", "value"),
            State("selector", "value"),
            State("move_step", "value"),
            State("path", "value"),
            State("mpath", "value"),
            State("plot_circle", "value"),
            State("thres", "value"),
            State("per_n", "value"),
            State("pairing_threshold", "value"),
            State("auto", "n_clicks")
        ],
    )

    def update_fig(clickData, relayout, blob, up, down, left, right, fit_gauss, frame, anchor,
                   average_frame, loadp, minf, maxf, reverse, channel, cal_drift_bt, load_drift, cal_intensity,
                   openp, configs, aoi_mode, graph_size_tab, ratio_thres, radius, selector,
                   move_step, path, mpath, plot, thres, per_n, pairing_threshold, auto):
        gs = global_state
        current_fig = gs.fig
        step_start = time.perf_counter()

        # Grab the size from the tab (e.g., "512" or "1024")
        camera_size = int(graph_size_tab) if graph_size_tab else 1024

        # 1. Figure out what triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            triggered_ids = ['No clicks yet']
        else:
            triggered_ids = [t['prop_id'] for t in ctx.triggered]

        # Keep a primary changed_id for legacy checks, but prioritize real events
        changed_id = triggered_ids[0]


        # 2. Check if the TAB was the thing that got clicked
        if changed_id == 'graph-size-tabs.value':
            
            # Instantly reset ALL global arrays and lists to the new size
            gs.set_camera_size(camera_size)
            
            # Make a totally blank array of that size so the axes update
            dummy_image = np.zeros((1, camera_size, camera_size))
            
            # Rebuild the figure with the new axes
            new_fig = create_initial_figure(dummy_image, minf, maxf, radius)
            
            # 👇 THE FIX: Force Plotly to redraw and break the memory cache!
            new_fig.update_layout(uirevision='constant', datarevision=time.time())
            gs.fig = new_fig
            
            import copy
            fresh_fig = copy.deepcopy(new_fig)
            
            # Return the fresh figure, and reset all sliders/buttons to 0 since the image is cleared
            return (fresh_fig, None, 0, True, True, 
                    0, 0, 0, 0, 0, 0, no_update, 
                    0, None, True, no_update)
        

        if "loadp" in changed_id:
            fsc.set("load_progress", "0")
            
            # 🌟 ADD THESE TWO LINES: Clear zoom memory on new load
            gs.x_range = None
            gs.y_range = None
            gs.loader, gs.image_g, gs.image_r, gs.image_b, gs.image_datas = load_path(thres, path, fsc, camera_size = camera_size)
            current_fig = create_initial_figure(gs.image_g, minf, maxf, radius)
            gs.blob_disable = False
            gs.fret_g = None
            frame = 0
            fsc.set("stage", "Image Loaded")
            logging.info("Image load in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()


        # 3. SET BASE ARRAYS USING THE PRIORITIZED CAMERA SIZE
        channel_dict = {
            "green": gs.image_g if gs.image_g is not None else np.zeros((1, camera_size, camera_size)),
            "red": gs.image_r if gs.image_r is not None else np.zeros((1, camera_size, camera_size)),
            "blue": gs.image_b if gs.image_b is not None else np.zeros((1, camera_size, camera_size))
        }


        if "blob" in changed_id:
            fsc.set("progress", 0)
            gs.loader.gen_dimg(anchor = anchor, mpath = mpath, maxf = maxf, minf = minf, laser = channel, average_frame = average_frame)
            blob_list = gs.loader.det_blob(plot=plot, fsc=fsc, thres=thres, r=radius, ratio_thres=float(ratio_thres))
            gs.blob_list = blob_list
            gs.coord_list = [b.get_coord() for b in blob_list]
            coord_array = np.array(gs.coord_list) if gs.coord_list else np.empty((0,))
            current_fig = draw_blobs(current_fig, coord_array, gs.dr if gs.dr is not None else radius, reverse)
            fsc.set("stage", "Blobing Finished")
            logging.info("Blob detection in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()

        if any(btn in changed_id for btn in ["up", "down", "left", "right"]):
            coord_array = np.array(gs.coord_list) if gs.coord_list else np.empty((0,))
            # Extract the specific direction that was clicked
            direction = changed_id.split('.')[0]
            coord_array = move_blobs(coord_array, selector, int(move_step), direction)
            gs.coord_list = coord_array.tolist()
            update_blobs_coords(gs.blob_list, coord_array)

            current_fig = draw_blobs(current_fig, coord_array, gs.dr if gs.dr is not None else radius, reverse)
            logging.info("Movement %s in %.3f sec", direction, time.perf_counter()-step_start)
            step_start = time.perf_counter()

        if 'fit_gauss' in changed_id:

            gs.loader.gen_dimg(anchor = anchor, mpath = mpath, maxf = maxf, minf = minf, laser = channel, average_frame = average_frame)
            logging.info("BM3D Image Processed in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()

            ch_dict = {
                'channel_r': 'red',
                'channel_g': 'green',
                'channel_b': 'blue'
            }
            for b in tqdm(gs.blob_list):
                b.set_image(gs.loader.dframe_r, laser = 'red')
                b.set_image(gs.loader.dframe_g, laser = 'green')
                b.set_image(gs.loader.dframe_b, laser = 'blue')
                b.gaussian_fit(ch = ch_dict[selector], laser = channel)
            gs.coord_list = [b.get_coord() for b in gs.blob_list]
            coord_array = np.array(gs.coord_list) 
            current_fig = draw_blobs(current_fig, coord_array, gs.dr, reverse)
            logging.info(f"Blob fitting for {ch_dict[selector]} with {channel} laser in %.3f sec", time.perf_counter()-step_start)

        if "openp" in changed_id:
            subprocess.Popen(f'explorer "{path}"')

        if "cal_drift" in changed_id:
            if gs.loader == None:
                logging.info("Error: No image loader detected")
            else:
                cal_drift(
                    gs = gs, 
                    channel_dict = channel_dict, 
                    fsc = fsc, 
                    mpath = mpath, 
                    path = path,
                    maxf = maxf, 
                    minf = minf, 
                    average_frame = average_frame, 
                    ratio_thres = ratio_thres,
                    channel = channel, 
                    per_n = per_n, 
                    pairing_threshold = pairing_threshold)
                
        if "load_drift" in changed_id:
            if gs.loader == None:
                logging.info("Error: No image loader detected")
            else:
                drifts_dir = os.path.join(path, 'drifts')
                os.makedirs(drifts_dir, exist_ok=True)
                for c1, attr in [('g', 'image_g'), ('b', 'image_b'), ('r', 'image_r')]:
                    try:
                        warped_image = np.load(os.path.join(drifts_dir, f'warped_{c1}.npy'))
                        setattr(gs, attr, warped_image)
                        setattr(gs.loader, attr, warped_image)
                        logging.info(f"Successfully loaded drift-corrected {attr} .")
                    except Exception as e:
                        logging.warning(f"Could not load drift-corrected {attr}: {e}")
            
        if "cal_intensity" in changed_id:
            fsc.set("cal_progress", 0)
            cal_blob_intensity(gs.loader, np.array(gs.coord_list), path, gs.image_datas, maxf, minf, fsc)
            fsc.set("stage", "Intensity Calculated")
            logging.info("Intensity calculation in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()


        # 🌟 1. CHANGE THIS TO EXACTLY MATCH RELAYOUT (ZOOM/PAN)
        if 'graph.relayoutData' in triggered_ids and len(triggered_ids) == 1:
            if isinstance(relayout, dict):
                # Save the exact zoom coordinates!
                if "xaxis.range[0]" in relayout and "yaxis.range[0]" in relayout:
                    gs.x_range = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
                    gs.y_range = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]
                # If the user double-clicks to reset the view, clear the memory
                elif "xaxis.autorange" in relayout:
                    gs.x_range = None
                    gs.y_range = None
                    # New added
                    # 👉 FIX 1: Reset the sizing memory when zooming all the way out!
                    gs.org_size = 1.0
                    gs.dr = float(radius)

            if isinstance(relayout, dict) and "xaxis.range[1]" in relayout:
                try:
                    # Fix the math: Use the dynamic 'camera_size' instead of hardcoded 512
                    zoom_range = relayout["xaxis.range[1]"] - relayout["xaxis.range[0]"]
                    zoom_ratio = camera_size / zoom_range 
                    new_size = np.round(zoom_ratio, 2)
                    
                    if new_size != gs.org_size:
                        gs.org_size = new_size
                        gs.dr = radius * new_size
                        
                        # Update the marker sizes in the current figure directly!
                        if np.any(np.array(gs.coord_list)):
                            new_marker_size = 2 * gs.dr + 1
                            color = '#1f77b4' if int(reverse) == 0 else 'yellow'

                            for trace_idx in [1, 2, 3]:
                                if len(current_fig.data) > trace_idx:
                                    current_fig.data[trace_idx].marker.size = new_marker_size
                                    current_fig.data[trace_idx].marker.color = color

                    # Notice we DO NOT return early anymore! 
                    # We let the code flow to the bottom to update the image pixels.

                except Exception as e:
                    logging.exception("Error during relayout: %s", e)

        # 🌟 2. CHANGE THIS TO EXACTLY MATCH CLICKS (DELETING BLOBS)
        if changed_id == "graph.clickData":
            if isinstance(clickData, dict):
                if clickData["points"][0]["curveNumber"] in [1,2,3]:
                    if aoi_mode == 0:
                        remove_id = clickData["points"][0]["pointNumber"]
                        gs.rem_list.append(gs.coord_list[remove_id])
                        gs.rem_list_blob.append(gs.blob_list[remove_id])
                        gs.coord_list = np.delete(np.array(gs.coord_list), remove_id, axis=0).tolist()
                        gs.blob_list.pop(remove_id)
                        current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)


        # (Undo, Save, Load, Clear AOI logic)
        if aoi_mode == 2:
            aoi_mode = 0
            if len(gs.rem_list_blob) > 0:
                new_coord = np.array(gs.rem_list.pop())
                if np.any(np.array(gs.coord_list)):
                    combined = np.concatenate((np.array(gs.coord_list), new_coord.reshape(1,12)), axis=0)
                else:
                    combined = new_coord.reshape(1,12)
                gs.coord_list = combined.tolist()
                gs.blob_list.append(gs.rem_list_blob.pop())
                current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)
        if aoi_mode == 3:
            aoi_mode = 0
            save_aoi_utils(gs.blob_list, path + r'\\aoi.dat')
            save_aoi_utils(gs.rem_list_blob, path + r'\\bad_aoi.dat')
            logging.info("Saved AOI")
        if aoi_mode == 4:
            aoi_mode = 0
            gs.blob_list = load_aoi_utils(path + r'\\aoi.dat')
            gs.coord_list = [b.get_coord() for b in gs.blob_list]
            current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)
            logging.info("Loaded AOI")
        if aoi_mode == 5:
            aoi_mode = 0
            gs.rem_list = gs.rem_list + gs.coord_list
            gs.rem_list_blob = gs.rem_list_blob + gs.blob_list
            gs.blob_list = []
            gs.coord_list = []
            current_fig = draw_blobs(current_fig, np.empty((0,)), gs.dr, reverse)
            logging.info("Cleared AOI")
        
        if "channel" in changed_id:
            if frame > channel_dict[channel].shape[0]:
                frame = 0
        if "anchor.value" in changed_id:
            if int(anchor) < channel_dict[channel].shape[0]:
                frame = int(anchor)


        if 'reverse.value' in changed_id:
            # 1. Safely handle the radius (fallback to default radius if not zoomed)
            radius_to_use = gs.dr if getattr(gs, 'dr', None) is not None else float(radius)
            
            # 2. Safely convert coordinates to a numpy array
            coords_to_use = np.array(gs.coord_list) if gs.coord_list else np.empty((0,))

            # 3. Update the background colorscale directly
            if int(reverse) == 0:
                current_fig.data[0].colorscale = 'gray'
            else:
                current_fig.data[0].colorscale = 'gray_r'
                
            # 4. Redraw using the safe variables!
            current_fig = draw_blobs(current_fig, coords_to_use, radius_to_use, reverse)



        end_idx = min(channel_dict[channel].shape[0], int(frame) + int(average_frame))
        start_idx = max(0, end_idx - int(average_frame))

        smooth_image = np.average(channel_dict[channel][start_idx:end_idx], axis=0)

        # --- SMART ZOOM RULE ---
        import cv2
        max_y, max_x = smooth_image.shape
        target_size = 512

        # 1. Determine viewing area based on saved zoom
        if getattr(gs, 'x_range', None) is not None and getattr(gs, 'y_range', None) is not None:
            x0 = max(0, int(min(gs.x_range)))
            x1 = min(max_x, int(max(gs.x_range)))
            y0 = max(0, int(min(gs.y_range)))
            y1 = min(max_y, int(max(gs.y_range)))
            # Prevent 0-width slices
            if x1 <= x0: x1 = x0 + 1
            if y1 <= y0: y1 = y0 + 1
        else:
            x0, x1 = 0, max_x
            y0, y1 = 0, max_y

        # 2. Slice the massive image down to ONLY what the user is looking at
        cropped_frame = smooth_image[y0:y1, x0:x1]

        # 3. Compress if looking at a large area, otherwise send raw pixels!
        if cropped_frame.shape[0] > target_size or cropped_frame.shape[1] > target_size:
            display_frame = cv2.resize(
                cropped_frame.astype(np.float32), 
                (target_size, target_size), 
                interpolation=cv2.INTER_AREA
            )
            x_coords = np.linspace(x0, x1, target_size)
            y_coords = np.linspace(y0, y1, target_size)
        else:
            # Zoomed in tightly: Use the raw pixels!
            display_frame = cropped_frame.astype(np.float32)
            x_coords = np.linspace(x0, x1, cropped_frame.shape[1])
            y_coords = np.linspace(y0, y1, cropped_frame.shape[0])

        # 4. Map the new pixels to the exact coordinates on the Plotly graph
        current_fig.data[0].z = display_frame
        current_fig.data[0].x = x_coords
        current_fig.data[0].y = y_coords
        # -----------------------

        current_fig.data[0].zmax = maxf
        current_fig.data[0].zmin = minf

        if channel == 'green':
            current_fig = update_fret_labels(current_fig, frame)
        else:
            current_fig.update_traces(customdata= [], selector=dict(name='blobs_r'))
            current_fig.update_traces(customdata= [], selector=dict(name='blobs_g'))

        slider_max = channel_dict[channel].shape[0]
        snap_g_max = max(channel_dict["green"].shape[0]-1, 0)
        r_max = max(channel_dict["red"].shape[0]-1, 0)
        snap_b_max = max(channel_dict["blue"].shape[0]-1, 0)
        g_max = max(channel_dict["green"].shape[0]-1, 0)
        current_max_frame = channel_dict[channel].shape[0]
        anchor = min(int(frame), current_max_frame - 1) if current_max_frame > 0 else 0
        
        aoi_num = len(gs.coord_list)
        auto_state = no_update if fsc.get("mode") != "auto" else (auto or 0) + 1

        # New add
        # 🌟 ADD THIS BLOCK: Force Plotly to respect our saved zoom!
        if getattr(gs, 'x_range', None) is not None and getattr(gs, 'y_range', None) is not None:
            current_fig.update_xaxes(range=gs.x_range, autorange=False)
            current_fig.update_yaxes(range=gs.y_range, autorange=False)

        # 👇 THE FIX: Add datarevision=time.time() to force Plotly to redraw!
        current_fig.update_layout(uirevision='constant', datarevision=time.time())
        gs.fig = current_fig

        # 🌟 FIX: Create a fresh Plotly Figure object to break the memory reference cache!
        import copy
        fresh_fig = copy.deepcopy(current_fig)


        return (fresh_fig, None, anchor, gs.blob_disable, gs.blob_disable,
                anchor, slider_max, snap_g_max, r_max, snap_b_max, g_max, aoi_mode,
                aoi_num, None, True, auto_state)

    return register_update_fig

