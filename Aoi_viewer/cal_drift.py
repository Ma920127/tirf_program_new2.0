from aoi_utils import load_path
from dash_extensions.enrich import FileSystemCache
from scipy.ndimage import affine_transform
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
import os
import logging
# import imageio.v2 as imageio  # use the recommended import to avoid warnings
from tqdm import tqdm



def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def estimate_robust_gaussian_drift(displacements, interval_idx, drifts_dir, max_ratio=2.0):
    drift = np.zeros(2)
    labels = ['X', 'Y']
    colors = ['skyblue', 'salmon']

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    for dim, label, color, ax in zip(range(2), labels, colors, axes):
        data = displacements[:, dim]
        med = np.median(data)
        mad = median_abs_deviation(data)
        mask = (data > med - 3 * mad) & (data < med + 3 * mad)
        filtered_data = data[mask]

        hist, bin_edges = np.histogram(filtered_data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.hist(filtered_data, bins=20, color=color, edgecolor='black', alpha=0.6, label=f'{label} drift')

        try:
            popt, _ = curve_fit(
                gaussian, bin_centers, hist,
                p0=[hist.max(), filtered_data.mean(), filtered_data.std()],
                maxfev=5000
            )
            fitted_mean = popt[1]
            if np.abs(fitted_mean - med) / (mad + 1e-9) > max_ratio:
                drift[dim] = med
                ax.set_title(f'Interval {interval_idx} - Drift {label} (Used median due to large deviation)')
                print(f'Used median for dimension {label} in interval {interval_idx} due to large deviation.')
            else:
                drift[dim] = fitted_mean
                x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
                y_fit = gaussian(x_fit, *popt)
                ax.plot(x_fit, y_fit, color='black', linestyle='--', label=f'Gaussian fit')
                ax.set_title(f'Interval {interval_idx} - Drift {label}')
        except RuntimeError:
            drift[dim] = med
            ax.set_title(f'Interval {interval_idx} - Drift {label} (Used median fallback)')
            print(f'Used median fallback for dimension {label} in interval {interval_idx}')

        ax.set_xlabel(f'Drift {label} (pixels)')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    os.makedirs(drifts_dir, exist_ok=True)
    plt.savefig(os.path.join(drifts_dir, f'filtered_drift_histogram_interval_{interval_idx}.png'))
    plt.close()

    return drift



# Drift tracking
def track_and_plot_drift(loader, image_stack, fsc, mpath, path, maxf=1500, minf=500,
                        average_frame=10, ratio_thres=1.5, per_n=200, pairing_threshold=1.5, channel = 'green'):
    previous_coords = None
    drift_history = []
    interval_idx = 0

    drifts_dir = os.path.join(path, 'drifts')
    os.makedirs(drifts_dir, exist_ok=True)

    anchors = np.arange(0, image_stack.shape[0], per_n)
    if anchors[-1] != image_stack.shape[0] - 1:
        anchors = np.append(anchors, image_stack.shape[0] - 1)
    print(channel)
    for anchor in anchors:
        im2, _ = loader.gen_dimg(anchor=anchor, mpath=mpath, maxf=maxf, minf=minf,
                                 laser=channel, average_frame=average_frame)
        blobs = loader.det_blob(plot=False, fsc=fsc, thres=7, r=3, ratio_thres = ratio_thres)
        current_coords = np.array([b.get_coord()[:2] + b.shift[0] for b in blobs])
        #add channel support
        if previous_coords is not None and len(previous_coords) > 0 and len(current_coords) > 0:
            dist, idx = cKDTree(current_coords).query(previous_coords, distance_upper_bound=pairing_threshold)
            print(idx)
            valid = dist < pairing_threshold

            if np.any(valid):
                displacements = current_coords[idx[valid]] - previous_coords[valid]
                drift_vector = estimate_robust_gaussian_drift(displacements, interval_idx, drifts_dir)
                drift_history.append((anchor, drift_vector))
                interval_idx += 1

        previous_coords = current_coords.copy()

    logging.info("Drift estimated successfully.")
    return drift_history

# Calculate cumulative drift per interval
def apply_drift_correction(target_channel, path, time_reference, time_target, drift_history, image_stack):
    cumulative_drift = np.zeros((image_stack.shape[0], 2))
    total_drift = np.zeros(2)
    logging.info(f"Applying Drift correction to {target_channel} channel")
    start = 0
    for i, (anchor, drift_vec) in enumerate(drift_history):
        end = anchor if i == 0 else drift_history[i][0]
        per_sec_drift = drift_vec / (time_reference[end] - time_reference[start])

        start_frame_target = np.searchsorted(time_target, time_reference[start], side='left')+1
        end_frame_target = np.searchsorted(time_target, time_reference[end], side='left')+1

        start_frame_target = min(start_frame_target, image_stack.shape[0] - 1)
        end_frame_target = min(end_frame_target, image_stack.shape[0] - 1)

        for frame in range(start_frame_target, end_frame_target):
            drift_step = per_sec_drift * (time_target[frame + 1] - time_target[frame])
            total_drift += drift_step
            cumulative_drift[frame + 1] = total_drift

        start = end

    last_valid_frame = np.max(np.nonzero(cumulative_drift[:, 0]))
    cumulative_drift[last_valid_frame + 1:] = cumulative_drift[last_valid_frame]

    # Plot cumulative drift
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_drift[:, 0], label='Y drift')
    plt.plot(cumulative_drift[:, 1], label='X drift')
    plt.title(f'Cumulative Drift ({target_channel.capitalize()} channel)')
    plt.xlabel('Frame')
    plt.ylabel('Pixels')
    plt.legend()
    plt.grid() 
    drifts_dir = os.path.join(path, 'drifts')
    os.makedirs(drifts_dir, exist_ok=True)
    plt.savefig(os.path.join(drifts_dir, f'cumulative_drift_{target_channel}.png'))
    plt.close()

    # Apply drift correction
    warped_stack = np.zeros_like(image_stack)
    for frame in range(image_stack.shape[0]):
        warped_stack[frame] = affine_transform(
            image_stack[frame], np.eye(2), offset = cumulative_drift[frame], order=1, mode='nearest'
        )
    logging.info(f"Drift correction applied to {target_channel} channel. Warped stack ready.")
    return warped_stack


# def save_movie_imageio(image_stack, filename):
#     with imageio.get_writer(filename, fps=fps, macro_block_size=1) as writer:
#         for frame in tqdm(range(image_stack.shape[0]), desc=f"Creating {filename}"):
#             fig, ax = plt.subplots(figsize=(5, 5))
#             ax.imshow(image_stack[frame], cmap='gray',
#                       vmin=np.percentile(image_stack[frame], 0),
#                       vmax=np.percentile(image_stack[frame], 99))
#             ax.axis('off')
#             plt.tight_layout()

#             plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
#             plt.close(fig)

#             image = imageio.imread('temp_frame.png')
#             writer.append_data(image)



def cal_drift(gs, channel_dict, fsc, mpath, path, maxf, minf, average_frame, channel, per_n = 200, pairing_threshold = 1.5, ratio_thres = 1.5):
    time_dict = {
    'green': gs.image_datas[0],
    'red': gs.image_datas[1],
    'blue': gs.image_datas[2],
    }

    drifts_dir = os.path.join(path, 'drifts')
    os.makedirs(drifts_dir, exist_ok=True)

    drift_history = track_and_plot_drift(
        loader = gs.loader,
        image_stack = channel_dict[channel],
        fsc = fsc,
        mpath = mpath,
        path = path,
        maxf = maxf,
        minf = minf,
        channel=channel,
        average_frame = average_frame,
        ratio_thres = ratio_thres,
        per_n = per_n,
        pairing_threshold = pairing_threshold
    )

    time_reference = time_dict[channel]

    if np.any(gs.image_g):
        warped_green = apply_drift_correction(
            target_channel='green',
            path = path,
            time_reference = time_reference,
            time_target = time_dict['green'],
            drift_history = drift_history,
            image_stack = gs.image_g,
        )
        gs.image_g = warped_green
        gs.loader.image_g = warped_green
        np.save(os.path.join(drifts_dir, f'warped_g.npy'), warped_green)
        


    if np.any(gs.image_b):
        warped_blue= apply_drift_correction(
            target_channel='blue',
            path = path,
            time_reference = time_reference,
            time_target =  time_dict['blue'],
            drift_history = drift_history,
            image_stack = gs.image_b,
        )
        gs.image_b = warped_blue
        gs.loader.image_b = warped_blue
        np.save(os.path.join(drifts_dir, f'warped_b.npy'), warped_blue)

    if np.any(gs.image_r):
        warped_red = apply_drift_correction(
            target_channel='red',
            path = path,
            time_reference = time_reference,
            time_target =  time_dict['red'],
            drift_history = drift_history,
            image_stack = gs.image_r,
        )
        gs.image_r = warped_red
        gs.loader.image_r = warped_red
        np.save(os.path.join(drifts_dir, f'warped_r.npy'), warped_red)













# path = r'J:\TIRF\20250421\lane1\F3101\20s'
# fsc = FileSystemCache("cache_dir")

# mpath = r'D:\TIRF_Program\Bkp_picker\mapping\20250423'
# maxf = 1500
# minf = 500
# channel = 'green'
# average_frame = 10
# per_n = 200
# pairing_threshold = 1.5  


# loader, image_g, image_r, image_b, image_datas = load_path(7, path, fsc)
# time_dict = {
#     'green': image_datas[0],
#     'red': image_datas[1],
#     'blue': image_datas[2],
# }



# drift_history = track_and_plot_drift(
#     loader = loader,
#     image_stack = image_g,
#     fsc = fsc,
#     mpath = mpath,
#     maxf = maxf,
#     minf = minf,
#     average_frame = average_frame,
#     per_n = per_n,
#     pairing_threshold = pairing_threshold,
#     channel = channel
# )

# time_reference = time_dict[channel]

# warped_green = apply_drift_correction(
#     target_channel='green',
#     time_reference = time_reference,
#     time_target = time_dict['green'],
#     drift_history = drift_history,
#     image_stack = image_g,
# )

# warped_blue= apply_drift_correction(
#     target_channel='blue',
#     time_reference = time_reference,
#     time_target =  time_dict['blue'],
#     drift_history = drift_history,
#     image_stack = image_b,
# )


# # # Video parameters
# # fps = 10
# # original_video_filename = 'original_stack_green.mp4'
# # warped_video_filename = 'warped_stack_green.mp4'
# # # Save original stack
# # save_movie_imageio(image_g, original_video_filename)
# # # Save warped stack
# # save_movie_imageio(warped_green, warped_video_filename)


# # original_video_filename = 'original_stack_blue.mp4'
# # warped_video_filename = 'warped_stack_blue.mp4'
# # #
# # save_movie_imageio(image_b, original_video_filename)
# # # Save warped stack
# # save_movie_imageio(warped_blue, warped_video_filename)

# # print("✅ Both movies saved successfully.")
