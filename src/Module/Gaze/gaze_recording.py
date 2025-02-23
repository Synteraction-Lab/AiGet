import time

import cv2
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

import zmq
from msgpack import loads
import numpy as np
from threading import Thread, Event


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

from src.Utilities.file import get_second_monitor_original_pos
from src.Utilities.screen_capture import ScreenCapture

HOP = 15

BOUNDARY = 0.15


# def overlay_heatmap_on_image(img, gaze_data, heatmap_detail=0.05, cmap='jet', downscale_factor=.5, pid="p1_1",
#                              show_cluster=False, show_trace=False):
#
#     # Extend the canvas by adding white boundaries
#     boundary_extension_y = int(BOUNDARY * img.shape[0])
#     boundary_extension_x = int(BOUNDARY * img.shape[1])
#     canvas = np.ones((img.shape[0] + 2 * boundary_extension_y, img.shape[1] + 2 * boundary_extension_x, 3)) * 255
#     canvas[boundary_extension_y:boundary_extension_y + img.shape[0],
#     boundary_extension_x:boundary_extension_x + img.shape[1]] = img
#
#     # Extract gaze coordinates
#     gaze_on_surf_x, gaze_on_surf_y = zip(*gaze_data)
#     gaze_on_surf_x = np.array(gaze_on_surf_x)
#     gaze_on_surf_y = np.array(gaze_on_surf_y)
#
#     # Adjust gaze data to the extended canvas
#     gaze_on_surf_x = (gaze_on_surf_x * img.shape[1] + boundary_extension_x) / canvas.shape[1]
#     gaze_on_surf_y = (gaze_on_surf_y * img.shape[0] + boundary_extension_y) / canvas.shape[0]
#
#     hist, _, _ = np.histogram2d(
#         gaze_on_surf_y,
#         gaze_on_surf_x,
#         range=[[0, 1], [0, 1]],
#         bins=canvas.shape[0:2]
#     )
#
#     # Downscale the histogram for faster Gaussian blur
#     hist_downscaled = zoom(hist, downscale_factor)
#
#     # Apply Gaussian blur to the downscaled histogram
#     filter_h = int(heatmap_detail * hist_downscaled.shape[0]) // 2 * 2 + 1
#     filter_w = int(heatmap_detail * hist_downscaled.shape[1]) // 2 * 2 + 1
#     heatmap_downscaled = gaussian_filter(hist_downscaled, sigma=(filter_w, filter_h), order=0)
#
#     # Upscale the heatmap back to the original resolution
#     heatmap = zoom(heatmap_downscaled, 1 / downscale_factor)
#
#     # Display the canvas with heatmap overlay and cluster numbers
#     fig, ax = plt.subplots()
#     ax.imshow(canvas.astype('uint8'))
#
#     ax.imshow(heatmap, cmap=cmap, alpha=0.3)
#
#
#     if show_trace:
#         # Visualize gaze path with a hop
#         ax.plot(gaze_on_surf_x[::HOP] * img.shape[1], gaze_on_surf_y[::HOP] * img.shape[0], color='lime', linestyle='-',
#                 linewidth=0.5, alpha=0.2)
#
#         # Visualize individual gaze points with a hop
#         ax.scatter(gaze_on_surf_x[::HOP] * img.shape[1], gaze_on_surf_y[::HOP] * img.shape[0], color='red', s=1,
#                    alpha=0.2)
#
#     if show_cluster:
#         # scale gaze data with hop
#         gaze_data = np.array(list(zip(gaze_on_surf_x[::2], gaze_on_surf_y[::2])))
#         print(gaze_data.shape)
#         scaled_gaze_data = StandardScaler().fit_transform(gaze_data)
#         dbscan = DBSCAN(eps=0.1, min_samples=5)
#         clusters = dbscan.fit_predict(scaled_gaze_data)
#         unique_clusters = np.unique(clusters)
#         sorted_clusters = []
#         for cluster in unique_clusters:
#             if cluster != -1:  # -1 represents noise in DBSCAN
#                 index = np.where(clusters == cluster)[0][0]
#                 sorted_clusters.append((cluster, index))
#         sorted_clusters = sorted(sorted_clusters, key=lambda x: x[1])
#
#         # Label clusters
#         # colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(sorted_clusters)))
#         for idx, (cluster, _) in enumerate(sorted_clusters):
#             cluster_center = gaze_data[clusters == cluster].mean(axis=0)
#             ax.text(cluster_center[0] * img.shape[1], cluster_center[1] * img.shape[0], str(idx + 1),
#                     ha='center', va='center', color='cyan', fontsize=7, fontweight='bold')
#
#         print(sorted_clusters)
#
#     plt.axis('off')
#
#     # Save the heatmap with timestamp
#     output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
#                                "data", pid, "heatmap", datetime.now().strftime("%H_%M_%S") + ".png")
#     if not os.path.exists(os.path.dirname(output_path)):
#         os.makedirs(os.path.dirname(output_path))
#     plt.savefig(output_path, dpi=400, format='png', bbox_inches='tight', pad_inches=0)
#     plt.show()
#     return output_path
def overlay_heatmap_on_image(img, gaze_data, heatmap_detail=0.05, cmap='jet', downscale_factor=.5, pid="p1_1",
                             show_cluster=False, show_trace=False, BOUNDARY=0.1, HOP=10):
    print("Overlaying heatmap on image...")
    # Extend the canvas by adding white boundaries
    boundary_extension_y = int(BOUNDARY * img.shape[0])
    boundary_extension_x = int(BOUNDARY * img.shape[1])
    canvas = np.ones((img.shape[0] + 2 * boundary_extension_y, img.shape[1] + 2 * boundary_extension_x, 3)) * 255
    canvas[boundary_extension_y:boundary_extension_y + img.shape[0],
    boundary_extension_x:boundary_extension_x + img.shape[1]] = img

    # Extract gaze coordinates
    gaze_on_surf_x, gaze_on_surf_y = zip(*gaze_data)
    gaze_on_surf_x = np.array(gaze_on_surf_x)
    gaze_on_surf_y = np.array(gaze_on_surf_y)

    # Adjust gaze data to the extended canvas
    gaze_on_surf_x = (gaze_on_surf_x * img.shape[1] + boundary_extension_x) / canvas.shape[1]
    gaze_on_surf_y = (gaze_on_surf_y * img.shape[0] + boundary_extension_y) / canvas.shape[0]

    hist, _, _ = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1], [0, 1]],
        bins=canvas.shape[0:2]
    )

    # Downscale the histogram for faster Gaussian blur
    hist_downscaled = zoom(hist, downscale_factor)

    # Apply Gaussian blur to the downscaled histogram
    filter_h = int(heatmap_detail * hist_downscaled.shape[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * hist_downscaled.shape[1]) // 2 * 2 + 1
    heatmap_downscaled = gaussian_filter(hist_downscaled, sigma=(filter_w, filter_h), order=0)

    # Upscale the heatmap back to the original resolution
    heatmap = zoom(heatmap_downscaled, 1 / downscale_factor)

    # Display the canvas with heatmap overlay and cluster numbers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(canvas.astype('uint8'))
    ax2.imshow(canvas.astype('uint8'))
    ax2.imshow(heatmap, cmap=cmap, alpha=0.3)

    if show_trace:
        # Visualize gaze path and individual gaze points with a hop
        ax2.plot(gaze_on_surf_x[::HOP] * img.shape[1], gaze_on_surf_y[::HOP] * img.shape[0], color='lime',
                 linestyle='-',
                 linewidth=0.5, alpha=0.2)
        ax2.scatter(gaze_on_surf_x[::HOP] * img.shape[1], gaze_on_surf_y[::HOP] * img.shape[0], color='red', s=1,
                    alpha=0.2)

    if show_cluster:
        # Cluster labeling
        gaze_data = np.array(list(zip(gaze_on_surf_x[::2], gaze_on_surf_y[::2])))
        scaled_gaze_data = StandardScaler().fit_transform(gaze_data)
        dbscan = DBSCAN(eps=0.1, min_samples=5)
        clusters = dbscan.fit_predict(scaled_gaze_data)
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            if cluster != -1:  # Ignore noise
                cluster_center = gaze_data[clusters == cluster].mean(axis=0)
                ax2.text(cluster_center[0] * img.shape[1], cluster_center[1] * img.shape[0], str(cluster + 1),
                         ha='center', va='center', color='cyan', fontsize=7, fontweight='bold')

    ax1.axis('off')
    ax2.axis('off')

    # Save the heatmap with timestamp
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                               "data", pid, "heatmap", datetime.now().strftime("%H_%M_%S") + ".png")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=400, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()
    return output_path

def load_and_visualize_gaze_data(img, gaze_data_npy_path=None, pid="p1_1", show_cluster=False, show_trace=False):
    if gaze_data_npy_path is None:
        gaze_data_npy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gaze_trace.npy")
    if os.path.exists(gaze_data_npy_path):
        gaze_data = np.load(gaze_data_npy_path)
        print(gaze_data.shape)
        return overlay_heatmap_on_image(img, gaze_data, pid=pid, show_cluster=show_cluster, show_trace=show_trace)
    else:
        print("Gaze data file not found!")
        return None


def get_screen_size():
    return get_second_monitor_original_pos()[2:]


class EyeTracker:
    def __init__(self, surface_name="screen", pid="p1_1"):
        self.start_time = None
        self.end_time = None
        self.surface_name = surface_name
        self.context = zmq.Context()
        self.addr = "127.0.0.1"
        self.req_port = "50020"
        self.sub = None
        self.x_dim, self.y_dim = get_screen_size()
        self.gaze_trace = np.empty((0, 2), float)
        self.recording_event = Event()
        self.recording_thread = None
        self.pid = pid

    def connect_to_server(self):
        req = self.context.socket(zmq.REQ)
        req.connect(f"tcp://{self.addr}:{self.req_port}")
        req.send_string("SUB_PORT")
        sub_port = req.recv_string()

        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(f"tcp://{self.addr}:{sub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{self.surface_name}")

    def record_data(self):
        smooth_x, smooth_y = 0.5, 0.5
        print("Recording gaze data...")
        self.start_time = time.time()
        try:
            while self.recording_event.is_set():
                topic, msg = self.sub.recv_multipart()
                gaze_position = loads(msg)
                if gaze_position["name"] == self.surface_name:
                    gaze_on_screen = gaze_position["gaze_on_surfaces"]
                    if len(gaze_on_screen) > 0:
                        if gaze_on_screen[-1]["confidence"] < 0.5:
                            continue
                        raw_x, raw_y = gaze_on_screen[-1]["norm_pos"]
                        smooth_x += 0.3 * (raw_x - smooth_x)
                        smooth_y += 0.3 * (raw_y - smooth_y)
                        x = smooth_x
                        y = smooth_y
                        y = 1 - y
                        x = np.clip(x, - BOUNDARY, 1 + BOUNDARY)
                        y = np.clip(y, - BOUNDARY, 1 + BOUNDARY)
                        # print norm_pos if out of (0, 1)
                        # if x < 0 or x > 1 or y < 0 or y > 1:
                        #     print(f"Out of boundary: {x}, {y}")
                        norm_pos = [x, y]
                        # x *= int(self.x_dim)
                        # y *= int(self.y_dim)
                        # print(f"X: {x}, Y: {y}")
                        # print(norm_pos)
                        self.gaze_trace = np.vstack([self.gaze_trace, norm_pos])

        except Exception as e:
            print(f"Error: {e}")

    def start_recording(self):
        self.recording_event.set()
        self.recording_thread = Thread(target=self.record_data, daemon=True)
        self.recording_thread.start()

    def clean_gaze_data(self):
        self.gaze_trace = np.empty((0, 2), float)

    def stop_recording(self):
        self.recording_event.clear()
        # self.recording_thread.join()
        self.end_time = time.time()
        print(f"Recording stopped. Duration: {self.end_time - self.start_time} seconds")

        # save to a new file each time recording stops
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                 'data', self.pid, 'gaze')
        # create base_path if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        file_name = f'gaze_trace_{len(os.listdir(base_path)) + 1}.npy'
        # file_name = f'gaze_trace.npy'
        export_path = os.path.join(base_path, file_name)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        np.save(export_path, self.gaze_trace)

        # Reset gaze_trace for the next recording
        self.gaze_trace = np.empty((0, 2), float)
        return export_path

    def get_gaze_data(self):
        return self.gaze_trace


if __name__ == "__main__":
    mode = "test_recording"  # "test_visualization" or "test_recording"
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                             'data', 'sample_data')
    path = os.path.join(base_path, 'image', '1.png')

    # load an image from the path
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mode == "test_visualization":
        file_name = 'gaze_trace_sample.npy'
        export_path = os.path.join(base_path, file_name)

        load_and_visualize_gaze_data(img=image, gaze_data_npy_path=export_path, show_cluster=False, show_trace=False)
    else:
        tracker = EyeTracker(surface_name="PandaExp")
        tracker.connect_to_server()
        try:
            tracker.start_recording()
            while True:
                pass
        except KeyboardInterrupt:
            print("Recording stopped by user.")
            gaze_data_file = tracker.stop_recording()
            load_and_visualize_gaze_data(img=image, gaze_data_npy_path=gaze_data_file)
