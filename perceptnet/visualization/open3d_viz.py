"""3D point-cloud + box visualization with Open3D.

Open3D is imported lazily inside the functions so that importing this module (and the
package) never requires Open3D and never opens a GUI. Rendering can run offscreen
(save a PNG) for headless machines / CI, or open an interactive window.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from perceptnet.geometry.boxes import boxes_to_corners_3d

# Box edges for an Open3D LineSet (same corner ordering as geometry.boxes).
_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def _point_cloud(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points)[:, :3])
    return pcd


def _box_lineset(boxes: np.ndarray, color=(1.0, 0.0, 0.0)):
    import open3d as o3d

    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    geoms = []
    for box in boxes:
        corners = boxes_to_corners_3d(box)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(np.array(_BOX_LINES)),
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(_BOX_LINES), 1)))
        geoms.append(ls)
    return geoms


def build_geometries(points: np.ndarray, boxes: Optional[np.ndarray] = None, box_color=(1, 0, 0)) -> list:
    """Build the Open3D geometry list (point cloud + box wireframes)."""
    geoms = [_point_cloud(points)]
    if boxes is not None and len(boxes):
        geoms.extend(_box_lineset(boxes, box_color))
    return geoms


def show(points: np.ndarray, boxes: Optional[np.ndarray] = None, window_name: str = "PerceptNet") -> None:
    """Open an interactive Open3D window (requires a display)."""
    import open3d as o3d

    o3d.visualization.draw_geometries(build_geometries(points, boxes), window_name=window_name)


def render_offscreen(
    points: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    out_path: str = "scene.png",
    width: int = 1280,
    height: int = 720,
) -> str:
    """Render the scene to a PNG without opening a window (headless-friendly)."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in build_geometries(points, boxes):
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    return str(out_path)
