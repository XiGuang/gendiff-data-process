#!/usr/bin/env python3
"""Generate a five-camera oblique-photography flight path and visualization.

The script computes a serpentine camera path that covers a rectangular scene
aligned to the XZ plane (Y is up). Camera poses are spaced according to the
camera field-of-view and desired forward/side overlap ratios. Each waypoint
expands into five camera orientations (front/back/left/right obliques plus a
nadir shot). Flight lines are padded beyond the scene bounds so that the
camera footprint, not its center, reaches the region edges. The resulting poses
are saved into a plain-text manifest and the path is visualized as an image for
quick inspection.

Example:
    python generate_oblique_path.py \
        --xmin 0 --xmax 120 --zmin 0 --zmax 80 \
        --path-height 120 --ground-y 20 --fov-deg 60 \
        --image-width 8192 --image-height 5460 \
        --forward-overlap 0.7 --side-overlap 0.65 \
        --oblique-pitch-deg -40 --nadir-pitch-deg -90 \
        --output-path ./output/path.txt --image-path ./output/path.png
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
from list_obj_bboxes import collect_obj_files, combine, compute_bbox, format_bbox


def positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def normalize_angle(angle_deg: float) -> float:
    """Wrap an angle to the [-180, 180) range for readability."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    # Avoid values like -0.0 showing up in dumps.
    return 0.0 if abs(wrapped) < 1e-9 else wrapped


@dataclass(frozen=True)
class SceneBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def validate(self) -> None:
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin.")
        if self.zmax <= self.zmin:
            raise ValueError("zmax must be greater than zmin.")
        if self.ymax < self.ymin:
            raise ValueError("ymax must be greater or equal to ymin.")


@dataclass(frozen=True)
class Pose:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


def generate_line_samples(start: float, end: float, step: float) -> Iterable[float]:
    step = positive_float(step, name="step")
    delta = abs(end - start)
    if delta == 0:
        yield start
        return
    direction = 1.0 if end >= start else -1.0
    segment_length = delta
    count = max(1, math.ceil(segment_length / step))
    for i in range(count + 1):
        offset = min(i * step, segment_length)
        yield start + direction * offset


def compute_footprint(height: float, fov_rad: float) -> float:
    positive_float(height, name="path height")
    half_angle = clamp(fov_rad * 0.5, math.radians(5), math.radians(170))
    return 2.0 * height * math.tan(half_angle)


def compute_vertical_fov(hfov_rad: float, image_width: int, image_height: int) -> float:
    aspect = image_height / image_width
    return 2.0 * math.atan(aspect * math.tan(hfov_rad * 0.5))

def ceil_to_factor(value, factor):
    """
    将一个值向上取整到最接近的指定因子的倍数。

    参数:
    value (float/int): 需要取整的值。
    factor (float/int): 倍数因子（必须 > 0）。

    返回:
    float/int: value 向上取整后的因子倍数。
    """
    if factor <= 0:
        raise ValueError("因子 (factor) 必须是正数。")
    return math.ceil(value / factor) * factor

def serpentine_path(
    bounds: SceneBounds,
    altitude: float,
    along_step: float,
    cross_step: float,
    base_pitch: float,
    base_roll: float,
    prefix: str,
) -> List[Pose]:
    bounds.validate()
    along_step = positive_float(along_step, name="along_step")
    cross_step = positive_float(cross_step, name="cross_step")

    waypoints: List[Pose] = []
    current_z = bounds.zmin
    left_to_right = True
    index = 0
    z_extent = bounds.zmax - bounds.zmin
    if z_extent == 0:  # Degenerate extent: single strip.
        num_rows = 1
    else:
        num_rows = max(1, math.ceil(z_extent / cross_step) + 1)

    current_z += (z_extent - (num_rows-2) * cross_step)/2 if z_extent - (num_rows-2) * cross_step > 0 else 0.0

    for row in range(num_rows):
        z = clamp(current_z, bounds.zmin, bounds.zmax)
        if left_to_right:
            x_start, x_end = bounds.xmin, bounds.xmax
        else:
            x_start, x_end = bounds.xmax, bounds.xmin

        for x in generate_line_samples(x_start, x_end, along_step):
            name = f"{prefix}_{index:04d}"
            waypoints.append(
                Pose(name=name, x=x, y=altitude, z=z, yaw=0.0, pitch=base_pitch, roll=base_roll)
            )
            index += 1

        current_z += cross_step
        left_to_right = not left_to_right
        if current_z > bounds.zmax:
            break

    if not waypoints:
        raise RuntimeError("No waypoints were generated; check your bounds and spacing.")

    # Update yaw angles based on track direction.
    for i, pose in enumerate(waypoints):
        if len(waypoints) == 1:
            yaw = 0.0
        elif i < len(waypoints) - 1:
            next_pose = waypoints[i + 1]
            yaw = math.degrees(math.atan2(next_pose.z - pose.z, next_pose.x - pose.x))
        else:
            prev_pose = waypoints[i - 1]
            yaw = math.degrees(math.atan2(pose.z - prev_pose.z, pose.x - prev_pose.x))
        waypoints[i] = Pose(
            name=pose.name,
            x=pose.x,
            y=pose.y,
            z=pose.z,
            yaw=yaw,
            pitch=pose.pitch,
            roll=pose.roll,
        )

    return waypoints


def expand_five_camera_rig(
    poses: Sequence[Pose],
    oblique_pitch_deg: float,
    nadir_pitch_deg: float,
) -> List[Pose]:
    """Duplicate each waypoint into a 5-camera rig (front/back/left/right/nadir)."""

    camera_offsets = (
        # ("F", 0.0, oblique_pitch_deg),
        # ("B", 180.0, oblique_pitch_deg),
        # ("L", -90.0, oblique_pitch_deg),
        # ("R", 90.0, oblique_pitch_deg),
        ("N", 0.0, nadir_pitch_deg),
    )

    rigged: List[Pose] = []
    for pose in poses:
        for suffix, yaw_offset, pitch_deg in camera_offsets:
            rigged.append(
                Pose(
                    name=f"{pose.name}_{suffix}.png",
                    x=pose.x,
                    y=pose.y,
                    z=pose.z,
                    # Ignore vehicle heading: each camera keeps a fixed world yaw.
                    yaw=normalize_angle(yaw_offset),
                    # yaw=normalize_angle(yaw_offset, + pose.yaw),
                    pitch=pitch_deg,
                    roll=pose.roll,
                )
            )
    return rigged


def save_manifest(poses: Sequence[Pose], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for pose in poses:
            file.write(
                f"{pose.name} {pose.x:.3f} {pose.y:.3f} {pose.z:.3f} "
                f"{pose.yaw:.3f} {pose.pitch:.3f} {pose.roll:.3f}\n"
            )


def visualize_path(
    poses: Sequence[Pose],
    bounds: SceneBounds,
    image_path: pathlib.Path,
    dpi: int = 200,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    zs = [p.z for p in poses]

    ax.plot(xs, zs, ys, color="#1f77b4", linewidth=1.5, label="Flight Path")
    ax.scatter(xs[0], zs[0], ys[0], color="green", s=50, label="Start")
    ax.scatter(xs[-1], zs[-1], ys[-1], color="red", s=50, label="End")

    # Draw the scene footprint.
    footprint_x = [bounds.xmin, bounds.xmax, bounds.xmax, bounds.xmin, bounds.xmin]
    footprint_z = [bounds.zmin, bounds.zmin, bounds.zmax, bounds.zmax, bounds.zmin]
    ax.plot(footprint_x, footprint_z, [bounds.ymin] * len(footprint_x), "--", color="gray", label="Scene")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Oblique Photography Flight Path")
    ax.legend(loc="upper right")
    ax.view_init(elev=60, azim=-60)
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an oblique-photography flight path.")
    parser.add_argument(
        "--obj-root",
        type=pathlib.Path,
        help="Root folder containing OBJ files; if set, scene bounds are derived from all OBJ vertices.",
    )
    parser.add_argument("--xmin", type=float, help="Scene minimum X (meters). Required if --obj-root is not set.")
    parser.add_argument("--xmax", type=float, help="Scene maximum X (meters). Required if --obj-root is not set.")
    parser.add_argument("--ymin", type=float, help="Scene minimum Y (meters). Required if --obj-root is not set.")
    parser.add_argument("--ymax", type=float, help="Scene maximum Y (meters). Required if --obj-root is not set.")
    parser.add_argument("--zmin", type=float, help="Scene minimum Z (meters). Required if --obj-root is not set.")
    parser.add_argument("--zmax", type=float, help="Scene maximum Z (meters). Required if --obj-root is not set.")
    parser.add_argument("--path-height", type=float, required=True, help="Camera altitude or height above ground (meters).")
    parser.add_argument("--height-mode", choices=["absolute", "relative"], default="relative", help="Interpretation of path-height: absolute Y or relative to ground-y.")
    parser.add_argument("--ground-y", type=float, default=0.0, help="Ground reference Y used when height-mode=relative.")
    parser.add_argument("--fov-deg", type=float, required=True, help="Horizontal camera field-of-view in degrees.")
    parser.add_argument("--image-width", type=int, required=True, help="Image width in pixels.")
    parser.add_argument("--image-height", type=int, required=True, help="Image height in pixels.")
    parser.add_argument("--forward-overlap", type=float, default=0.7, help="Desired overlap along the flight direction (0-0.95).")
    parser.add_argument("--side-overlap", type=float, default=0.65, help="Desired overlap between parallel flight lines (0-0.95).")
    parser.add_argument("--oblique-pitch-deg", type=float, default=-40.0, help="Pitch angle for the four oblique cameras (degrees, negative looks downward).")
    parser.add_argument("--nadir-pitch-deg", type=float, default=-90.0, help="Pitch angle for the nadir camera (degrees).")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Camera roll angle (degrees).")
    parser.add_argument("--name-prefix", type=str, default="IMG", help="Prefix for generated image names.")
    parser.add_argument("--output-path", type=pathlib.Path, default=pathlib.Path("./oblique_path.txt"), help="Destination txt file for the poses.")
    parser.add_argument("--image-path", type=pathlib.Path, default=pathlib.Path("./oblique_path.png"), help="Path to save the visualization image.")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI for the visualization output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    obj_root = args.obj_root
    manual_bound_args = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    has_manual_bounds = any(getattr(args, name) is not None for name in manual_bound_args)

    if obj_root is not None and has_manual_bounds:
        print("Do not mix --obj-root with manual bounds --xmin/--xmax/--ymin/--ymax/--zmin/--zmax.", file=sys.stderr)
        raise SystemExit(1)

    if obj_root is not None:
        if not obj_root.exists():
            print(f"OBJ root folder does not exist: {obj_root}", file=sys.stderr)
            raise SystemExit(1)
        if not obj_root.is_dir():
            print(f"OBJ root path is not a directory: {obj_root}", file=sys.stderr)
            raise SystemExit(1)

        obj_files = list(collect_obj_files(obj_root))
        if not obj_files:
            print(f"No OBJ files found under: {obj_root}", file=sys.stderr)
            raise SystemExit(1)

        per_file_bboxes = []
        for obj_path in obj_files:
            try:
                bbox = compute_bbox(obj_path)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            per_file_bboxes.append(bbox)

        if not per_file_bboxes:
            print("No usable OBJ files found when deriving bounds.", file=sys.stderr)
            raise SystemExit(1)

        summary_bbox = combine(per_file_bboxes)
        print(
            f"Derived scene bounds from OBJ files in {obj_root}: {format_bbox(summary_bbox)}",
            file=sys.stderr,
        )
        xmin, xmax, ymin, ymax, zmin, zmax = summary_bbox
    else:
        missing = [name for name in manual_bound_args if getattr(args, name) is None]
        if missing:
            print(
                "Either set --obj-root or provide all of --xmin/--xmax/--ymin/--ymax/--zmin/--zmax.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        xmin = float(args.xmin)
        xmax = float(args.xmax)
        ymin = float(args.ymin)
        ymax = float(args.ymax)
        zmin = float(args.zmin)
        zmax = float(args.zmax)

    bounds = SceneBounds(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )
    bounds.validate()

    path_height = positive_float(float(args.path_height), name="path height")
    height_mode = args.height_mode
    ground_y = float(args.ground_y)
    if height_mode == "absolute":
        camera_y = path_height
        height_for_footprint = max(camera_y - bounds.ymax, 0.1)
    else:
        camera_y = ground_y + path_height
        height_for_footprint = path_height

    # Input is horizontal FOV; derive vertical FOV from image aspect ratio.
    hfov_deg = clamp(float(args.fov_deg), 10.0, 175.0)
    hfov_rad = math.radians(hfov_deg)
    image_width = positive_int(args.image_width, name="image width")
    image_height = positive_int(args.image_height, name="image height")
    vfov_rad = compute_vertical_fov(hfov_rad, image_width, image_height)
    vfov_deg = math.degrees(vfov_rad)

    # Along-track coverage uses vertical FOV (image height direction).
    # Cross-track coverage uses horizontal FOV (image width direction).
    footprint_along = compute_footprint(height_for_footprint, vfov_rad)
    footprint_cross = compute_footprint(height_for_footprint, hfov_rad)

    forward_overlap = clamp(float(args.forward_overlap), 0.0, 0.95)
    side_overlap = clamp(float(args.side_overlap), 0.0, 0.95)

    along_step = max(footprint_along * (1.0 - forward_overlap), footprint_along * 0.1)
    cross_step = max(footprint_cross * (1.0 - side_overlap), footprint_cross * 0.1)

    base_roll = float(args.roll_deg)

    # pad_x = footprint_along * 0.5
    # pad_z = footprint_cross * 0.5
    # coverage_bounds = SceneBounds(
    #     xmin=bounds.xmin - pad_x,
    #     xmax=bounds.xmax + pad_x,
    #     ymin=bounds.ymin,
    #     ymax=bounds.ymax,
    #     zmin=bounds.zmin - pad_z,
    #     zmax=bounds.zmax + pad_z,
    # )
    pad_x = (ceil_to_factor(bounds.xmax-bounds.xmin, along_step)- (bounds.xmax - bounds.xmin))/2
    coverage_bounds = SceneBounds(
        xmin=bounds.xmin-pad_x,
        xmax=bounds.xmax+pad_x,
        ymin=bounds.ymin,
        ymax=bounds.ymax,
        zmin=bounds.zmin,
        zmax=bounds.zmax,
    )

    coverage_bounds.validate()

    poses = serpentine_path(
        bounds=coverage_bounds,
        altitude=camera_y,
        along_step=along_step,
        cross_step=cross_step,
        base_pitch=0.0,
        base_roll=base_roll,
        prefix=str(args.name_prefix),
    )

    rigged_poses = expand_five_camera_rig(
        poses,
        oblique_pitch_deg=float(args.oblique_pitch_deg),
        nadir_pitch_deg=float(args.nadir_pitch_deg),
    )

    output_path = pathlib.Path(args.output_path)
    image_path = pathlib.Path(args.image_path)
    save_manifest(rigged_poses, output_path)
    visualize_path(poses, bounds, image_path, dpi=int(args.dpi))

    print(
        f"Derived vertical FOV: {vfov_deg:.2f}° from image aspect ratio {image_width}:{image_height}"
    )
    # print(
    #     f"Applied padding (m): X ±{pad_x:.2f}, Z ±{pad_z:.2f} so coverage meets scene edges"
    # )
    print(f"Generated {len(rigged_poses)} camera records ({len(poses)} waypoints × 5) -> {output_path}")
    print(f"Visualization saved to {image_path}")


if __name__ == "__main__":
    main()
