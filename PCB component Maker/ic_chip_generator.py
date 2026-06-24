from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import trimesh
from vedo import Line, Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "ic_chip_generator"
BRIDGE_DIR = DEFAULT_OUTPUT_DIR / "viewer_bridge"
PROJECTS_ROOT_DIR = DEFAULT_OUTPUT_DIR / "projects"
CANVAS_BACKGROUND = "#f7f0e4"
CANVAS_GRID = "#e6dac7"
CANVAS_AXIS = "#c7b39b"
DEFAULT_CANVAS_WIDTH = 920
DEFAULT_CANVAS_HEIGHT = 660
DEFAULT_SCALE_PX_PER_MM = 24.0
DEFAULT_BODY_COLOR = "#24201c"
DEFAULT_LEAD_COLOR = "#d4af72"
COMPONENT_CLEARANCE_MM = 0.1
SNAP_ANGLE_THRESHOLD_DEG = 12.0
SNAP_HOVER_DELAY_MS = 2000
POINT_AXIS_SNAP_THRESHOLD_PX = 10.0


@dataclass(slots=True)
class LeadProfile:
    points_mm: list[tuple[float, float]]
    closed: bool = True


@dataclass(slots=True)
class SideSettings:
    name: str
    count: int
    pitch_mm: float
    pitch_axis: str = "x"
    rotation_x_deg: float = 0.0
    rotation_y_deg: float = 0.0
    rotation_z_deg: float = 0.0


def read_bridge_payload(bridge_path: Path) -> dict:
    if not bridge_path.exists():
        return {}
    try:
        payload = json.loads(bridge_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_bridge_payload(bridge_path: Path, payload: dict) -> None:
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def _point_in_triangle(
    point: tuple[float, float],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    px, py = point
    (ax, ay), (bx, by), (cx, cy) = triangle
    denominator = ((by - cy) * (ax - cx)) + ((cx - bx) * (ay - cy))
    if abs(denominator) < 1e-12:
        return False
    alpha = (((by - cy) * (px - cx)) + ((cx - bx) * (py - cy))) / denominator
    beta = (((cy - ay) * (px - cx)) + ((ax - cx) * (py - cy))) / denominator
    gamma = 1.0 - alpha - beta
    return alpha >= -1e-9 and beta >= -1e-9 and gamma >= -1e-9


def _simplify_profile_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    simplified: list[tuple[float, float]] = []
    for point in points:
        if not simplified or point != simplified[-1]:
            simplified.append(point)
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified.pop()
    changed = True
    while changed and len(simplified) >= 3:
        changed = False
        filtered: list[tuple[float, float]] = []
        count = len(simplified)
        for index, current in enumerate(simplified):
            prev_point = simplified[index - 1]
            next_point = simplified[(index + 1) % count]
            cross = (
                (current[0] - prev_point[0]) * (next_point[1] - current[1])
                - (current[1] - prev_point[1]) * (next_point[0] - current[0])
            )
            if abs(cross) < 1e-9:
                changed = True
                continue
            filtered.append(current)
        simplified = filtered
    return simplified


def _triangulate_polygon(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    simplified = _simplify_profile_points(points)
    if len(simplified) < 3:
        raise ValueError("Closed shapes need at least 3 unique points.")
    winding = 1.0 if _signed_area(simplified) > 0.0 else -1.0
    remaining = list(range(len(simplified)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear_found = False
        for offset, current in enumerate(remaining):
            prev_index = remaining[offset - 1]
            next_index = remaining[(offset + 1) % len(remaining)]
            ax, ay = simplified[prev_index]
            bx, by = simplified[current]
            cx, cy = simplified[next_index]
            cross = ((bx - ax) * (cy - ay)) - ((by - ay) * (cx - ax))
            if (cross * winding) <= 1e-9:
                continue
            triangle = (simplified[prev_index], simplified[current], simplified[next_index])
            contains_vertex = any(
                candidate not in (prev_index, current, next_index)
                and _point_in_triangle(simplified[candidate], triangle)
                for candidate in remaining
            )
            if contains_vertex:
                continue
            triangles.append((prev_index, current, next_index))
            del remaining[offset]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Failed to triangulate shape. Check for self-intersections.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def extrude_closed_polygon(points_mm: list[tuple[float, float]], height_mm: float) -> trimesh.Trimesh:
    simplified = _simplify_profile_points(points_mm)
    triangles = _triangulate_polygon(simplified)
    vertex_count = len(simplified)
    vertices: list[list[float]] = []
    for z_coord in (0.0, height_mm):
        for x_coord, y_coord in simplified:
            vertices.append([x_coord, y_coord, z_coord])
    faces: list[list[int]] = []
    for a_index, b_index, c_index in triangles:
        faces.append([a_index, c_index, b_index])
        faces.append([a_index + vertex_count, b_index + vertex_count, c_index + vertex_count])
    for edge_index in range(vertex_count):
        next_index = (edge_index + 1) % vertex_count
        bottom_a = edge_index
        bottom_b = next_index
        top_a = edge_index + vertex_count
        top_b = next_index + vertex_count
        faces.append([bottom_a, top_a, top_b])
        faces.append([bottom_a, top_b, bottom_b])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def rotation_matrix_xyz(rotation_deg_xyz: tuple[float, float, float]) -> list[list[float]]:
    rot_x = math.radians(rotation_deg_xyz[0])
    rot_y = math.radians(rotation_deg_xyz[1])
    rot_z = math.radians(rotation_deg_xyz[2])
    cx, sx = math.cos(rot_x), math.sin(rot_x)
    cy, sy = math.cos(rot_y), math.sin(rot_y)
    cz, sz = math.cos(rot_z), math.sin(rot_z)
    rx = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    ry = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rz = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix_multiply(matrix_multiply(rz, ry), rx)


def identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_transform(dx: float, dy: float, dz: float) -> list[list[float]]:
    matrix = identity_transform()
    matrix[0][3] = dx
    matrix[1][3] = dy
    matrix[2][3] = dz
    return matrix


def matrix_multiply(a_mat: list[list[float]], b_mat: list[list[float]]) -> list[list[float]]:
    result = [[0.0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            result[row][col] = sum(a_mat[row][idx] * b_mat[idx][col] for idx in range(4))
    return result


def apply_transform_to_mesh(mesh: trimesh.Trimesh, transform: list[list[float]]) -> trimesh.Trimesh:
    transformed = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    transformed.apply_transform(transform)
    return transformed


def build_ic_payload(
    profile_points_mm: list[tuple[float, float]],
    *,
    leg_length_mm: float,
    body_width_mm: float,
    body_depth_mm: float,
    body_height_mm: float,
    side_settings: list[SideSettings],
    status_message: str = "",
) -> dict:
    return {
        "profile_points_mm": [list(point) for point in profile_points_mm],
        "leg_length_mm": leg_length_mm,
        "body_width_mm": body_width_mm,
        "body_depth_mm": body_depth_mm,
        "body_height_mm": body_height_mm,
        "side_settings": [asdict(side) for side in side_settings],
        "status_message": status_message,
    }


def _safe_stage_slug(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")
    return slug or "stage"


def _coerce_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def build_ic_meshes(payload: dict) -> tuple[trimesh.Trimesh | None, list[tuple[str, trimesh.Trimesh]]]:
    raw_points = payload.get("profile_points_mm", [])
    profile_points = [tuple(point) for point in raw_points if isinstance(point, list | tuple) and len(point) == 2]
    if len(profile_points) < 3:
        return None, []
    leg_length_mm = float(payload.get("leg_length_mm", 0.0))
    if leg_length_mm <= 0.0:
        raise ValueError("Lead length must be greater than 0 mm.")
    body_width_mm = float(payload.get("body_width_mm", 0.0))
    body_depth_mm = float(payload.get("body_depth_mm", 0.0))
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    if body_width_mm <= 0.0 or body_depth_mm <= 0.0 or body_height_mm <= 0.0:
        raise ValueError("Body width, depth, and height must be positive.")
    lead_offset_mm = float(payload.get("lead_offset_mm", 0.0))

    base_leg = extrude_closed_polygon(profile_points, leg_length_mm)
    side_payloads = payload.get("side_settings", [])
    side_meshes: list[tuple[str, trimesh.Trimesh]] = []
    for side_payload in side_payloads:
        side = SideSettings(**side_payload)
        if side.count <= 0:
            continue
        if side.pitch_mm <= 0.0:
            raise ValueError(f"{side.name} pitch must be greater than 0 mm.")
        spread = (side.count - 1) * side.pitch_mm
        pitch_axis = side.pitch_axis.lower()
        for index in range(side.count):
            position = (index * side.pitch_mm) - (spread / 2.0)
            rotated_mesh = apply_transform_to_mesh(
                base_leg,
                matrix_multiply(
                    rotation_matrix_xyz((side.rotation_x_deg, side.rotation_y_deg, side.rotation_z_deg)),
                    (
                        rotation_matrix_xyz((90.0, 0.0, 0.0))
                        if side.name == "Top"
                        else rotation_matrix_xyz((-90.0, 0.0, 0.0))
                        if side.name == "Bottom"
                        else rotation_matrix_xyz((90.0, 0.0, 90.0))
                        if side.name == "Left"
                        else rotation_matrix_xyz((90.0, 0.0, -90.0))
                    ),
                ),
            )
            min_corner = rotated_mesh.bounds[0].tolist()
            max_corner = rotated_mesh.bounds[1].tolist()
            center = [
                (min_corner[0] + max_corner[0]) / 2.0,
                (min_corner[1] + max_corner[1]) / 2.0,
                (min_corner[2] + max_corner[2]) / 2.0,
            ]

            if side.name == "Top":
                target = [0.0, body_depth_mm / 2.0, body_height_mm / 2.0]
                target[{"x": 0, "y": 1, "z": 2}.get(pitch_axis, 0)] += position
                translation = [
                    target[0] - center[0],
                    target[1] - min_corner[1] + lead_offset_mm,
                    target[2] - center[2],
                ]
            elif side.name == "Bottom":
                target = [0.0, -body_depth_mm / 2.0, body_height_mm / 2.0]
                target[{"x": 0, "y": 1, "z": 2}.get(pitch_axis, 0)] += position
                translation = [
                    target[0] - center[0],
                    target[1] - max_corner[1] - lead_offset_mm,
                    target[2] - center[2],
                ]
            elif side.name == "Left":
                target = [-body_width_mm / 2.0, 0.0, body_height_mm / 2.0]
                target[{"x": 0, "y": 1, "z": 2}.get(pitch_axis, 1)] += position
                translation = [
                    target[0] - max_corner[0] - lead_offset_mm,
                    target[1] - center[1],
                    target[2] - center[2],
                ]
            else:
                target = [body_width_mm / 2.0, 0.0, body_height_mm / 2.0]
                target[{"x": 0, "y": 1, "z": 2}.get(pitch_axis, 1)] += position
                translation = [
                    target[0] - min_corner[0] + lead_offset_mm,
                    target[1] - center[1],
                    target[2] - center[2],
                ]
            side_meshes.append((side.name, apply_transform_to_mesh(rotated_mesh, translation_transform(*translation))))

    body_mesh = trimesh.creation.box(extents=(body_width_mm, body_depth_mm, body_height_mm))
    body_mesh.apply_translation([0.0, 0.0, body_height_mm / 2.0])
    return body_mesh, side_meshes


def build_body_plane_mesh(payload: dict, thickness_mm: float = 0.08) -> trimesh.Trimesh | None:
    body_width_mm = float(payload.get("body_width_mm", 0.0))
    body_depth_mm = float(payload.get("body_depth_mm", 0.0))
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    if body_width_mm <= 0.0 or body_depth_mm <= 0.0:
        return None
    plane_mesh = trimesh.creation.box(
        extents=(body_width_mm, body_depth_mm, max(thickness_mm, 0.01)),
    )
    plane_mesh.apply_translation([0.0, 0.0, -(max(thickness_mm, 0.01) / 2.0) - COMPONENT_CLEARANCE_MM])
    return plane_mesh


def build_die_leadframe_mesh(
    payload: dict,
    side_meshes: list[tuple[str, trimesh.Trimesh]],
) -> tuple[trimesh.Trimesh | None, dict[str, float]]:
    body_width_mm = float(payload.get("body_width_mm", 0.0))
    body_depth_mm = float(payload.get("body_depth_mm", 0.0))
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    ratio_percent = float(payload.get("die_leadframe_ratio_percent", 80.0))
    clearance_mm = max(0.0, float(payload.get("die_leadframe_clearance_mm", 0.05)))
    thickness_mm = max(0.01, float(payload.get("die_leadframe_thickness_mm", 0.08)))
    if body_width_mm <= 0.0 or body_depth_mm <= 0.0:
        return None, {}

    body_left = -body_width_mm / 2.0
    body_right = body_width_mm / 2.0
    body_bottom = -body_depth_mm / 2.0
    body_top = body_depth_mm / 2.0

    left_limit = body_left
    right_limit = body_right
    bottom_limit = body_bottom
    top_limit = body_top

    for side_name, mesh in side_meshes:
        min_corner = mesh.bounds[0].tolist()
        max_corner = mesh.bounds[1].tolist()
        if side_name == "Left":
            left_limit = max(left_limit, max_corner[0] + clearance_mm)
        elif side_name == "Right":
            right_limit = min(right_limit, min_corner[0] - clearance_mm)
        elif side_name == "Top":
            top_limit = min(top_limit, min_corner[1] - clearance_mm)
        elif side_name == "Bottom":
            bottom_limit = max(bottom_limit, max_corner[1] + clearance_mm)

    max_half_width = max(0.0, min(abs(left_limit), abs(right_limit)))
    max_half_depth = max(0.0, min(abs(bottom_limit), abs(top_limit)))
    desired_half_width = (body_width_mm * max(1.0, min(100.0, ratio_percent)) / 100.0) / 2.0
    desired_half_depth = (body_depth_mm * max(1.0, min(100.0, ratio_percent)) / 100.0) / 2.0
    final_half_width = min(desired_half_width, max_half_width)
    final_half_depth = min(desired_half_depth, max_half_depth)
    if final_half_width <= 0.0 or final_half_depth <= 0.0:
        return None, {
            "final_width_mm": 0.0,
            "final_depth_mm": 0.0,
            "ratio_percent": ratio_percent,
        }

    mesh = trimesh.creation.box(
        extents=(final_half_width * 2.0, final_half_depth * 2.0, thickness_mm),
    )
    mesh.apply_translation([0.0, 0.0, body_height_mm + COMPONENT_CLEARANCE_MM + (thickness_mm / 2.0)])
    return mesh, {
        "final_width_mm": final_half_width * 2.0,
        "final_depth_mm": final_half_depth * 2.0,
        "ratio_percent": ratio_percent,
        "clearance_mm": clearance_mm,
        "thickness_mm": thickness_mm,
    }


def build_silicon_die_mesh(
    payload: dict,
    die_leadframe_mesh: trimesh.Trimesh | None,
    die_info: dict[str, float],
) -> tuple[trimesh.Trimesh | None, dict[str, float]]:
    if die_leadframe_mesh is None:
        return None, {}
    leadframe_width_mm = float(die_info.get("final_width_mm", 0.0))
    leadframe_depth_mm = float(die_info.get("final_depth_mm", 0.0))
    if leadframe_width_mm <= 0.0 or leadframe_depth_mm <= 0.0:
        return None, {}

    requested_width_mm = max(0.01, float(payload.get("silicon_die_width_mm", leadframe_width_mm * 0.6)))
    requested_depth_mm = max(0.01, float(payload.get("silicon_die_depth_mm", leadframe_depth_mm * 0.6)))
    thickness_mm = max(0.01, float(payload.get("silicon_die_thickness_mm", 0.12)))
    final_width_mm = min(requested_width_mm, leadframe_width_mm)
    final_depth_mm = min(requested_depth_mm, leadframe_depth_mm)

    min_corner = die_leadframe_mesh.bounds[0].tolist()
    max_corner = die_leadframe_mesh.bounds[1].tolist()
    center_x = (min_corner[0] + max_corner[0]) / 2.0
    center_y = (min_corner[1] + max_corner[1]) / 2.0
    top_z = max_corner[2]

    mesh = trimesh.creation.box(extents=(final_width_mm, final_depth_mm, thickness_mm))
    mesh.apply_translation([center_x, center_y, top_z + COMPONENT_CLEARANCE_MM + (thickness_mm / 2.0)])
    return mesh, {
        "final_width_mm": final_width_mm,
        "final_depth_mm": final_depth_mm,
        "thickness_mm": thickness_mm,
    }


def build_leg_pick_markers(
    payload: dict,
    side_meshes: list[tuple[str, trimesh.Trimesh]],
) -> list[dict]:
    pick_distance_mm = max(0.0, float(payload.get("leg_pick_distance_mm", 0.2)))
    marker_size_mm = max(0.02, float(payload.get("leg_pick_marker_size_mm", 0.08)))
    marker_thickness_mm = min(marker_size_mm * 0.25, 0.02)
    markers: list[dict] = []
    for side_name, mesh in side_meshes:
        min_corner = mesh.bounds[0].tolist()
        max_corner = mesh.bounds[1].tolist()
        center = [
            (min_corner[0] + max_corner[0]) / 2.0,
            (min_corner[1] + max_corner[1]) / 2.0,
            max_corner[2] + COMPONENT_CLEARANCE_MM + (marker_thickness_mm / 2.0),
        ]
        if side_name == "Top":
            max_distance = max(0.0, max_corner[1] - min_corner[1])
            center[1] = min_corner[1] + min(pick_distance_mm, max_distance)
        elif side_name == "Bottom":
            max_distance = max(0.0, max_corner[1] - min_corner[1])
            center[1] = max_corner[1] - min(pick_distance_mm, max_distance)
        elif side_name == "Left":
            max_distance = max(0.0, max_corner[0] - min_corner[0])
            center[0] = max_corner[0] - min(pick_distance_mm, max_distance)
        else:
            max_distance = max(0.0, max_corner[0] - min_corner[0])
            center[0] = min_corner[0] + min(pick_distance_mm, max_distance)
        marker = trimesh.creation.box(extents=(marker_size_mm, marker_size_mm, marker_thickness_mm))
        marker.apply_translation(center)
        markers.append(
            {
                "side_name": side_name,
                "mesh": marker,
                "center_xy": (center[0], center[1]),
                "anchor_xyz": (center[0], center[1], max_corner[2] + COMPONENT_CLEARANCE_MM),
            }
        )
    return markers


def build_die_region_meshes_and_pick(
    payload: dict,
    silicon_die_mesh: trimesh.Trimesh | None,
    silicon_die_info: dict[str, float],
) -> tuple[list[dict], trimesh.Trimesh | None, dict[str, float]]:
    if silicon_die_mesh is None:
        return [], None, {}

    die_width_mm = float(silicon_die_info.get("final_width_mm", 0.0))
    die_depth_mm = float(silicon_die_info.get("final_depth_mm", 0.0))
    if die_width_mm <= 0.0 or die_depth_mm <= 0.0:
        return [], None, {}

    span_percent = max(1.0, min(100.0, float(payload.get("die_region_span_percent", 70.0))))
    depth_mm = max(0.01, float(payload.get("die_region_depth_mm", 0.15)))
    offset_mm = max(0.0, float(payload.get("die_region_offset_mm", 0.05)))
    min_corner = silicon_die_mesh.bounds[0].tolist()
    max_corner = silicon_die_mesh.bounds[1].tolist()
    center_x = (min_corner[0] + max_corner[0]) / 2.0
    center_y = (min_corner[1] + max_corner[1]) / 2.0
    top_z = max_corner[2]
    thickness_mm = 0.01

    horizontal_span = die_width_mm * (span_percent / 100.0)
    vertical_span = die_depth_mm * (span_percent / 100.0)
    max_vertical_band = max(0.0, ((die_depth_mm - vertical_span) / 2.0) - offset_mm)
    max_horizontal_band = max(0.0, ((die_width_mm - horizontal_span) / 2.0) - offset_mm)
    top_bottom_depth = min(depth_mm, max_vertical_band) if max_vertical_band > 0.0 else 0.0
    left_right_depth = min(depth_mm, max_horizontal_band) if max_horizontal_band > 0.0 else 0.0

    side_counts = {
        str(item.get("name", "")).strip().title(): max(0, int(item.get("count", 0)))
        for item in payload.get("side_settings", [])
        if isinstance(item, dict)
    }

    regions: list[dict] = []

    def add_segmented_regions(side_name: str, count: int, span_length: float, band_depth: float) -> None:
        if count <= 0 or span_length <= 0.0 or band_depth <= 0.0:
            return
        segment_length = span_length / count
        for segment_index in range(count):
            if side_name in {"Top", "Bottom"}:
                segment_center_x = (center_x - (span_length / 2.0)) + (segment_length * (segment_index + 0.5))
                segment_center_y = (
                    max_corner[1] - offset_mm - (band_depth / 2.0)
                    if side_name == "Top"
                    else min_corner[1] + offset_mm + (band_depth / 2.0)
                )
                region = trimesh.creation.box(extents=(segment_length, band_depth, thickness_mm))
                region.apply_translation([segment_center_x, segment_center_y, top_z + COMPONENT_CLEARANCE_MM + (thickness_mm / 2.0)])
            else:
                segment_center_x = (
                    min_corner[0] + offset_mm + (band_depth / 2.0)
                    if side_name == "Left"
                    else max_corner[0] - offset_mm - (band_depth / 2.0)
                )
                segment_center_y = (center_y - (span_length / 2.0)) + (segment_length * (segment_index + 0.5))
                region = trimesh.creation.box(extents=(band_depth, segment_length, thickness_mm))
                region.apply_translation([segment_center_x, segment_center_y, top_z + COMPONENT_CLEARANCE_MM + (thickness_mm / 2.0)])
            regions.append(
                {
                    "side_name": side_name,
                    "section_index": segment_index + 1,
                    "mesh": region,
                    "center_xy": (segment_center_x, segment_center_y),
                    "anchor_xyz": (segment_center_x, segment_center_y, top_z + COMPONENT_CLEARANCE_MM + thickness_mm),
                }
            )

    add_segmented_regions("Top", side_counts.get("Top", 0), horizontal_span, top_bottom_depth)
    add_segmented_regions("Bottom", side_counts.get("Bottom", 0), horizontal_span, top_bottom_depth)
    add_segmented_regions("Left", side_counts.get("Left", 0), vertical_span, left_right_depth)
    add_segmented_regions("Right", side_counts.get("Right", 0), vertical_span, left_right_depth)

    selected_region = str(payload.get("die_pick_region", "Top")).strip().title()
    selected_section_index = max(1, int(float(payload.get("die_pick_section_index", 1))))
    selected_percent = max(0.0, min(100.0, float(payload.get("die_pick_position_percent", 50.0))))
    point_size_mm = max(0.02, float(payload.get("die_pick_marker_size_mm", 0.06)))
    selected_mesh: trimesh.Trimesh | None = None
    for region_data in regions:
        side_name = str(region_data["side_name"])
        section_index = int(region_data["section_index"])
        region_mesh = region_data["mesh"]
        if side_name != selected_region or section_index != selected_section_index:
            continue
        region_min = region_mesh.bounds[0].tolist()
        region_max = region_mesh.bounds[1].tolist()
        if side_name in {"Top", "Bottom"}:
            pick_x = region_min[0] + ((region_max[0] - region_min[0]) * (selected_percent / 100.0))
            pick_y = (region_min[1] + region_max[1]) / 2.0
        else:
            pick_x = (region_min[0] + region_max[0]) / 2.0
            pick_y = region_min[1] + ((region_max[1] - region_min[1]) * (selected_percent / 100.0))
        pick_z = region_max[2] + COMPONENT_CLEARANCE_MM + 0.005
        selected_mesh = trimesh.creation.box(extents=(point_size_mm, point_size_mm, 0.01))
        selected_mesh.apply_translation([pick_x, pick_y, pick_z])
        break

    return regions, selected_mesh, {
        "span_percent": span_percent,
        "depth_mm": depth_mm,
        "offset_mm": offset_mm,
        "top_bottom_depth_mm": top_bottom_depth,
        "left_right_depth_mm": left_right_depth,
        "region_count": len(regions),
    }


def _clockwise_indexed_items(items: list[dict]) -> list[dict]:
    side_order = {"Top": 0, "Right": 1, "Bottom": 2, "Left": 3}

    def sort_key(item: dict) -> tuple[float, float]:
        side_name = str(item.get("side_name", "")).title()
        x_coord, y_coord = item.get("center_xy", (0.0, 0.0))
        order = side_order.get(side_name, 99)
        if side_name == "Top":
            secondary = x_coord
        elif side_name == "Right":
            secondary = -y_coord
        elif side_name == "Bottom":
            secondary = -x_coord
        elif side_name == "Left":
            secondary = y_coord
        else:
            secondary = x_coord
        return (order, secondary)

    return sorted(items, key=sort_key)


def _sample_cubic_bezier(
    start: tuple[float, float, float],
    control_1: tuple[float, float, float],
    control_2: tuple[float, float, float],
    end: tuple[float, float, float],
    sample_count: int,
) -> list[list[float]]:
    points: list[list[float]] = []
    for sample_index in range(sample_count + 1):
        t_value = sample_index / sample_count
        one_minus_t = 1.0 - t_value
        x_coord = (
            (one_minus_t ** 3) * start[0]
            + (3.0 * (one_minus_t ** 2) * t_value * control_1[0])
            + (3.0 * one_minus_t * (t_value ** 2) * control_2[0])
            + ((t_value ** 3) * end[0])
        )
        y_coord = (
            (one_minus_t ** 3) * start[1]
            + (3.0 * (one_minus_t ** 2) * t_value * control_1[1])
            + (3.0 * one_minus_t * (t_value ** 2) * control_2[1])
            + ((t_value ** 3) * end[1])
        )
        z_coord = (
            (one_minus_t ** 3) * start[2]
            + (3.0 * (one_minus_t ** 2) * t_value * control_1[2])
            + (3.0 * one_minus_t * (t_value ** 2) * control_2[2])
            + ((t_value ** 3) * end[2])
        )
        points.append([x_coord, y_coord, z_coord])
    return points


def _sample_catmull_rom_spline(control_points: list[tuple[float, float, float]], sample_count: int) -> list[list[float]]:
    if len(control_points) < 2:
        return [list(point) for point in control_points]
    if len(control_points) == 2:
        start_point, end_point = control_points
        return [
            [
                start_point[0] + ((end_point[0] - start_point[0]) * (index / sample_count)),
                start_point[1] + ((end_point[1] - start_point[1]) * (index / sample_count)),
                start_point[2] + ((end_point[2] - start_point[2]) * (index / sample_count)),
            ]
            for index in range(sample_count + 1)
        ]

    extended_points = [control_points[0], *control_points, control_points[-1]]
    segment_count = len(control_points) - 1
    spline_points: list[list[float]] = []

    for sample_index in range(sample_count + 1):
        t_global = sample_index / sample_count
        segment_position = min(segment_count - 1e-9, t_global * segment_count)
        segment_index = min(segment_count - 1, int(segment_position))
        t_local = segment_position - segment_index

        p0 = extended_points[segment_index]
        p1 = extended_points[segment_index + 1]
        p2 = extended_points[segment_index + 2]
        p3 = extended_points[segment_index + 3]

        point_coords: list[float] = []
        for axis_index in range(3):
            coord = 0.5 * (
                (2.0 * p1[axis_index])
                + ((-p0[axis_index] + p2[axis_index]) * t_local)
                + ((2.0 * p0[axis_index] - 5.0 * p1[axis_index] + 4.0 * p2[axis_index] - p3[axis_index]) * (t_local ** 2))
                + ((-p0[axis_index] + 3.0 * p1[axis_index] - 3.0 * p2[axis_index] + p3[axis_index]) * (t_local ** 3))
            )
            point_coords.append(coord)
        spline_points.append(point_coords)

    spline_points[0] = list(control_points[0])
    spline_points[-1] = list(control_points[-1])
    return spline_points


def _rotate_points_xy(points_xy: list[tuple[float, float]], angle_rad: float) -> list[tuple[float, float]]:
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return [
        (
            (x_coord * cos_angle) - (y_coord * sin_angle),
            (x_coord * sin_angle) + (y_coord * cos_angle),
        )
        for x_coord, y_coord in points_xy
    ]


def _trim_polyline_from_end(points: list[list[float]], trim_distance_mm: float) -> list[list[float]]:
    if trim_distance_mm <= 1e-9 or len(points) < 2:
        return [list(point) for point in points]
    remaining_trim = trim_distance_mm
    trimmed = [list(point) for point in points]
    while len(trimmed) >= 2 and remaining_trim > 1e-9:
        last_point = trimmed[-1]
        previous_point = trimmed[-2]
        segment_length = math.dist(last_point, previous_point)
        if segment_length <= 1e-9:
            trimmed.pop()
            continue
        if segment_length <= remaining_trim:
            trimmed.pop()
            remaining_trim -= segment_length
            continue
        ratio = (segment_length - remaining_trim) / segment_length
        trimmed[-1] = [
            previous_point[0] + ((last_point[0] - previous_point[0]) * ratio),
            previous_point[1] + ((last_point[1] - previous_point[1]) * ratio),
            previous_point[2] + ((last_point[2] - previous_point[2]) * ratio),
        ]
        remaining_trim = 0.0
    return trimmed if len(trimmed) >= 2 else [list(point) for point in points[:2]]


def _vector_add(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a_vec[0] + b_vec[0], a_vec[1] + b_vec[1], a_vec[2] + b_vec[2])


def _vector_sub(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a_vec[0] - b_vec[0], a_vec[1] - b_vec[1], a_vec[2] - b_vec[2])


def _vector_scale(vec: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _vector_length(vec: tuple[float, float, float]) -> float:
    return math.sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]))


def _vector_normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _vector_length(vec)
    if length <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _vector_cross(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (a_vec[1] * b_vec[2]) - (a_vec[2] * b_vec[1]),
        (a_vec[2] * b_vec[0]) - (a_vec[0] * b_vec[2]),
        (a_vec[0] * b_vec[1]) - (a_vec[1] * b_vec[0]),
    )


def _vector_lerp(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float], t_value: float) -> tuple[float, float, float]:
    return (
        a_vec[0] + ((b_vec[0] - a_vec[0]) * t_value),
        a_vec[1] + ((b_vec[1] - a_vec[1]) * t_value),
        a_vec[2] + ((b_vec[2] - a_vec[2]) * t_value),
    )


def build_connection_paths(
    payload: dict,
    leg_markers: list[dict],
    die_regions: list[dict],
    ball_bond_meshes: list[dict] | None = None,
) -> list[dict]:
    indexed_leg_markers = _clockwise_indexed_items(leg_markers)
    indexed_die_regions = _clockwise_indexed_items(die_regions)
    pair_count = min(len(indexed_leg_markers), len(indexed_die_regions))
    if pair_count <= 0:
        return []

    arc_height_mm = max(0.0, float(payload.get("arc_height_mm", 0.5)))
    arc_xy_noise_mm = max(0.0, float(payload.get("arc_xy_noise_mm", 0.0)))
    point_spacing_mm = max(0.01, float(payload.get("wire_arc_point_spacing_mm", 0.08)))
    wire_rise_z_mm = max(0.01, float(payload.get("wire_rise_z_mm", 0.12)))
    wedge_approach_run_mm = max(0.02, float(payload.get("wedge_approach_run_mm", 0.18)))
    indexed_ball_bonds = ball_bond_meshes or []
    arcs: list[dict] = []

    for index in range(pair_count):
        leg_data = indexed_leg_markers[index]
        die_data = indexed_die_regions[index]
        bond_data = indexed_ball_bonds[index] if index < len(indexed_ball_bonds) else {}
        start = bond_data.get("wire_start_xyz", die_data.get("anchor_xyz", (0.0, 0.0, 0.0)))
        end = leg_data.get("anchor_xyz", (0.0, 0.0, 0.0))
        start_x, start_y, start_z = start
        end_x, end_y, end_z = end
        mid_x = (start_x + end_x) / 2.0
        mid_y = (start_y + end_y) / 2.0
        mid_z = max(start_z, end_z) + arc_height_mm

        dx = end_x - start_x
        dy = end_y - start_y
        planar_length = math.hypot(dx, dy)
        landing_run_mm = 0.0
        landing_x = end_x
        landing_y = end_y
        if planar_length > 1e-9:
            perp_x = -dy / planar_length
            perp_y = dx / planar_length
            direction = -1.0 if index % 2 else 1.0
            mid_x += perp_x * arc_xy_noise_mm * direction
            mid_y += perp_y * arc_xy_noise_mm * direction
            landing_run_mm = min(wedge_approach_run_mm, planar_length * 0.55)
            landing_x = end_x - ((dx / planar_length) * landing_run_mm)
            landing_y = end_y - ((dy / planar_length) * landing_run_mm)

        rise_point = (start_x, start_y, start_z + wire_rise_z_mm)
        arc_control_1 = (start_x, start_y, mid_z)
        arc_control_2 = (landing_x, landing_y, end_z)
        approximate_length = (
            math.dist(start, rise_point)
            + math.dist(rise_point, arc_control_1)
            + math.dist(arc_control_1, arc_control_2)
            + math.dist(arc_control_2, end)
        )
        sample_count = max(8, int(math.ceil(approximate_length / point_spacing_mm)))

        rise_length = math.dist(start, rise_point)
        rise_sample_count = max(2, int(math.ceil(rise_length / point_spacing_mm)))
        arc_sample_count = max(6, sample_count - rise_sample_count + 1)

        rise_points = [
            [
                start_x,
                start_y,
                start_z + ((wire_rise_z_mm * sample_index) / rise_sample_count),
            ]
            for sample_index in range(rise_sample_count + 1)
        ]
        arc_points = _sample_cubic_bezier(rise_point, arc_control_1, arc_control_2, end, arc_sample_count)
        points = rise_points[:-1] + arc_points

        arcs.append(
            {
                "leg_index": index + 1,
                "die_index": index + 1,
                "points": points,
                "leg_side_name": leg_data.get("side_name", ""),
                "die_side_name": die_data.get("side_name", ""),
                "die_section_index": die_data.get("section_index", 0),
                "landing_run_mm": landing_run_mm,
            }
        )

    return arcs


def build_ball_bond_meshes(
    payload: dict,
    leg_markers: list[dict],
    die_regions: list[dict],
) -> list[dict]:
    indexed_leg_markers = _clockwise_indexed_items(leg_markers)
    indexed_die_regions = _clockwise_indexed_items(die_regions)
    pair_count = min(len(indexed_leg_markers), len(indexed_die_regions))
    if pair_count <= 0:
        return []

    diameter_mm = max(0.02, float(payload.get("ball_bond_diameter_mm", 0.12)))
    rectangle_length_mm = max(0.01, float(payload.get("ball_bond_length_mm", 0.08)))
    revolution_steps = max(6, int(float(payload.get("ball_bond_revolution_steps", 24))))
    radius_mm = diameter_mm / 2.0

    arc_samples = max(8, revolution_steps // 2)
    profile_points: list[list[float]] = [[0.0, 0.0], [rectangle_length_mm, 0.0]]
    for sample_index in range(1, arc_samples):
        angle = (-math.pi / 2.0) + ((math.pi * sample_index) / arc_samples)
        profile_points.append(
            [
                rectangle_length_mm + (radius_mm * math.cos(angle)),
                radius_mm + (radius_mm * math.sin(angle)),
            ]
        )
    profile_points.extend([[rectangle_length_mm, diameter_mm], [0.0, diameter_mm]])

    base_mesh = trimesh.creation.revolve(
        profile_points,
        angle=(2.0 * math.pi),
        cap=True,
        sections=revolution_steps,
    )

    meshes: list[dict] = []
    for index in range(pair_count):
        die_data = indexed_die_regions[index]
        anchor_x, anchor_y, anchor_z = die_data.get("anchor_xyz", (0.0, 0.0, 0.0))
        bond_mesh = trimesh.Trimesh(vertices=base_mesh.vertices.copy(), faces=base_mesh.faces.copy(), process=False)
        bond_mesh.apply_translation([anchor_x, anchor_y, anchor_z])
        min_corner = bond_mesh.bounds[0].tolist()
        max_corner = bond_mesh.bounds[1].tolist()
        meshes.append(
            {
                "mesh": bond_mesh,
                "die_index": index + 1,
                "side_name": die_data.get("side_name", ""),
                "section_index": die_data.get("section_index", 0),
                "wire_start_xyz": (
                    (min_corner[0] + max_corner[0]) / 2.0,
                    (min_corner[1] + max_corner[1]) / 2.0,
                    max_corner[2],
                ),
            }
        )
    return meshes


def build_tube_connection_meshes(
    payload: dict,
    connection_paths: list[dict],
    *,
    trim_end_distance_mm: float = 0.0,
) -> list[dict]:
    tube_side_count = max(3, int(float(payload.get("wire_tube_side_count", 10))))
    wire_radius_mm = max(0.005, float(payload.get("wire_diameter_mm", 0.03)) / 2.0)
    tube_meshes: list[dict] = []
    for path_data in connection_paths:
        points = _trim_polyline_from_end(path_data.get("points", []), trim_end_distance_mm)
        if len(points) < 2:
            continue
        segment_meshes: list[trimesh.Trimesh] = []
        for start_point, end_point in zip(points[:-1], points[1:]):
            segment_length = math.dist(start_point, end_point)
            if segment_length <= 1e-9:
                continue
            segment_meshes.append(
                trimesh.creation.cylinder(
                    radius=wire_radius_mm,
                    segment=[start_point, end_point],
                    sections=tube_side_count,
                )
            )
        if not segment_meshes:
            continue
        tube_meshes.append(
            {
                "mesh": trimesh.util.concatenate(segment_meshes),
                "leg_index": path_data["leg_index"],
                "die_index": path_data["die_index"],
                "leg_side_name": path_data["leg_side_name"],
                "die_side_name": path_data["die_side_name"],
                "die_section_index": path_data["die_section_index"],
            }
        )
    return tube_meshes


def build_wedge_bond_meshes(payload: dict, connection_paths: list[dict]) -> list[dict]:
    wedge_length_mm = max(0.04, float(payload.get("wedge_bond_length_mm", 0.18)))
    wedge_width_mm = max(0.02, float(payload.get("wedge_bond_width_mm", 0.08)))
    wedge_thickness_mm = max(0.005, float(payload.get("wedge_bond_thickness_mm", 0.02)))
    radius_mm = wedge_width_mm / 2.0
    straight_half_mm = max(0.0, (wedge_length_mm / 2.0) - radius_mm)
    arc_steps = 12

    profile_points: list[tuple[float, float]] = []
    for step_index in range(arc_steps + 1):
        angle = math.pi / 2.0 - ((math.pi * step_index) / arc_steps)
        profile_points.append(
            (
                straight_half_mm + (radius_mm * math.cos(angle)),
                radius_mm * math.sin(angle),
            )
        )
    for step_index in range(arc_steps + 1):
        angle = -math.pi / 2.0 + ((math.pi * step_index) / arc_steps)
        profile_points.append(
            (
                -straight_half_mm + (radius_mm * math.cos(angle)),
                radius_mm * math.sin(angle),
            )
        )

    base_mesh = extrude_closed_polygon(profile_points, wedge_thickness_mm)
    wedge_meshes: list[dict] = []
    for path_data in connection_paths:
        points = path_data.get("points", [])
        if len(points) < 2:
            continue
        end_point = points[-1]
        previous_point = points[max(0, len(points) - 3)]
        dx = end_point[0] - previous_point[0]
        dy = end_point[1] - previous_point[1]
        planar_length = math.hypot(dx, dy)
        if planar_length > 1e-9:
            unit_x = dx / planar_length
            unit_y = dy / planar_length
            angle_rad = math.atan2(dy, dx)
        else:
            unit_x = 1.0
            unit_y = 0.0
            angle_rad = 0.0
        # Align the incoming rounded end of the wedge with the wire direction.
        center_x = end_point[0] + (straight_half_mm * unit_x)
        center_y = end_point[1] + (straight_half_mm * unit_y)
        wedge_mesh = trimesh.Trimesh(vertices=base_mesh.vertices.copy(), faces=base_mesh.faces.copy(), process=False)
        wedge_mesh.apply_transform(rotation_matrix_xyz((0.0, 0.0, math.degrees(angle_rad))))
        wedge_mesh.apply_translation([center_x, center_y, end_point[2]])
        wedge_meshes.append(
            {
                "mesh": wedge_mesh,
                "leg_index": path_data["leg_index"],
                "die_index": path_data["die_index"],
                "leg_side_name": path_data["leg_side_name"],
                "die_side_name": path_data["die_side_name"],
                "die_section_index": path_data["die_section_index"],
            }
        )
    return wedge_meshes


def build_integrated_wire_terminal_meshes(payload: dict, connection_paths: list[dict]) -> list[dict]:
    tube_side_count = max(8, int(float(payload.get("wire_tube_side_count", 10))))
    wire_radius_mm = max(0.005, float(payload.get("wire_diameter_mm", 0.03)) / 2.0)
    wedge_length_mm = max(0.04, float(payload.get("wedge_bond_length_mm", 0.18)))
    wedge_width_mm = max(0.02, float(payload.get("wedge_bond_width_mm", 0.08)))
    wedge_thickness_mm = max(0.005, float(payload.get("wedge_bond_thickness_mm", 0.02)))
    transition_length_mm = max(
        wedge_length_mm * 0.45,
        min(float(payload.get("wedge_approach_run_mm", 0.18)), wedge_length_mm * 1.2),
    )

    integrated_meshes: list[dict] = []
    for path_data in connection_paths:
        base_points = [tuple(point) for point in path_data.get("points", [])]
        if len(base_points) < 2:
            continue

        terminal_direction = _vector_normalize(_vector_sub(base_points[-1], base_points[max(0, len(base_points) - 3)]))
        extension_step_mm = max(0.01, wedge_length_mm / 4.0)
        extension_count = max(2, int(math.ceil(wedge_length_mm / extension_step_mm)))
        extended_points = list(base_points)
        for extension_index in range(1, extension_count + 1):
            extended_points.append(
                _vector_add(base_points[-1], _vector_scale(terminal_direction, (wedge_length_mm * extension_index) / extension_count))
            )

        distances = [0.0]
        for start_point, end_point in zip(extended_points[:-1], extended_points[1:]):
            distances.append(distances[-1] + math.dist(start_point, end_point))
        total_length = distances[-1]
        transition_start_distance = max(0.0, total_length - (wedge_length_mm + transition_length_mm))
        wedge_start_distance = max(0.0, total_length - wedge_length_mm)

        sections: list[list[tuple[float, float, float]]] = []
        for index, center in enumerate(extended_points):
            if index == 0:
                tangent = _vector_normalize(_vector_sub(extended_points[1], extended_points[0]))
            elif index == len(extended_points) - 1:
                tangent = _vector_normalize(_vector_sub(extended_points[-1], extended_points[-2]))
            else:
                tangent = _vector_normalize(_vector_sub(extended_points[index + 1], extended_points[index - 1]))

            up_reference = (0.0, 0.0, 1.0)
            lateral = _vector_cross(up_reference, tangent)
            if _vector_length(lateral) <= 1e-9:
                lateral = (1.0, 0.0, 0.0)
            lateral = _vector_normalize(lateral)
            vertical = _vector_normalize(_vector_cross(tangent, lateral))

            distance_along = distances[index]
            if distance_along <= transition_start_distance:
                width_radius = wire_radius_mm
                height_radius = wire_radius_mm
            elif distance_along >= wedge_start_distance:
                width_radius = wedge_width_mm / 2.0
                height_radius = wedge_thickness_mm / 2.0
            else:
                morph_t = (distance_along - transition_start_distance) / max(1e-9, wedge_start_distance - transition_start_distance)
                width_radius = wire_radius_mm + (((wedge_width_mm / 2.0) - wire_radius_mm) * morph_t)
                height_radius = wire_radius_mm + (((wedge_thickness_mm / 2.0) - wire_radius_mm) * morph_t)

            section: list[tuple[float, float, float]] = []
            for side_index in range(tube_side_count):
                angle = (2.0 * math.pi * side_index) / tube_side_count
                offset = _vector_add(
                    _vector_scale(lateral, math.cos(angle) * width_radius),
                    _vector_scale(vertical, math.sin(angle) * height_radius),
                )
                section.append(_vector_add(center, offset))
            sections.append(section)

        vertices: list[list[float]] = []
        for section in sections:
            for vertex in section:
                vertices.append([vertex[0], vertex[1], vertex[2]])

        faces: list[list[int]] = []
        section_size = tube_side_count
        for section_index in range(len(sections) - 1):
            base_index = section_index * section_size
            next_index = (section_index + 1) * section_size
            for vertex_index in range(section_size):
                current_a = base_index + vertex_index
                current_b = base_index + ((vertex_index + 1) % section_size)
                next_a = next_index + vertex_index
                next_b = next_index + ((vertex_index + 1) % section_size)
                faces.append([current_a, next_a, next_b])
                faces.append([current_a, next_b, current_b])

        start_center_index = len(vertices)
        vertices.append(list(extended_points[0]))
        for vertex_index in range(section_size):
            next_vertex_index = (vertex_index + 1) % section_size
            faces.append([start_center_index, next_vertex_index, vertex_index])

        end_center_index = len(vertices)
        vertices.append(list(extended_points[-1]))
        end_base_index = (len(sections) - 1) * section_size
        for vertex_index in range(section_size):
            next_vertex_index = (vertex_index + 1) % section_size
            faces.append([end_center_index, end_base_index + vertex_index, end_base_index + next_vertex_index])

        integrated_meshes.append(
            {
                "mesh": trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                "leg_index": path_data["leg_index"],
                "die_index": path_data["die_index"],
                "leg_side_name": path_data["leg_side_name"],
                "die_side_name": path_data["die_side_name"],
                "die_section_index": path_data["die_section_index"],
            }
        )

    return integrated_meshes


def build_ball_bond_wire_assembly_meshes(
    ball_bond_meshes: list[dict],
    wire_meshes: list[dict],
) -> list[dict]:
    wire_map = {
        (int(item.get("leg_index", 0)), int(item.get("die_index", 0))): item
        for item in wire_meshes
    }
    assemblies: list[dict] = []
    for bond_data in ball_bond_meshes:
        die_index = int(bond_data.get("die_index", 0))
        key = (die_index, die_index)
        wire_data = wire_map.get(key)
        if wire_data is None:
            combined_mesh = bond_data["mesh"]
            leg_index = die_index
            leg_side_name = ""
        else:
            combined_mesh = trimesh.util.concatenate([bond_data["mesh"], wire_data["mesh"]])
            leg_index = int(wire_data.get("leg_index", die_index))
            leg_side_name = str(wire_data.get("leg_side_name", ""))
        assemblies.append(
            {
                "mesh": combined_mesh,
                "leg_index": leg_index,
                "die_index": die_index,
                "leg_side_name": leg_side_name,
                "die_side_name": str(bond_data.get("side_name", "")),
                "die_section_index": int(bond_data.get("section_index", 0)),
            }
        )
    return assemblies


def build_axis_meshes(payload: dict) -> list[tuple[str, trimesh.Trimesh, str]]:
    body_width_mm = max(0.0, float(payload.get("body_width_mm", 0.0)))
    body_depth_mm = max(0.0, float(payload.get("body_depth_mm", 0.0)))
    body_height_mm = max(0.0, float(payload.get("body_height_mm", 0.0)))
    max_dimension = max(body_width_mm, body_depth_mm, body_height_mm, 2.0)
    axis_length_mm = max_dimension * 0.75
    axis_thickness_mm = max(0.04, axis_length_mm * 0.03)
    half_length = axis_length_mm / 2.0

    x_axis = trimesh.creation.box(extents=(axis_length_mm, axis_thickness_mm, axis_thickness_mm))
    x_axis.apply_translation([half_length, 0.0, 0.0])

    y_axis = trimesh.creation.box(extents=(axis_thickness_mm, axis_length_mm, axis_thickness_mm))
    y_axis.apply_translation([0.0, half_length, 0.0])

    z_axis = trimesh.creation.box(extents=(axis_thickness_mm, axis_thickness_mm, axis_length_mm))
    z_axis.apply_translation([0.0, 0.0, half_length])

    return [
        ("X", x_axis, "#b91c1c"),
        ("Y", y_axis, "#15803d"),
        ("Z", z_axis, "#1d4ed8"),
    ]


def collect_export_meshes(payload: dict) -> list[trimesh.Trimesh]:
    current_step_index = int(payload.get("current_step_index", 0))
    meshes: list[trimesh.Trimesh] = []

    if current_step_index <= 1:
        lead_mesh = build_single_lead_mesh(payload)
        if lead_mesh is not None:
            meshes.append(lead_mesh)
        return meshes

    body_mesh, side_meshes = build_ic_meshes(payload)
    if current_step_index >= 3 and body_mesh is not None:
        meshes.append(body_mesh)

    meshes.extend(mesh for _side_name, mesh in side_meshes)

    die_leadframe_mesh, die_info = build_die_leadframe_mesh(payload, side_meshes)
    if current_step_index >= 5 and die_leadframe_mesh is not None:
        meshes.append(die_leadframe_mesh)

    silicon_die_mesh, silicon_die_info = build_silicon_die_mesh(payload, die_leadframe_mesh, die_info)
    if current_step_index >= 6 and silicon_die_mesh is not None:
        meshes.append(silicon_die_mesh)

    leg_pick_markers = build_leg_pick_markers(payload, side_meshes)
    die_regions, _die_pick_marker, _die_region_info = build_die_region_meshes_and_pick(payload, silicon_die_mesh, silicon_die_info)
    ball_bond_meshes = build_ball_bond_meshes(payload, leg_pick_markers, die_regions)
    connection_paths = build_connection_paths(payload, leg_pick_markers, die_regions, ball_bond_meshes)
    tube_connection_meshes = build_tube_connection_meshes(payload, connection_paths)
    integrated_terminal_meshes = build_integrated_wire_terminal_meshes(payload, connection_paths)
    ball_bond_wire_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, tube_connection_meshes)
    ball_bond_terminal_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, integrated_terminal_meshes)

    if current_step_index >= 10:
        meshes.extend(
            item["mesh"]
            for item in (
                ball_bond_terminal_meshes
                if current_step_index >= 12
                else ball_bond_wire_meshes
                if current_step_index >= 11
                else ball_bond_meshes
            )
        )

    return meshes


def build_single_lead_mesh(payload: dict) -> trimesh.Trimesh | None:
    raw_points = payload.get("profile_points_mm", [])
    profile_points = [tuple(point) for point in raw_points if isinstance(point, list | tuple) and len(point) == 2]
    if len(profile_points) < 3:
        return None
    leg_length_mm = float(payload.get("leg_length_mm", 0.0))
    if leg_length_mm <= 0.0:
        raise ValueError("Lead length must be greater than 0 mm.")
    return extrude_closed_polygon(profile_points, leg_length_mm)


class IcLeadViewer:
    def __init__(self, bridge_path: Path) -> None:
        self.bridge_path = bridge_path
        self.plotter = Plotter(
            title="IC Lead Placement Preview",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1200, 820),
        )
        self.info = Text2D("", pos="top-left", s=0.8, c="#2d241f", bg=None, font="Courier")
        self.actors: list = []
        self.last_signature: tuple | None = None
        self.hover_legend_callback_id: int | None = None
        self.ui_buttons: list = []
        self.ui_button_handlers: dict[int, callable] = {}
        self.ui_click_callback_id: int | None = None
        self.is_orthographic = False
        self.show_helper_objects = True
        self.helper_toggle_button: Text2D | None = None

    def _payload_signature(self, payload: dict) -> tuple:
        return (
            payload.get("current_step_index", 0),
            tuple(tuple(point) for point in payload.get("profile_points_mm", [])),
            payload.get("leg_length_mm", 0.0),
            payload.get("lead_offset_mm", 0.0),
            payload.get("die_leadframe_ratio_percent", 80.0),
            payload.get("die_leadframe_clearance_mm", 0.05),
            payload.get("die_leadframe_thickness_mm", 0.08),
            payload.get("silicon_die_width_mm", 0.0),
            payload.get("silicon_die_depth_mm", 0.0),
            payload.get("silicon_die_thickness_mm", 0.12),
            payload.get("leg_pick_distance_mm", 0.2),
            payload.get("leg_pick_marker_size_mm", 0.08),
            payload.get("die_region_span_percent", 70.0),
            payload.get("die_region_depth_mm", 0.15),
            payload.get("die_region_offset_mm", 0.05),
            payload.get("die_pick_region", "Top"),
            payload.get("die_pick_section_index", 1),
            payload.get("die_pick_position_percent", 50.0),
            payload.get("die_pick_marker_size_mm", 0.06),
            payload.get("arc_height_mm", 0.5),
            payload.get("arc_xy_noise_mm", 0.0),
            payload.get("wire_arc_point_spacing_mm", 0.08),
            payload.get("ball_bond_diameter_mm", 0.12),
            payload.get("ball_bond_length_mm", 0.08),
            payload.get("ball_bond_revolution_steps", 24),
            payload.get("wire_diameter_mm", 0.03),
            payload.get("wire_rise_z_mm", 0.12),
            payload.get("wire_tube_side_count", 10),
            payload.get("wedge_bond_length_mm", 0.18),
            payload.get("wedge_bond_width_mm", 0.08),
            payload.get("wedge_bond_thickness_mm", 0.02),
            payload.get("wedge_approach_run_mm", 0.18),
            payload.get("body_width_mm", 0.0),
            payload.get("body_depth_mm", 0.0),
            payload.get("body_height_mm", 0.0),
            tuple(
                (
                    side.get("name", ""),
                    side.get("count", 0),
                    side.get("pitch_mm", 0.0),
                    side.get("pitch_axis", "x"),
                    side.get("rotation_x_deg", 0.0),
                    side.get("rotation_y_deg", 0.0),
                    side.get("rotation_z_deg", 0.0),
                )
                for side in payload.get("side_settings", [])
            ),
            payload.get("status_message", ""),
        )

    def _build_scene(self, payload: dict) -> None:
        for actor in self.actors:
            self.plotter.remove(actor)
        self.actors.clear()

        status_message = str(payload.get("status_message", "")).strip()
        current_step_index = int(payload.get("current_step_index", 0))
        try:
            if current_step_index <= 1:
                lead_mesh = build_single_lead_mesh(payload)
                if lead_mesh is not None:
                    lead_actor = Mesh([lead_mesh.vertices.tolist(), lead_mesh.faces.tolist()]).c(DEFAULT_LEAD_COLOR).alpha(1.0)
                    lead_actor.info = "Lead Profile Extrusion"
                    self.actors.append(lead_actor)
                    self.plotter += lead_actor
                    summary = "Preview: single lead extrusion"
                else:
                    summary = "Draw a closed lead profile to preview the extrusion."
            else:
                body_mesh, side_meshes = build_ic_meshes(payload)
                body_plane_mesh = build_body_plane_mesh(payload)
                die_leadframe_mesh, die_info = build_die_leadframe_mesh(payload, side_meshes)
                silicon_die_mesh, silicon_die_info = build_silicon_die_mesh(payload, die_leadframe_mesh, die_info)
                leg_pick_markers = build_leg_pick_markers(payload, side_meshes)
                die_regions, die_pick_marker, die_region_info = build_die_region_meshes_and_pick(payload, silicon_die_mesh, silicon_die_info)
                ball_bond_meshes = build_ball_bond_meshes(payload, leg_pick_markers, die_regions)
                connection_paths = build_connection_paths(payload, leg_pick_markers, die_regions, ball_bond_meshes)
                tube_connection_meshes = build_tube_connection_meshes(payload, connection_paths)
                integrated_terminal_meshes = build_integrated_wire_terminal_meshes(payload, connection_paths)
                ball_bond_wire_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, tube_connection_meshes)
                ball_bond_terminal_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, integrated_terminal_meshes)
                if self.show_helper_objects:
                    for axis_name, axis_mesh, axis_color in build_axis_meshes(payload):
                        axis_actor = Mesh([axis_mesh.vertices.tolist(), axis_mesh.faces.tolist()]).c(axis_color).alpha(1.0)
                        axis_actor.info = f"{axis_name} Axis"
                        self.actors.append(axis_actor)
                        self.plotter += axis_actor
                if self.show_helper_objects and current_step_index >= 2 and body_plane_mesh is not None:
                    plane_actor = Mesh([body_plane_mesh.vertices.tolist(), body_plane_mesh.faces.tolist()]).c("#d9c9ad").alpha(1.0)
                    plane_actor.info = "Body Placement Plane"
                    self.actors.append(plane_actor)
                    self.plotter += plane_actor
                if current_step_index >= 3 and body_mesh is not None:
                    body_actor = Mesh([body_mesh.vertices.tolist(), body_mesh.faces.tolist()]).c(DEFAULT_BODY_COLOR).alpha(1.0)
                    body_actor.info = "Package Body"
                    self.actors.append(body_actor)
                    self.plotter += body_actor
                side_colors = {
                    "Top": "#d4af72",
                    "Bottom": "#e0bf8f",
                    "Left": "#c9975b",
                    "Right": "#e7cfa7",
                }
                for side_name, mesh in side_meshes:
                    actor = Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(side_colors.get(side_name, DEFAULT_LEAD_COLOR)).alpha(1.0)
                    actor.info = f"{side_name} Lead"
                    self.actors.append(actor)
                    self.plotter += actor
                body_status = "ready" if (current_step_index >= 3 and body_mesh is not None) else "hidden until final placement"
                plane_status = "xy placement plane shown" if body_plane_mesh is not None else "xy placement plane waiting"
                if current_step_index >= 5 and die_leadframe_mesh is not None:
                    die_actor = Mesh([die_leadframe_mesh.vertices.tolist(), die_leadframe_mesh.faces.tolist()]).c("#8e3f2b").alpha(1.0)
                    die_actor.info = "Leadframe"
                    self.actors.append(die_actor)
                    self.plotter += die_actor
                if current_step_index >= 6 and silicon_die_mesh is not None:
                    silicon_actor = Mesh([silicon_die_mesh.vertices.tolist(), silicon_die_mesh.faces.tolist()]).c("#232323").alpha(1.0)
                    silicon_actor.info = "Silicon Die"
                    self.actors.append(silicon_actor)
                    self.plotter += silicon_actor
                if self.show_helper_objects and current_step_index >= 7:
                    indexed_leg_markers = _clockwise_indexed_items(leg_pick_markers)
                    for leg_index, marker_data in enumerate(indexed_leg_markers, start=1):
                        marker_mesh = marker_data["mesh"]
                        marker_actor = Mesh([marker_mesh.vertices.tolist(), marker_mesh.faces.tolist()]).c("#da4f2f").alpha(1.0)
                        marker_actor.info = f"Leg Index {leg_index}\nSide: {marker_data['side_name']}"
                        self.actors.append(marker_actor)
                        self.plotter += marker_actor
                if self.show_helper_objects and current_step_index >= 8:
                    indexed_die_regions = _clockwise_indexed_items(die_regions)
                    region_palette = [
                        "#ef4444",
                        "#f97316",
                        "#eab308",
                        "#22c55e",
                        "#06b6d4",
                        "#3b82f6",
                        "#8b5cf6",
                        "#ec4899",
                    ]
                    for silicon_index, region_data in enumerate(indexed_die_regions, start=1):
                        region_mesh = region_data["mesh"]
                        region_color = region_palette[(silicon_index - 1) % len(region_palette)]
                        region_actor = Mesh([region_mesh.vertices.tolist(), region_mesh.faces.tolist()]).c(region_color).alpha(1.0)
                        region_actor.info = (
                            f"Silicon Index {silicon_index}\n"
                            f"Side: {region_data['side_name']}\n"
                            f"Section: {region_data['section_index']}"
                        )
                        self.actors.append(region_actor)
                        self.plotter += region_actor
                    if die_pick_marker is not None:
                        pick_actor = Mesh([die_pick_marker.vertices.tolist(), die_pick_marker.faces.tolist()]).c("#0f766e").alpha(1.0)
                        pick_actor.info = "Selected Die Pick Position"
                        self.actors.append(pick_actor)
                        self.plotter += pick_actor
                if self.show_helper_objects and 9 <= current_step_index < 11:
                    for arc_data in connection_paths:
                        arc_actor = Line(arc_data["points"], lw=3, c="#2563eb", alpha=1.0)
                        arc_actor.info = (
                            f"Bond Arc\n"
                            f"Leg Index {arc_data['leg_index']} -> Die Index {arc_data['die_index']}\n"
                            f"Leg Side: {arc_data['leg_side_name']}\n"
                            f"Die Side: {arc_data['die_side_name']} Section {arc_data['die_section_index']}"
                        )
                        self.actors.append(arc_actor)
                        self.plotter += arc_actor
                if current_step_index >= 10:
                    assembly_source = (
                        ball_bond_terminal_meshes
                        if current_step_index >= 12
                        else ball_bond_wire_meshes
                        if current_step_index >= 11
                        else ball_bond_meshes
                    )
                    assembly_label = (
                        "Integrated Ball Bond Terminal"
                        if current_step_index >= 12
                        else "Ball Bond + Wire"
                        if current_step_index >= 11
                        else "Ball Bond"
                    )
                    assembly_color = "#2563eb" if current_step_index >= 11 else "#f59e0b"
                    for assembly_data in assembly_source:
                        assembly_mesh = assembly_data["mesh"]
                        assembly_actor = Mesh([assembly_mesh.vertices.tolist(), assembly_mesh.faces.tolist()]).c(assembly_color).alpha(1.0)
                        assembly_actor.info = (
                            f"{assembly_label}\n"
                            f"Leg Index {assembly_data.get('leg_index', assembly_data['die_index'])} -> Die Index {assembly_data['die_index']}\n"
                            f"Leg Side: {assembly_data.get('leg_side_name', '')}\n"
                            f"Die Side: {assembly_data.get('die_side_name', assembly_data.get('side_name', ''))} Section {assembly_data.get('die_section_index', assembly_data.get('section_index', 0))}"
                        )
                        self.actors.append(assembly_actor)
                        self.plotter += assembly_actor
                die_status = (
                    f"die frame {die_info.get('final_width_mm', 0.0):.2f} x {die_info.get('final_depth_mm', 0.0):.2f} mm"
                    if current_step_index >= 5 and die_leadframe_mesh is not None
                    else "die frame waiting"
                )
                silicon_status = (
                    f"silicon die {silicon_die_info.get('final_width_mm', 0.0):.2f} x {silicon_die_info.get('final_depth_mm', 0.0):.2f} mm"
                    if current_step_index >= 6 and silicon_die_mesh is not None
                    else "silicon die waiting"
                )
                leg_pick_status = f"leg picks {len(leg_pick_markers)}" if current_step_index >= 7 else "leg picks waiting"
                die_region_status = (
                    f"die regions {die_region_info.get('span_percent', 0.0):.0f}% span"
                    if current_step_index >= 8 and die_regions
                    else "die regions waiting"
                )
                arc_status = f"arcs {len(connection_paths)}" if current_step_index >= 9 and connection_paths else "arcs waiting"
                bond_status = f"ball bonds {len(ball_bond_meshes)}" if current_step_index >= 10 and ball_bond_meshes else "ball bonds waiting"
                tube_status = f"bond-wire assemblies {len(ball_bond_wire_meshes)}" if current_step_index >= 11 and ball_bond_wire_meshes else "bond-wire assemblies waiting"
                wedge_status = f"integrated terminals {len(ball_bond_terminal_meshes)}" if current_step_index >= 12 and ball_bond_terminal_meshes else "integrated terminals waiting"
                summary = (
                    f"Leads: {len(side_meshes)}  Body: {body_status}  Plane: {plane_status}  "
                    f"Leadframe: {die_status}  Die: {silicon_status}  Leg Picks: {leg_pick_status}  Regions: {die_region_status}  Arcs: {arc_status}  Bonds: {bond_status}  Tubes: {tube_status}  Wedges: {wedge_status}"
                )
        except Exception as exc:
            summary = f"Preview blocked: {exc}"

        self.info.text(
            "IC Chip Generator Preview\n"
            f"{summary}\n"
            f"Projection: {'Orthographic' if self.is_orthographic else 'Perspective'}\n"
            f"Helpers: {'Visible' if self.show_helper_objects else 'Hidden'}\n"
            f"{status_message or 'Draw a closed lead profile, set length, then place it on each side.'}"
        )
        self.plotter.render()

    def _set_projection_mode(self, orthographic: bool) -> None:
        self.is_orthographic = orthographic
        camera = getattr(self.plotter, "camera", None)
        if camera is not None:
            camera.SetParallelProjection(1 if orthographic else 0)
        self.plotter.render()

    def _toggle_projection_mode(self, *_args) -> None:
        self._set_projection_mode(not self.is_orthographic)

    def _helper_toggle_label(self) -> str:
        return "[x] Helpers" if self.show_helper_objects else "[ ] Helpers"

    def _toggle_helper_objects(self) -> None:
        self.show_helper_objects = not self.show_helper_objects
        if self.helper_toggle_button is not None:
            self.helper_toggle_button.text(f" {self._helper_toggle_label()} ")
            self.helper_toggle_button.info = self._helper_toggle_label()
        payload = read_bridge_payload(self.bridge_path)
        if payload:
            self._build_scene(payload)
            self.last_signature = self._payload_signature(payload)
        else:
            self.plotter.render()

    def _make_ui_button(self, label: str, pos: tuple[float, float], handler, scale: float = 0.8) -> Text2D:
        button = Text2D(
            f" {label} ",
            pos=pos,
            s=scale,
            c="#2d241f",
            bg="#f7f0e4",
            font="Courier",
            bold=True,
            alpha=1.0,
        )
        button.PickableOn()
        button.info = label
        self.plotter.add(button)
        self.ui_buttons.append(button)
        self.ui_button_handlers[id(button)] = handler
        return button

    def _on_ui_button_click(self, event) -> None:
        actor = getattr(event, "actor", None)
        handler = self.ui_button_handlers.get(id(actor))
        if handler is None:
            return
        handler()

    def _visible_scene_bounds(self) -> tuple[float, float, float, float, float, float]:
        renderer = getattr(self.plotter, "renderer", None)
        if renderer is None:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        bounds = renderer.ComputeVisiblePropBounds()
        if not bounds or len(bounds) != 6:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        if any(abs(value) > 1e20 for value in bounds):
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        return tuple(float(value) for value in bounds)

    def _set_axis_view(self, axis_name: str) -> None:
        bounds = self._visible_scene_bounds()
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z
        distance = max(span_x, span_y, span_z, 2.0) * 2.2

        position_map = {
            "+X": ((center_x + distance, center_y, center_z), (0.0, 0.0, 1.0)),
            "-X": ((center_x - distance, center_y, center_z), (0.0, 0.0, 1.0)),
            "+Y": ((center_x, center_y + distance, center_z), (0.0, 0.0, 1.0)),
            "-Y": ((center_x, center_y - distance, center_z), (0.0, 0.0, 1.0)),
            "+Z": ((center_x, center_y, center_z + distance), (0.0, 1.0, 0.0)),
            "-Z": ((center_x, center_y, center_z - distance), (0.0, 1.0, 0.0)),
        }
        position, view_up = position_map.get(axis_name, ((center_x + distance, center_y, center_z), (0.0, 0.0, 1.0)))

        camera = getattr(self.plotter, "camera", None)
        if camera is not None:
            camera.SetFocalPoint(center_x, center_y, center_z)
            camera.SetPosition(*position)
            camera.SetViewUp(*view_up)
            self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _on_timer(self, _event) -> None:
        payload = read_bridge_payload(self.bridge_path)
        if not payload:
            return
        signature = self._payload_signature(payload)
        if signature != self.last_signature:
            self._build_scene(payload)
            self.last_signature = signature

    def _wait_for_initial_payload(self, timeout_seconds: float = 3.0, poll_interval_seconds: float = 0.05) -> dict:
        deadline = time.monotonic() + timeout_seconds
        payload: dict = {}
        while time.monotonic() < deadline:
            payload = read_bridge_payload(self.bridge_path)
            if payload:
                return payload
            time.sleep(poll_interval_seconds)
        return payload

    def run(self) -> None:
        payload = self._wait_for_initial_payload()
        if not payload:
            raise RuntimeError(f"Viewer bridge payload missing: {self.bridge_path}")
        self.plotter.show(self.info, zoom="tight", interactive=False)
        if not self.ui_buttons:
            self.helper_toggle_button = self._make_ui_button(self._helper_toggle_label(), (0.60, 0.93), self._toggle_helper_objects, scale=0.85)
            self._make_ui_button("Toggle Ortho", (0.78, 0.93), self._toggle_projection_mode, scale=0.9)
            view_specs = [
                ("+X", (0.70, 0.93)),
                ("-X", (0.70, 0.89)),
                ("+Y", (0.78, 0.89)),
                ("-Y", (0.86, 0.89)),
                ("+Z", (0.78, 0.85)),
                ("-Z", (0.86, 0.85)),
            ]
            for label, position in view_specs:
                self._make_ui_button(label, position, lambda axis_name=label: self._set_axis_view(axis_name), scale=0.8)
        if self.ui_click_callback_id is None:
            self.ui_click_callback_id = self.plotter.add_callback("LeftButtonPress", self._on_ui_button_click)
        if self.hover_legend_callback_id is None:
            self.hover_legend_callback_id = self.plotter.add_hover_legend(
                c="#2d241f",
                pos="bottom-right",
                bg="#f7f0e4",
                alpha=0.9,
                use_info=True,
            )
        self._build_scene(payload)
        self.last_signature = self._payload_signature(payload)
        self.plotter.add_callback("Timer", self._on_timer)
        self.plotter.timer_callback("create", dt=150)
        self.plotter.interactive()


class IcChipGeneratorApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("IC Chip Generator")
        self.root.geometry("1480x900")
        self.root.minsize(1240, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.scale_px_per_mm = DEFAULT_SCALE_PX_PER_MM
        self.view_offset_px = (0.0, 0.0)
        self.is_panning_canvas = False
        self.last_pan_anchor_px: tuple[float, float] | None = None

        self.profile = LeadProfile(points_mm=[])
        self.current_points_px: list[tuple[float, float]] = []
        self.preview_cursor_px: tuple[float, float] | None = None
        self.dragging_vertex_index: int | None = None
        self.canvas_vertex_items: dict[int, int] = {}
        self.snap_hover_axis: str | None = None
        self.snap_hover_after_id: str | None = None
        self.snap_locked_axis: str | None = None
        self.snap_anchor_px: tuple[float, float] | None = None
        self.snapped_preview_cursor_px: tuple[float, float] | None = None
        self.drag_snap_x_mm: float | None = None
        self.drag_snap_y_mm: float | None = None
        self.distance_pick_active = False
        self.distance_point_indices: list[int] = []

        self.step_index = 0
        self.step_titles = [
            "1. Draw Lead Profile",
            "2. Extrude Lead Length",
            "3. Set Legs Per Side",
            "4. Overall 3D Placement",
            "5. Lead Offset",
            "6. Die Leadframe",
            "7. Silicon Die",
            "8. Leg Positions",
            "9. Die Regions",
            "10. Bond Arcs",
            "11. Ball Bond Formation",
            "12. Bond Wire Tube",
            "13. Wedge Bond Ending",
        ]

        self.leg_length_var = tk.DoubleVar(value=4.0)
        self.body_width_var = tk.DoubleVar(value=10.0)
        self.body_depth_var = tk.DoubleVar(value=10.0)
        self.body_height_var = tk.DoubleVar(value=1.6)
        self.top_count_var = tk.IntVar(value=4)
        self.bottom_count_var = tk.IntVar(value=4)
        self.left_count_var = tk.IntVar(value=4)
        self.right_count_var = tk.IntVar(value=4)
        self.top_pitch_var = tk.DoubleVar(value=1.27)
        self.bottom_pitch_var = tk.DoubleVar(value=1.27)
        self.left_pitch_var = tk.DoubleVar(value=1.27)
        self.right_pitch_var = tk.DoubleVar(value=1.27)
        self.top_pitch_axis_var = tk.StringVar(value="x")
        self.bottom_pitch_axis_var = tk.StringVar(value="x")
        self.left_pitch_axis_var = tk.StringVar(value="y")
        self.right_pitch_axis_var = tk.StringVar(value="y")
        self.top_rx_var = tk.DoubleVar(value=0.0)
        self.top_ry_var = tk.DoubleVar(value=0.0)
        self.top_rz_var = tk.DoubleVar(value=0.0)
        self.bottom_rx_var = tk.DoubleVar(value=0.0)
        self.bottom_ry_var = tk.DoubleVar(value=0.0)
        self.bottom_rz_var = tk.DoubleVar(value=0.0)
        self.left_rx_var = tk.DoubleVar(value=0.0)
        self.left_ry_var = tk.DoubleVar(value=0.0)
        self.left_rz_var = tk.DoubleVar(value=0.0)
        self.right_rx_var = tk.DoubleVar(value=0.0)
        self.right_ry_var = tk.DoubleVar(value=0.0)
        self.right_rz_var = tk.DoubleVar(value=0.0)
        self.distance_target_var = tk.DoubleVar(value=1.0)
        self.lead_offset_var = tk.DoubleVar(value=0.0)
        self.die_leadframe_ratio_var = tk.DoubleVar(value=80.0)
        self.die_leadframe_clearance_var = tk.DoubleVar(value=0.05)
        self.die_leadframe_thickness_var = tk.DoubleVar(value=0.08)
        self.silicon_die_width_var = tk.DoubleVar(value=2.5)
        self.silicon_die_depth_var = tk.DoubleVar(value=2.5)
        self.silicon_die_thickness_var = tk.DoubleVar(value=0.12)
        self.leg_pick_distance_var = tk.DoubleVar(value=0.2)
        self.leg_pick_marker_size_var = tk.DoubleVar(value=0.08)
        self.die_region_span_percent_var = tk.DoubleVar(value=70.0)
        self.die_region_depth_var = tk.DoubleVar(value=0.15)
        self.die_region_offset_var = tk.DoubleVar(value=0.05)
        self.die_pick_region_var = tk.StringVar(value="Top")
        self.die_pick_section_index_var = tk.IntVar(value=1)
        self.die_pick_position_percent_var = tk.DoubleVar(value=50.0)
        self.die_pick_marker_size_var = tk.DoubleVar(value=0.06)
        self.arc_height_var = tk.DoubleVar(value=0.5)
        self.arc_xy_noise_var = tk.DoubleVar(value=0.0)
        self.wire_arc_point_spacing_var = tk.DoubleVar(value=0.08)
        self.ball_bond_diameter_var = tk.DoubleVar(value=0.12)
        self.ball_bond_length_var = tk.DoubleVar(value=0.08)
        self.ball_bond_revolution_steps_var = tk.IntVar(value=24)
        self.wire_diameter_var = tk.DoubleVar(value=0.03)
        self.wire_rise_z_var = tk.DoubleVar(value=0.12)
        self.wire_tube_side_count_var = tk.IntVar(value=10)
        self.wedge_bond_length_var = tk.DoubleVar(value=0.18)
        self.wedge_bond_width_var = tk.DoubleVar(value=0.08)
        self.wedge_bond_thickness_var = tk.DoubleVar(value=0.02)
        self.wedge_approach_run_var = tk.DoubleVar(value=0.18)
        self.project_name_var = tk.StringVar(value="untitled_chip")
        self.status_var = tk.StringVar(value="Step 1: draw a closed 2D lead profile.")

        self.bridge_path = BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
        self.viewer_process: subprocess.Popen | None = None
        self.preview_refresh_after_id: str | None = None
        self.suspend_preview_refresh = False
        self.current_project_dir = self._project_dir_for_name(self.project_name_var.get())

        self._build_ui()
        self._bind_live_preview_vars()
        self._redraw_canvas()
        self._push_payload()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        left = ttk.Frame(root_frame, width=380)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        right = ttk.Frame(root_frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(left, text="IC Chip Generator", font=("Georgia", 18, "bold")).pack(anchor="w")
        ttk.Label(
            left,
            text="Work through the lead profile, extrusion, side placement, and rotation assist one step at a time.",
            wraplength=330,
        ).pack(anchor="w", pady=(6, 10))

        self.step_label = ttk.Label(left, text=self.step_titles[self.step_index], font=("Georgia", 13, "bold"))
        self.step_label.pack(anchor="w", pady=(0, 8))

        nav_row = ttk.Frame(left)
        nav_row.pack(fill="x", pady=(0, 12))
        ttk.Button(nav_row, text="Previous", command=self._previous_step).pack(side="left", fill="x", expand=True)
        ttk.Button(nav_row, text="Next", command=self._next_step).pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.step_frames: list[ttk.LabelFrame] = []
        self.step_frames.append(self._build_draw_step(left))
        self.step_frames.append(self._build_length_step(left))
        self.step_frames.append(self._build_side_count_step(left))
        self.step_frames.append(self._build_overall_step(left))
        self.step_frames.append(self._build_lead_offset_step(left))
        self.step_frames.append(self._build_die_leadframe_step(left))
        self.step_frames.append(self._build_silicon_die_step(left))
        self.step_frames.append(self._build_leg_positions_step(left))
        self.step_frames.append(self._build_die_regions_step(left))
        self.step_frames.append(self._build_bond_arcs_step(left))
        self.step_frames.append(self._build_ball_bond_step(left))
        self.step_frames.append(self._build_bond_wire_tube_step(left))
        self.step_frames.append(self._build_wedge_bond_step(left))

        self.canvas_title_label = ttk.Label(right, text="Lead Profile Sketch", font=("Georgia", 15, "bold"))
        self.canvas_title_label.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.canvas = tk.Canvas(
            right,
            width=DEFAULT_CANVAS_WIDTH,
            height=DEFAULT_CANVAS_HEIGHT,
            bg=CANVAS_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#bca88d",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_release)
        self.canvas.bind("<Button-2>", self._on_canvas_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_canvas_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self._on_canvas_pan_end)
        self.canvas.bind("<Button-3>", self._on_canvas_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_canvas_pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_pan_end)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self._on_canvas_zoom)
        self.canvas.bind("<Button-4>", self._on_canvas_zoom)
        self.canvas.bind("<Button-5>", self._on_canvas_zoom)
        self._show_step()

    def _bind_live_preview_vars(self) -> None:
        preview_vars = [
            self.leg_length_var,
            self.top_count_var,
            self.bottom_count_var,
            self.left_count_var,
            self.right_count_var,
            self.top_pitch_var,
            self.bottom_pitch_var,
            self.left_pitch_var,
            self.right_pitch_var,
            self.top_pitch_axis_var,
            self.bottom_pitch_axis_var,
            self.left_pitch_axis_var,
            self.right_pitch_axis_var,
            self.top_rx_var,
            self.top_ry_var,
            self.top_rz_var,
            self.bottom_rx_var,
            self.bottom_ry_var,
            self.bottom_rz_var,
            self.left_rx_var,
            self.left_ry_var,
            self.left_rz_var,
            self.right_rx_var,
            self.right_ry_var,
            self.right_rz_var,
            self.body_width_var,
            self.body_depth_var,
            self.body_height_var,
            self.lead_offset_var,
            self.die_leadframe_ratio_var,
            self.die_leadframe_clearance_var,
            self.die_leadframe_thickness_var,
            self.silicon_die_width_var,
            self.silicon_die_depth_var,
            self.silicon_die_thickness_var,
            self.leg_pick_distance_var,
            self.leg_pick_marker_size_var,
            self.die_region_span_percent_var,
            self.die_region_depth_var,
            self.die_region_offset_var,
            self.die_pick_region_var,
            self.die_pick_section_index_var,
            self.die_pick_position_percent_var,
            self.die_pick_marker_size_var,
            self.arc_height_var,
            self.arc_xy_noise_var,
            self.wire_arc_point_spacing_var,
            self.ball_bond_diameter_var,
            self.ball_bond_length_var,
            self.ball_bond_revolution_steps_var,
            self.wire_diameter_var,
            self.wire_rise_z_var,
            self.wire_tube_side_count_var,
            self.wedge_bond_length_var,
            self.wedge_bond_width_var,
            self.wedge_bond_thickness_var,
            self.wedge_approach_run_var,
        ]
        for variable in preview_vars:
            variable.trace_add("write", self._schedule_preview_refresh)

    def _schedule_preview_refresh(self, *_args) -> None:
        if self.preview_refresh_after_id is not None:
            self.root.after_cancel(self.preview_refresh_after_id)
        self.preview_refresh_after_id = self.root.after(180, self._refresh_preview_from_vars)

    def _refresh_preview_from_vars(self) -> None:
        self.preview_refresh_after_id = None
        if self.suspend_preview_refresh:
            return
        if self.step_index < 1:
            self._push_payload()
            return
        self._push_payload(launch_if_missing=True)

    def _project_dir_for_name(self, project_name: str) -> Path:
        slug = _safe_stage_slug(project_name)
        return PROJECTS_ROOT_DIR / slug

    def _project_snapshot_path(self, project_dir: Path | None = None) -> Path:
        target_dir = project_dir if project_dir is not None else self.current_project_dir
        return target_dir / "ic_chip_generator_project.json"

    def _project_stage_dir(self, project_dir: Path | None = None) -> Path:
        target_dir = project_dir if project_dir is not None else self.current_project_dir
        return target_dir / "stages"

    def _ensure_current_project_dir(self) -> None:
        self.current_project_dir.mkdir(parents=True, exist_ok=True)
        self._project_stage_dir().mkdir(parents=True, exist_ok=True)

    def _create_or_switch_project_folder(self) -> None:
        project_name = self.project_name_var.get().strip() or "untitled_chip"
        self.project_name_var.set(project_name)
        self.current_project_dir = self._project_dir_for_name(project_name)
        self._push_payload()
        self.status_var.set(f"Project folder active: {self.current_project_dir.name}")

    def _prompt_load_project_folder(self) -> None:
        project_dir = filedialog.askdirectory(
            parent=self.root,
            title="Load IC chip project folder",
            initialdir=str(PROJECTS_ROOT_DIR if PROJECTS_ROOT_DIR.exists() else DEFAULT_OUTPUT_DIR),
            mustexist=True,
        )
        if not project_dir:
            return
        try:
            self._load_project_folder(Path(project_dir))
        except Exception as exc:
            messagebox.showerror("Load Project Failed", str(exc), parent=self.root)
            self.status_var.set(f"Load project failed: {exc}")

    def _load_project_folder(self, project_dir: Path) -> None:
        snapshot_path = self._project_snapshot_path(project_dir)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Project snapshot not found: {snapshot_path}")
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Project snapshot is not a JSON object.")

        self.suspend_preview_refresh = True
        try:
            project_name = str(payload.get("project_name", project_dir.name)).strip() or project_dir.name
            self.project_name_var.set(project_name)
            self.current_project_dir = project_dir
            self.step_index = max(0, min(len(self.step_titles) - 1, _coerce_int(payload.get("current_step_index", 0), 0)))
            self.profile = LeadProfile(
                points_mm=[
                    tuple(point)
                    for point in payload.get("profile_points_mm", [])
                    if isinstance(point, list | tuple) and len(point) == 2
                ]
            )
            self.current_points_px = [
                tuple(point)
                for point in payload.get("draft_points_px", [])
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            self.distance_target_var.set(_coerce_float(payload.get("distance_target_mm", 1.0), 1.0))
            self.lead_offset_var.set(_coerce_float(payload.get("lead_offset_mm", 0.0), 0.0))
            self.die_leadframe_ratio_var.set(_coerce_float(payload.get("die_leadframe_ratio_percent", 80.0), 80.0))
            self.die_leadframe_clearance_var.set(_coerce_float(payload.get("die_leadframe_clearance_mm", 0.05), 0.05))
            self.die_leadframe_thickness_var.set(_coerce_float(payload.get("die_leadframe_thickness_mm", 0.08), 0.08))
            self.silicon_die_width_var.set(_coerce_float(payload.get("silicon_die_width_mm", 2.5), 2.5))
            self.silicon_die_depth_var.set(_coerce_float(payload.get("silicon_die_depth_mm", 2.5), 2.5))
            self.silicon_die_thickness_var.set(_coerce_float(payload.get("silicon_die_thickness_mm", 0.12), 0.12))
            self.leg_pick_distance_var.set(_coerce_float(payload.get("leg_pick_distance_mm", 0.2), 0.2))
            self.leg_pick_marker_size_var.set(_coerce_float(payload.get("leg_pick_marker_size_mm", 0.08), 0.08))
            self.die_region_span_percent_var.set(_coerce_float(payload.get("die_region_span_percent", 70.0), 70.0))
            self.die_region_depth_var.set(_coerce_float(payload.get("die_region_depth_mm", 0.15), 0.15))
            self.die_region_offset_var.set(_coerce_float(payload.get("die_region_offset_mm", 0.05), 0.05))
            self.die_pick_region_var.set(str(payload.get("die_pick_region", "Top")).strip().title() or "Top")
            self.die_pick_section_index_var.set(_coerce_int(payload.get("die_pick_section_index", 1), 1))
            self.die_pick_position_percent_var.set(_coerce_float(payload.get("die_pick_position_percent", 50.0), 50.0))
            self.die_pick_marker_size_var.set(_coerce_float(payload.get("die_pick_marker_size_mm", 0.06), 0.06))
            self.arc_height_var.set(_coerce_float(payload.get("arc_height_mm", 0.5), 0.5))
            self.arc_xy_noise_var.set(_coerce_float(payload.get("arc_xy_noise_mm", 0.0), 0.0))
            self.wire_arc_point_spacing_var.set(_coerce_float(payload.get("wire_arc_point_spacing_mm", 0.08), 0.08))
            self.ball_bond_diameter_var.set(_coerce_float(payload.get("ball_bond_diameter_mm", 0.12), 0.12))
            self.ball_bond_length_var.set(_coerce_float(payload.get("ball_bond_length_mm", 0.08), 0.08))
            self.ball_bond_revolution_steps_var.set(_coerce_int(payload.get("ball_bond_revolution_steps", 24), 24))
            self.wire_diameter_var.set(_coerce_float(payload.get("wire_diameter_mm", 0.03), 0.03))
            self.wire_rise_z_var.set(_coerce_float(payload.get("wire_rise_z_mm", 0.12), 0.12))
            self.wire_tube_side_count_var.set(_coerce_int(payload.get("wire_tube_side_count", 10), 10))
            self.wedge_bond_length_var.set(_coerce_float(payload.get("wedge_bond_length_mm", 0.18), 0.18))
            self.wedge_bond_width_var.set(_coerce_float(payload.get("wedge_bond_width_mm", 0.08), 0.08))
            self.wedge_bond_thickness_var.set(_coerce_float(payload.get("wedge_bond_thickness_mm", 0.02), 0.02))
            self.wedge_approach_run_var.set(_coerce_float(payload.get("wedge_approach_run_mm", 0.18), 0.18))
            self.distance_pick_active = bool(payload.get("distance_pick_active", False))
            self.distance_point_indices = [
                _coerce_int(index, -1)
                for index in payload.get("distance_point_indices", [])
                if 0 <= _coerce_int(index, -1) < len(self.profile.points_mm)
            ]

            self.leg_length_var.set(_coerce_float(payload.get("leg_length_mm", 4.0), 4.0))
            self.body_width_var.set(_coerce_float(payload.get("body_width_mm", 10.0), 10.0))
            self.body_depth_var.set(_coerce_float(payload.get("body_depth_mm", 10.0), 10.0))
            self.body_height_var.set(_coerce_float(payload.get("body_height_mm", 1.6), 1.6))

            side_payloads = payload.get("side_settings", [])
            side_map = {
                str(item.get("name", "")).strip(): item
                for item in side_payloads
                if isinstance(item, dict)
            }
            for side_name, count_var, pitch_var, pitch_axis_var, rx_var, ry_var, rz_var in [
                ("Top", self.top_count_var, self.top_pitch_var, self.top_pitch_axis_var, self.top_rx_var, self.top_ry_var, self.top_rz_var),
                ("Bottom", self.bottom_count_var, self.bottom_pitch_var, self.bottom_pitch_axis_var, self.bottom_rx_var, self.bottom_ry_var, self.bottom_rz_var),
                ("Left", self.left_count_var, self.left_pitch_var, self.left_pitch_axis_var, self.left_rx_var, self.left_ry_var, self.left_rz_var),
                ("Right", self.right_count_var, self.right_pitch_var, self.right_pitch_axis_var, self.right_rx_var, self.right_ry_var, self.right_rz_var),
            ]:
                side_payload = side_map.get(side_name, {})
                count_var.set(_coerce_int(side_payload.get("count", count_var.get()), count_var.get()))
                pitch_var.set(_coerce_float(side_payload.get("pitch_mm", pitch_var.get()), pitch_var.get()))
                pitch_axis = str(side_payload.get("pitch_axis", pitch_axis_var.get())).strip().lower()
                pitch_axis_var.set(pitch_axis if pitch_axis in {"x", "y", "z"} else "x")
                rx_var.set(_coerce_float(side_payload.get("rotation_x_deg", rx_var.get()), rx_var.get()))
                ry_var.set(_coerce_float(side_payload.get("rotation_y_deg", ry_var.get()), ry_var.get()))
                rz_var.set(_coerce_float(side_payload.get("rotation_z_deg", rz_var.get()), rz_var.get()))

            self.status_var.set(str(payload.get("status_message", f"Loaded project {project_dir.name}.")))
        finally:
            self.suspend_preview_refresh = False

        self.dragging_vertex_index = None
        self.drag_snap_x_mm = None
        self.drag_snap_y_mm = None
        self.preview_cursor_px = None
        self._show_step()
        self._redraw_canvas()
        self._push_payload(launch_if_missing=self.step_index >= 1)

    def _build_draw_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 1: Draw Contact Lead", padding=10)
        project_box = ttk.LabelFrame(frame, text="Project", padding=8)
        project_box.pack(fill="x", pady=(0, 10))
        ttk.Label(project_box, text="Project Name").pack(anchor="w")
        ttk.Entry(project_box, textvariable=self.project_name_var).pack(fill="x", pady=(2, 8))
        ttk.Button(project_box, text="Create / Switch Project Folder", command=self._create_or_switch_project_folder).pack(fill="x")
        ttk.Button(project_box, text="Load Project Folder", command=self._prompt_load_project_folder).pack(fill="x", pady=(8, 0))

        ttk.Label(
            frame,
            text="Left click adds points. Click Finish Closed Shape when the outline is complete. Drag existing handles to adjust the profile.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Button(frame, text="Undo Point", command=self._undo_point).pack(fill="x", pady=(10, 0))
        ttk.Button(frame, text="Finish Closed Shape", command=self._finish_closed_shape).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="Clear Lead Profile", command=self._clear_profile).pack(fill="x", pady=(8, 0))
        distance_box = ttk.LabelFrame(frame, text="Distance Between Points", padding=8)
        distance_box.pack(fill="x", pady=(10, 0))
        ttk.Label(
            distance_box,
            text="Pick 2 saved points, enter the target distance in mm, then apply to move the second point along the current direction.",
            wraplength=310,
        ).pack(anchor="w")
        ttk.Button(distance_box, text="Pick Distance Points", command=self._toggle_distance_point_pick).pack(fill="x", pady=(8, 0))
        ttk.Label(distance_box, text="Target Distance (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(distance_box, textvariable=self.distance_target_var).pack(fill="x", pady=(2, 0))
        ttk.Button(distance_box, text="Apply Distance", command=self._apply_distance_between_points).pack(fill="x", pady=(8, 0))
        ttk.Button(distance_box, text="Clear Distance Picks", command=self._clear_distance_point_picks).pack(fill="x", pady=(8, 0))
        return frame

    def _build_length_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 2: Extrude Each Lead", padding=10)
        ttk.Label(frame, text="Lead Length (mm)").pack(anchor="w")
        ttk.Entry(frame, textvariable=self.leg_length_var).pack(fill="x", pady=(2, 8))
        ttk.Label(
            frame,
            text="This becomes the extrusion length for every lead created from the profile.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Button(frame, text="Refresh 3D Preview", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_side_count_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 3: Legs Per Side", padding=10)
        ttk.Label(
            frame,
            text="Set count, pitch, and pitch axis for Top, Bottom, Left, and Right. The placement rectangle belongs to the chip XY plane, while your lead sketch stays on its own profile plane.",
            wraplength=330,
        ).pack(anchor="w")
        rectangle_box = ttk.LabelFrame(frame, text="Placement Rectangle", padding=8)
        rectangle_box.pack(fill="x", pady=(10, 0))
        ttk.Label(rectangle_box, text="Rectangle Width / Depth (mm)").pack(anchor="w")
        rectangle_row = ttk.Frame(rectangle_box)
        rectangle_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(rectangle_row, textvariable=self.body_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(rectangle_row, textvariable=self.body_depth_var, width=8).pack(side="left", fill="x", expand=True, padx=6)

        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=(10, 0))
        headers = ("Side", "Count", "Pitch (mm)", "Axis")
        for column, text in enumerate(headers):
            ttk.Label(grid, text=text).grid(row=0, column=column, sticky="w", padx=(0, 8))
        rows = [
            ("Top", self.top_count_var, self.top_pitch_var, self.top_pitch_axis_var),
            ("Bottom", self.bottom_count_var, self.bottom_pitch_var, self.bottom_pitch_axis_var),
            ("Left", self.left_count_var, self.left_pitch_var, self.left_pitch_axis_var),
            ("Right", self.right_count_var, self.right_pitch_var, self.right_pitch_axis_var),
        ]
        for row_index, (name, count_var, pitch_var, axis_var) in enumerate(rows, start=1):
            ttk.Label(grid, text=name).grid(row=row_index, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(grid, textvariable=count_var, width=10).grid(row=row_index, column=1, sticky="ew", padx=(0, 8), pady=(6, 0))
            ttk.Entry(grid, textvariable=pitch_var, width=10).grid(row=row_index, column=2, sticky="ew", pady=(6, 0))
            ttk.Combobox(
                grid,
                textvariable=axis_var,
                values=("x", "y", "z"),
                state="readonly",
                width=10,
            ).grid(row=row_index, column=3, sticky="ew", padx=(8, 0), pady=(6, 0))

        rotation_box = ttk.LabelFrame(frame, text="Leg Rotation In This Stage", padding=8)
        rotation_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            rotation_box,
            text="Adjust per-side X / Y / Z leg rotation here while you set spacing and pitch.",
            wraplength=310,
        ).pack(anchor="w")
        self._build_rotation_rows(rotation_box)
        return frame

    def _build_rotation_rows(self, parent: ttk.Frame) -> None:
        rows = [
            ("Top", self.top_rx_var, self.top_ry_var, self.top_rz_var),
            ("Bottom", self.bottom_rx_var, self.bottom_ry_var, self.bottom_rz_var),
            ("Left", self.left_rx_var, self.left_ry_var, self.left_rz_var),
            ("Right", self.right_rx_var, self.right_ry_var, self.right_rz_var),
        ]
        for name, rx_var, ry_var, rz_var in rows:
            box = ttk.LabelFrame(parent, text=name, padding=8)
            box.pack(fill="x", pady=(8, 0))
            row = ttk.Frame(box)
            row.pack(fill="x")
            ttk.Entry(row, textvariable=rx_var, width=8).pack(side="left", fill="x", expand=True)
            ttk.Entry(row, textvariable=ry_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
            ttk.Entry(row, textvariable=rz_var, width=8).pack(side="left", fill="x", expand=True)

    def _build_overall_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 4: Overall 3D Placement", padding=10)
        ttk.Label(frame, text="Chip Body Width / Depth / Height (mm)").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self.body_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(row, textvariable=self.body_depth_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Entry(row, textvariable=self.body_height_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Button(frame, text="Launch Live 3D Preview", command=self._start_viewer).pack(fill="x", pady=(12, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(8, 0))
        ttk.Label(frame, textvariable=self.status_var, wraplength=330).pack(anchor="w", pady=(10, 0))
        return frame

    def _build_lead_offset_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 5: Lead Offset", padding=10)
        ttk.Label(
            frame,
            text="Move the placed leads toward or away from the placement rectangle. Positive values push them outward; negative values pull them inward.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Lead Offset Distance (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.lead_offset_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_die_leadframe_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 6: Die Leadframe", padding=10)
        ttk.Label(
            frame,
            text="Create a centered die leadframe rectangle on top of the black body. It starts at 80% of the body rectangle, then automatically shrinks if needed to avoid touching any legs.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Leadframe Size (%)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_leadframe_ratio_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Distance Away From Leg (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_leadframe_clearance_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Frame Thickness (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_leadframe_thickness_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_silicon_die_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 7: Silicon Die", padding=10)
        ttk.Label(
            frame,
            text="Place a centered silicon die on top of the leadframe. You control its width, depth, and thickness; the tool keeps it centered for you.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Die Width / Depth / Thickness (mm)").pack(anchor="w", pady=(10, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self.silicon_die_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(row, textvariable=self.silicon_die_depth_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Entry(row, textvariable=self.silicon_die_thickness_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_leg_positions_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 8: Leg Positions", padding=10)
        ttk.Label(
            frame,
            text="Choose the 3D position on every leg from the inner touching end. The preview shows a flat square pick pad on each leg instead of a 3D block.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Distance From Inner Leg End (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.leg_pick_distance_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Marker Size (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.leg_pick_marker_size_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_die_regions_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 9: Die Regions", padding=10)
        ttk.Label(
            frame,
            text="Build 2D edge regions on the silicon die. Each visible side is divided by that side's leg count, hidden when the side has no legs, and offset inward from the die edge.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Region Span Percent").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_span_percent_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Region Band Depth (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_depth_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Region Offset From Die Edge (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_offset_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Pick Region").pack(anchor="w", pady=(10, 0))
        ttk.Combobox(frame, textvariable=self.die_pick_region_var, values=("Top", "Bottom", "Left", "Right"), state="readonly").pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Pick Section Index").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_pick_section_index_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Pick Position Along Region (%)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_pick_position_percent_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Pick Marker Size (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_pick_marker_size_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_bond_arcs_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 10: Bond Arcs", padding=10)
        ttk.Label(
            frame,
            text="Create indexed connections from the leg pick pads to the die regions. For now the tool pairs them automatically as leg index 1 to die index 1, leg index 2 to die index 2, and so on.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Arc Height Above Anchors (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.arc_height_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Curve Noise In XY (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.arc_xy_noise_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Distance Between Arc Points (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wire_arc_point_spacing_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_ball_bond_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 11: Ball Bond Formation", padding=10)
        ttk.Label(
            frame,
            text="Build a revolved ball-bond solid from a rectangle with a semicircle attached to one end. The sharp rectangle side becomes the revolve axis, producing the 3D bond form on each indexed die anchor.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Rectangle Height / Semicircle Diameter (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.ball_bond_diameter_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Rectangle Length (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.ball_bond_length_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Revolution Degree Step Count").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.ball_bond_revolution_steps_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_bond_wire_tube_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 12: Bond Wire Tube", padding=10)
        ttk.Label(
            frame,
            text="Wrap a circular tube around the bond path. The tube begins at the ball bond, rises upward first in Z, then follows the indexed curve down to each leg.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Wire Diameter (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wire_diameter_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Initial Rise In +Z (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wire_rise_z_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Distance Between Arc Points (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wire_arc_point_spacing_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Circle Polygon Side Count").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wire_tube_side_count_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_wedge_bond_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 13: Wedge Bond Ending", padding=10)
        ttk.Label(
            frame,
            text="Create a flattened oval wedge bond at the leg end. The wire is brought down earlier so the final approach runs almost parallel to the XY plane before it transitions into the pressed wedge shape.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Wedge Length (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wedge_bond_length_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Wedge Width (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wedge_bond_width_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Wedge Thickness (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wedge_bond_thickness_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Shallow Approach Run (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.wedge_approach_run_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        ttk.Button(frame, text="Export Combined STL", command=self._export_combined_stl).pack(fill="x", pady=(8, 0))
        return frame

    def _show_step(self) -> None:
        self.step_label.configure(text=self.step_titles[self.step_index])
        for index, frame in enumerate(self.step_frames):
            if index == self.step_index:
                frame.pack(fill="x", pady=(0, 12))
            else:
                frame.pack_forget()
        if self.step_index == 0:
            self.canvas_title_label.grid()
            self.canvas.grid()
        else:
            self.canvas_title_label.grid_remove()
            self.canvas.grid_remove()

    def _previous_step(self) -> None:
        self.step_index = max(0, self.step_index - 1)
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.")

    def _next_step(self) -> None:
        self.step_index = min(len(self.step_titles) - 1, self.step_index + 1)
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.")
        if self.step_index >= 1:
            self._push_payload(launch_if_missing=True)

    def _world_to_canvas(self, point_mm: tuple[float, float]) -> tuple[float, float]:
        x_mm, y_mm = point_mm
        offset_x_px, offset_y_px = self.view_offset_px
        return (
            (DEFAULT_CANVAS_WIDTH / 2.0) + offset_x_px + (x_mm * self.scale_px_per_mm),
            (DEFAULT_CANVAS_HEIGHT / 2.0) + offset_y_px - (y_mm * self.scale_px_per_mm),
        )

    def _canvas_to_world(self, point_px: tuple[float, float]) -> tuple[float, float]:
        x_px, y_px = point_px
        offset_x_px, offset_y_px = self.view_offset_px
        return (
            (x_px - (DEFAULT_CANVAS_WIDTH / 2.0) - offset_x_px) / self.scale_px_per_mm,
            ((DEFAULT_CANVAS_HEIGHT / 2.0) + offset_y_px - y_px) / self.scale_px_per_mm,
        )

    def _draw_grid(self) -> None:
        spacing_px = self.scale_px_per_mm
        step_px = max(8, int(spacing_px))
        for x_coord in range(0, DEFAULT_CANVAS_WIDTH, step_px):
            self.canvas.create_line(x_coord, 0, x_coord, DEFAULT_CANVAS_HEIGHT, fill=CANVAS_GRID)
        for y_coord in range(0, DEFAULT_CANVAS_HEIGHT, step_px):
            self.canvas.create_line(0, y_coord, DEFAULT_CANVAS_WIDTH, y_coord, fill=CANVAS_GRID)
        self.canvas.create_line(DEFAULT_CANVAS_WIDTH / 2.0, 0, DEFAULT_CANVAS_WIDTH / 2.0, DEFAULT_CANVAS_HEIGHT, fill=CANVAS_AXIS, width=2)
        self.canvas.create_line(0, DEFAULT_CANVAS_HEIGHT / 2.0, DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT / 2.0, fill=CANVAS_AXIS, width=2)

    def _draw_stage_three_side_guide(self) -> None:
        if self.step_index != 2:
            return
        body_width_mm = max(0.5, float(self.body_width_var.get()))
        body_depth_mm = max(0.5, float(self.body_depth_var.get()))
        left_top = self._world_to_canvas((-body_width_mm / 2.0, body_depth_mm / 2.0))
        right_bottom = self._world_to_canvas((body_width_mm / 2.0, -body_depth_mm / 2.0))
        left_px, top_px = left_top[0], left_top[1]
        right_px, bottom_px = right_bottom[0], right_bottom[1]

        self.canvas.create_rectangle(
            left_px,
            top_px,
            right_px,
            bottom_px,
            fill="#efe4cf",
            outline="#6f5132",
            width=2,
            dash=(6, 4),
        )

        # Side emphasis bars help connect the Step 3 controls to the body edges.
        accent = "#3f6b5b"
        self.canvas.create_line(left_px, top_px, right_px, top_px, fill=accent, width=3)
        self.canvas.create_line(left_px, bottom_px, right_px, bottom_px, fill=accent, width=3)
        self.canvas.create_line(left_px, top_px, left_px, bottom_px, fill=accent, width=3)
        self.canvas.create_line(right_px, top_px, right_px, bottom_px, fill=accent, width=3)

        center_x = (left_px + right_px) / 2.0
        center_y = (top_px + bottom_px) / 2.0
        self.canvas.create_text(center_x, top_px - 16, text="TOP", fill=accent, font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(center_x, bottom_px + 16, text="BOTTOM", fill=accent, font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(left_px - 22, center_y, text="LEFT", fill=accent, font=("Segoe UI", 10, "bold"), angle=90)
        self.canvas.create_text(right_px + 22, center_y, text="RIGHT", fill=accent, font=("Segoe UI", 10, "bold"), angle=270)

        self.canvas.create_text(
            center_x,
            center_y,
            text=f"Placement Rectangle\n{body_width_mm:.2f} x {body_depth_mm:.2f} mm",
            fill="#2d241f",
            font=("Segoe UI", 10, "bold"),
            justify="center",
        )

    def _find_vertex_hit(self, event_x: float, event_y: float) -> int | None:
        for index, point in enumerate(self.profile.points_mm):
            px, py = self._world_to_canvas(point)
            if abs(px - event_x) <= 7 and abs(py - event_y) <= 7:
                return index
        return None

    def _cancel_hover_snap_timer(self) -> None:
        if self.snap_hover_after_id is not None:
            self.root.after_cancel(self.snap_hover_after_id)
            self.snap_hover_after_id = None

    def _clear_snap_state(self, *, keep_lock: bool = False) -> None:
        self._cancel_hover_snap_timer()
        self.snap_hover_axis = None
        self.snap_anchor_px = None
        if not keep_lock:
            self.snap_locked_axis = None
            self.snapped_preview_cursor_px = None

    def _orthogonal_axis_for_cursor(self, anchor_px: tuple[float, float], cursor_px: tuple[float, float]) -> str | None:
        dx = cursor_px[0] - anchor_px[0]
        dy = cursor_px[1] - anchor_px[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        horizontal_delta = min(abs(angle_deg), abs(180.0 - angle_deg))
        vertical_delta = abs(90.0 - angle_deg)
        if horizontal_delta <= SNAP_ANGLE_THRESHOLD_DEG and horizontal_delta <= vertical_delta:
            return "horizontal"
        if vertical_delta <= SNAP_ANGLE_THRESHOLD_DEG:
            return "vertical"
        return None

    def _snapped_cursor_for_axis(self, anchor_px: tuple[float, float], cursor_px: tuple[float, float], axis_name: str) -> tuple[float, float]:
        if axis_name == "horizontal":
            return (cursor_px[0], anchor_px[1])
        return (anchor_px[0], cursor_px[1])

    def _activate_hover_snap(self, axis_name: str, anchor_px: tuple[float, float]) -> None:
        self.snap_locked_axis = axis_name
        self.snap_hover_axis = axis_name
        self.snap_anchor_px = anchor_px
        if self.preview_cursor_px is not None:
            self.snapped_preview_cursor_px = self._snapped_cursor_for_axis(anchor_px, self.preview_cursor_px, axis_name)
        self.snap_hover_after_id = None
        axis_label = "horizontal" if axis_name == "horizontal" else "vertical"
        self.status_var.set(f"90-degree snap locked on {axis_label} after hover.")
        self._redraw_canvas()

    def _snap_dragged_point_to_other_vertices(
        self,
        candidate_mm: tuple[float, float],
        dragged_index: int,
    ) -> tuple[float, float]:
        threshold_mm = POINT_AXIS_SNAP_THRESHOLD_PX / max(self.scale_px_per_mm, 1e-9)
        best_x: float | None = None
        best_y: float | None = None
        best_x_delta = float("inf")
        best_y_delta = float("inf")

        for index, point_mm in enumerate(self.profile.points_mm):
            if index == dragged_index:
                continue
            delta_x = abs(point_mm[0] - candidate_mm[0])
            delta_y = abs(point_mm[1] - candidate_mm[1])
            if delta_x <= threshold_mm and delta_x < best_x_delta:
                best_x = point_mm[0]
                best_x_delta = delta_x
            if delta_y <= threshold_mm and delta_y < best_y_delta:
                best_y = point_mm[1]
                best_y_delta = delta_y

        self.drag_snap_x_mm = best_x
        self.drag_snap_y_mm = best_y
        snapped_x = best_x if best_x is not None else candidate_mm[0]
        snapped_y = best_y if best_y is not None else candidate_mm[1]
        return (snapped_x, snapped_y)

    def _on_canvas_left_click(self, event) -> None:
        hit_index = self._find_vertex_hit(event.x, event.y)
        if self.distance_pick_active:
            if hit_index is None:
                self.status_var.set("Distance pick mode is active. Click a saved profile point.")
                return
            if hit_index in self.distance_point_indices:
                self.status_var.set(f"Point {hit_index + 1} is already selected for distance.")
                return
            if len(self.distance_point_indices) >= 2:
                self.distance_point_indices = []
            self.distance_point_indices.append(hit_index)
            if len(self.distance_point_indices) == 2:
                point_a = self.profile.points_mm[self.distance_point_indices[0]]
                point_b = self.profile.points_mm[self.distance_point_indices[1]]
                current_distance = math.dist(point_a, point_b)
                self.status_var.set(
                    f"Distance points selected: {self.distance_point_indices[0] + 1} and {self.distance_point_indices[1] + 1}. "
                    f"Current distance {current_distance:.3f} mm."
                )
            else:
                self.status_var.set(f"Picked point {hit_index + 1}. Pick one more point.")
            self._redraw_canvas()
            return
        if hit_index is not None:
            self.dragging_vertex_index = hit_index
            return
        point_px = (event.x, event.y)
        if self.snapped_preview_cursor_px is not None and self.current_points_px:
            point_px = self.snapped_preview_cursor_px
        self.current_points_px.append(point_px)
        self._clear_snap_state()
        self.status_var.set(f"Draft points: {len(self.current_points_px)}. Finish the outline when ready.")
        self._redraw_canvas()

    def _on_canvas_drag_motion(self, event) -> None:
        if self.dragging_vertex_index is None:
            return
        if 0 <= self.dragging_vertex_index < len(self.profile.points_mm):
            candidate_mm = self._canvas_to_world((event.x, event.y))
            snapped_mm = self._snap_dragged_point_to_other_vertices(candidate_mm, self.dragging_vertex_index)
            self.profile.points_mm[self.dragging_vertex_index] = snapped_mm
            self._redraw_canvas()
            self._push_payload()

    def _on_canvas_left_release(self, _event) -> None:
        self.dragging_vertex_index = None
        self.drag_snap_x_mm = None
        self.drag_snap_y_mm = None

    def _on_canvas_pan_start(self, event) -> None:
        self.is_panning_canvas = True
        self.last_pan_anchor_px = (event.x, event.y)

    def _on_canvas_pan_motion(self, event) -> None:
        if not self.is_panning_canvas or self.last_pan_anchor_px is None:
            return
        dx = event.x - self.last_pan_anchor_px[0]
        dy = event.y - self.last_pan_anchor_px[1]
        self.view_offset_px = (self.view_offset_px[0] + dx, self.view_offset_px[1] + dy)
        self.last_pan_anchor_px = (event.x, event.y)
        self._redraw_canvas()

    def _on_canvas_pan_end(self, _event) -> None:
        self.is_panning_canvas = False
        self.last_pan_anchor_px = None

    def _on_canvas_motion(self, event) -> None:
        self.preview_cursor_px = (event.x, event.y)
        if not self.current_points_px or self.dragging_vertex_index is not None or self.is_panning_canvas:
            self._clear_snap_state()
            self._redraw_canvas()
            return

        anchor_px = self.current_points_px[-1]
        axis_name = self._orthogonal_axis_for_cursor(anchor_px, self.preview_cursor_px)
        if axis_name is None:
            self._clear_snap_state()
        else:
            if self.snap_locked_axis == axis_name and self.snap_anchor_px == anchor_px:
                self.snapped_preview_cursor_px = self._snapped_cursor_for_axis(anchor_px, self.preview_cursor_px, axis_name)
            elif self.snap_locked_axis is not None and self.snap_locked_axis != axis_name:
                self._clear_snap_state()
                self.snap_hover_axis = axis_name
                self.snap_anchor_px = anchor_px
                self.snap_hover_after_id = self.root.after(
                    SNAP_HOVER_DELAY_MS,
                    lambda: self._activate_hover_snap(axis_name, anchor_px),
                )
            elif self.snap_hover_axis != axis_name or self.snap_anchor_px != anchor_px or self.snap_hover_after_id is None:
                self._cancel_hover_snap_timer()
                self.snap_hover_axis = axis_name
                self.snap_anchor_px = anchor_px
                self.snap_hover_after_id = self.root.after(
                    SNAP_HOVER_DELAY_MS,
                    lambda: self._activate_hover_snap(axis_name, anchor_px),
                )
        self._redraw_canvas()

    def _on_canvas_zoom(self, event) -> None:
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            delta = 1
        elif getattr(event, "num", None) == 5:
            delta = -1
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self.scale_px_per_mm = min(200.0, max(4.0, self.scale_px_per_mm * factor))
        self._redraw_canvas()

    def _undo_point(self) -> None:
        if self.current_points_px:
            self.current_points_px.pop()
            self._clear_snap_state()
            self.status_var.set(f"Draft points: {len(self.current_points_px)}.")
            self._redraw_canvas()
            return
        if self.profile.points_mm:
            self.profile.points_mm.pop()
            self.distance_point_indices = [index for index in self.distance_point_indices if index < len(self.profile.points_mm)]
            self.status_var.set("Removed last saved profile point.")
            self._redraw_canvas()
            self._push_payload()

    def _finish_closed_shape(self) -> None:
        if len(self.current_points_px) < 3:
            messagebox.showerror("Shape Incomplete", "Draw at least 3 points before closing the lead profile.", parent=self.root)
            return
        points_mm = [self._canvas_to_world(point_px) for point_px in self.current_points_px]
        try:
            simplified = _simplify_profile_points(points_mm)
            if len(simplified) < 3:
                raise ValueError("The shape needs at least 3 unique points.")
            if abs(_signed_area(simplified)) <= 1e-9:
                raise ValueError("The shape has zero area.")
            _triangulate_polygon(simplified)
        except Exception as exc:
            messagebox.showerror("Invalid Shape", str(exc), parent=self.root)
            return
        self.profile = LeadProfile(points_mm=simplified, closed=True)
        self.current_points_px.clear()
        self._clear_snap_state()
        self.distance_pick_active = False
        self.distance_point_indices.clear()
        self.status_var.set("Lead profile saved. Move to Step 2 to set the extrusion length.")
        self._redraw_canvas()
        self._push_payload()

    def _clear_profile(self) -> None:
        self.profile = LeadProfile(points_mm=[])
        self.current_points_px.clear()
        self._clear_snap_state()
        self.drag_snap_x_mm = None
        self.drag_snap_y_mm = None
        self.distance_pick_active = False
        self.distance_point_indices.clear()
        self.status_var.set("Lead profile cleared.")
        self._redraw_canvas()
        self._push_payload()

    def _toggle_distance_point_pick(self) -> None:
        if len(self.profile.points_mm) < 2:
            messagebox.showerror("Distance Pick Blocked", "Save a lead profile with at least 2 points first.", parent=self.root)
            return
        self.distance_pick_active = not self.distance_pick_active
        if self.distance_pick_active:
            self.distance_point_indices.clear()
            self.status_var.set("Distance pick mode ON. Click 2 saved profile points.")
        else:
            self.status_var.set("Distance pick mode OFF.")
        self._redraw_canvas()

    def _clear_distance_point_picks(self) -> None:
        self.distance_pick_active = False
        self.distance_point_indices.clear()
        self.status_var.set("Distance point picks cleared.")
        self._redraw_canvas()

    def _apply_distance_between_points(self) -> None:
        if len(self.distance_point_indices) != 2:
            messagebox.showerror("Distance Apply Blocked", "Pick exactly 2 saved profile points first.", parent=self.root)
            return
        first_index, second_index = self.distance_point_indices
        if first_index >= len(self.profile.points_mm) or second_index >= len(self.profile.points_mm):
            messagebox.showerror("Distance Apply Blocked", "Picked points are no longer valid.", parent=self.root)
            return
        point_a = self.profile.points_mm[first_index]
        point_b = self.profile.points_mm[second_index]
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        current_distance = math.hypot(dx, dy)
        if current_distance <= 1e-9:
            messagebox.showerror("Distance Apply Blocked", "The current point distance is zero, so direction is undefined.", parent=self.root)
            return
        target_distance = float(self.distance_target_var.get())
        if target_distance <= 0.0:
            messagebox.showerror("Distance Apply Blocked", "Target distance must be greater than 0 mm.", parent=self.root)
            return
        scale = target_distance / current_distance
        new_point_b = (
            point_a[0] + (dx * scale),
            point_a[1] + (dy * scale),
        )
        self.profile.points_mm[second_index] = new_point_b
        self.distance_pick_active = False
        self.status_var.set(
            f"Set distance between points {first_index + 1} and {second_index + 1} to {target_distance:.3f} mm."
        )
        self._redraw_canvas()
        self._push_payload()

    def _draw_saved_profile(self) -> None:
        if len(self.profile.points_mm) < 2:
            return
        flat_points: list[float] = []
        for point_mm in self.profile.points_mm:
            x_px, y_px = self._world_to_canvas(point_mm)
            flat_points.extend([x_px, y_px])
        self.canvas.create_polygon(
            *flat_points,
            fill="#d9c4a1",
            outline="#6f5132",
            width=2,
            stipple="gray25",
        )
        for index, point_mm in enumerate(self.profile.points_mm):
            x_px, y_px = self._world_to_canvas(point_mm)
            point_fill = "#2d241f"
            point_outline = ""
            if index in self.distance_point_indices:
                point_fill = "#3f6b5b"
                point_outline = "#dceee6"
            self.canvas.create_oval(x_px - 5, y_px - 5, x_px + 5, y_px + 5, fill=point_fill, outline=point_outline, width=2)
            self.canvas.create_text(x_px + 12, y_px - 12, text=str(index + 1), fill="#2d241f", font=("Segoe UI", 9, "bold"))

        if len(self.distance_point_indices) == 2:
            first_point = self.profile.points_mm[self.distance_point_indices[0]]
            second_point = self.profile.points_mm[self.distance_point_indices[1]]
            first_x, first_y = self._world_to_canvas(first_point)
            second_x, second_y = self._world_to_canvas(second_point)
            self.canvas.create_line(first_x, first_y, second_x, second_y, fill="#3f6b5b", dash=(6, 4), width=2)
            midpoint_x = (first_x + second_x) / 2.0
            midpoint_y = (first_y + second_y) / 2.0
            distance_mm = math.dist(first_point, second_point)
            self.canvas.create_rectangle(midpoint_x - 34, midpoint_y - 12, midpoint_x + 34, midpoint_y + 12, fill="#f7f0e4", outline="#3f6b5b")
            self.canvas.create_text(midpoint_x, midpoint_y, text=f"{distance_mm:.3f} mm", fill="#2d241f", font=("Segoe UI", 9, "bold"))

        if self.dragging_vertex_index is not None:
            if self.drag_snap_x_mm is not None:
                snap_x_px, _snap_y_px = self._world_to_canvas((self.drag_snap_x_mm, 0.0))
                self.canvas.create_line(
                    snap_x_px,
                    0,
                    snap_x_px,
                    DEFAULT_CANVAS_HEIGHT,
                    fill="#3f6b5b",
                    dash=(4, 4),
                    width=1,
                )
            if self.drag_snap_y_mm is not None:
                _snap_x_px, snap_y_px = self._world_to_canvas((0.0, self.drag_snap_y_mm))
                self.canvas.create_line(
                    0,
                    snap_y_px,
                    DEFAULT_CANVAS_WIDTH,
                    snap_y_px,
                    fill="#3f6b5b",
                    dash=(4, 4),
                    width=1,
                )

    def _draw_draft(self) -> None:
        if not self.current_points_px:
            return
        flat_points: list[float] = []
        for x_px, y_px in self.current_points_px:
            flat_points.extend([x_px, y_px])
            self.canvas.create_oval(x_px - 4, y_px - 4, x_px + 4, y_px + 4, fill="#7d4b2f", outline="")
        if len(flat_points) >= 4:
            self.canvas.create_line(*flat_points, fill="#7d4b2f", width=2)
        preview_point = self.snapped_preview_cursor_px if self.snapped_preview_cursor_px is not None else self.preview_cursor_px
        if preview_point is not None:
            last_x, last_y = self.current_points_px[-1]
            line_color = "#3f6b5b" if self.snapped_preview_cursor_px is not None else "#7d4b2f"
            self.canvas.create_line(last_x, last_y, preview_point[0], preview_point[1], fill=line_color, dash=(5, 4), width=2)
            if self.snapped_preview_cursor_px is not None:
                self.canvas.create_oval(preview_point[0] - 4, preview_point[1] - 4, preview_point[0] + 4, preview_point[1] + 4, fill=line_color, outline="")

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_stage_three_side_guide()
        self._draw_saved_profile()
        self._draw_draft()

    def _collect_side_settings(self) -> list[SideSettings]:
        return [
            SideSettings("Top", self.top_count_var.get(), self.top_pitch_var.get(), self.top_pitch_axis_var.get(), self.top_rx_var.get(), self.top_ry_var.get(), self.top_rz_var.get()),
            SideSettings("Bottom", self.bottom_count_var.get(), self.bottom_pitch_var.get(), self.bottom_pitch_axis_var.get(), self.bottom_rx_var.get(), self.bottom_ry_var.get(), self.bottom_rz_var.get()),
            SideSettings("Left", self.left_count_var.get(), self.left_pitch_var.get(), self.left_pitch_axis_var.get(), self.left_rx_var.get(), self.left_ry_var.get(), self.left_rz_var.get()),
            SideSettings("Right", self.right_count_var.get(), self.right_pitch_var.get(), self.right_pitch_axis_var.get(), self.right_rx_var.get(), self.right_ry_var.get(), self.right_rz_var.get()),
        ]

    def _base_project_payload(self) -> dict:
        return build_ic_payload(
            self.profile.points_mm,
            leg_length_mm=float(self.leg_length_var.get()),
            body_width_mm=float(self.body_width_var.get()),
            body_depth_mm=float(self.body_depth_var.get()),
            body_height_mm=float(self.body_height_var.get()),
            side_settings=self._collect_side_settings(),
            status_message=self.status_var.get(),
        )

    def _project_payload(self) -> dict:
        payload = self._base_project_payload()
        payload.update(
            {
                "project_name": self.project_name_var.get().strip() or self.current_project_dir.name,
                "project_dir": str(self.current_project_dir),
                "current_step_index": self.step_index,
                "current_step_title": self.step_titles[self.step_index],
                "draft_points_px": [list(point) for point in self.current_points_px],
                "distance_target_mm": float(self.distance_target_var.get()),
                "lead_offset_mm": float(self.lead_offset_var.get()),
                "die_leadframe_ratio_percent": float(self.die_leadframe_ratio_var.get()),
                "die_leadframe_clearance_mm": float(self.die_leadframe_clearance_var.get()),
                "die_leadframe_thickness_mm": float(self.die_leadframe_thickness_var.get()),
                "silicon_die_width_mm": float(self.silicon_die_width_var.get()),
                "silicon_die_depth_mm": float(self.silicon_die_depth_var.get()),
                "silicon_die_thickness_mm": float(self.silicon_die_thickness_var.get()),
                "leg_pick_distance_mm": float(self.leg_pick_distance_var.get()),
                "leg_pick_marker_size_mm": float(self.leg_pick_marker_size_var.get()),
                "die_region_span_percent": float(self.die_region_span_percent_var.get()),
                "die_region_depth_mm": float(self.die_region_depth_var.get()),
                "die_region_offset_mm": float(self.die_region_offset_var.get()),
                "die_pick_region": self.die_pick_region_var.get().strip().title() or "Top",
                "die_pick_section_index": int(self.die_pick_section_index_var.get()),
                "die_pick_position_percent": float(self.die_pick_position_percent_var.get()),
                "die_pick_marker_size_mm": float(self.die_pick_marker_size_var.get()),
                "arc_height_mm": float(self.arc_height_var.get()),
                "arc_xy_noise_mm": float(self.arc_xy_noise_var.get()),
                "wire_arc_point_spacing_mm": float(self.wire_arc_point_spacing_var.get()),
                "ball_bond_diameter_mm": float(self.ball_bond_diameter_var.get()),
                "ball_bond_length_mm": float(self.ball_bond_length_var.get()),
                "ball_bond_revolution_steps": int(self.ball_bond_revolution_steps_var.get()),
                "wire_diameter_mm": float(self.wire_diameter_var.get()),
                "wire_rise_z_mm": float(self.wire_rise_z_var.get()),
                "wire_tube_side_count": int(self.wire_tube_side_count_var.get()),
                "wedge_bond_length_mm": float(self.wedge_bond_length_var.get()),
                "wedge_bond_width_mm": float(self.wedge_bond_width_var.get()),
                "wedge_bond_thickness_mm": float(self.wedge_bond_thickness_var.get()),
                "wedge_approach_run_mm": float(self.wedge_approach_run_var.get()),
                "distance_pick_active": self.distance_pick_active,
                "distance_point_indices": list(self.distance_point_indices),
            }
        )
        return payload

    def _stage_payload(self, stage_index: int) -> dict:
        side_settings = self._collect_side_settings()
        payload = {
            "saved_stage_index": stage_index,
            "saved_stage_title": self.step_titles[stage_index],
            "current_step_index": self.step_index,
            "current_step_title": self.step_titles[self.step_index],
            "status_message": self.status_var.get(),
        }
        if stage_index >= 0:
            payload["lead_profile"] = {
                "profile_points_mm": [list(point) for point in self.profile.points_mm],
                "draft_points_px": [list(point) for point in self.current_points_px],
                "distance_target_mm": float(self.distance_target_var.get()),
                "distance_point_indices": list(self.distance_point_indices),
            }
        if stage_index >= 1:
            payload["lead_extrusion"] = {
                "leg_length_mm": float(self.leg_length_var.get()),
            }
        if stage_index >= 2:
            payload["legs_per_side"] = [
                {
                    "name": side.name,
                    "count": side.count,
                    "pitch_mm": side.pitch_mm,
                    "pitch_axis": side.pitch_axis,
                    "rotation_x_deg": side.rotation_x_deg,
                    "rotation_y_deg": side.rotation_y_deg,
                    "rotation_z_deg": side.rotation_z_deg,
                }
                for side in side_settings
            ]
        if stage_index >= 3:
            payload["overall_3d_placement"] = {
                "body_width_mm": float(self.body_width_var.get()),
                "body_depth_mm": float(self.body_depth_var.get()),
                "body_height_mm": float(self.body_height_var.get()),
            }
        if stage_index >= 4:
            payload["lead_offset"] = {
                "lead_offset_mm": float(self.lead_offset_var.get()),
            }
        if stage_index >= 5:
            payload["die_leadframe"] = {
                "die_leadframe_ratio_percent": float(self.die_leadframe_ratio_var.get()),
                "die_leadframe_clearance_mm": float(self.die_leadframe_clearance_var.get()),
                "die_leadframe_thickness_mm": float(self.die_leadframe_thickness_var.get()),
            }
        if stage_index >= 6:
            payload["silicon_die"] = {
                "silicon_die_width_mm": float(self.silicon_die_width_var.get()),
                "silicon_die_depth_mm": float(self.silicon_die_depth_var.get()),
                "silicon_die_thickness_mm": float(self.silicon_die_thickness_var.get()),
            }
        if stage_index >= 7:
            payload["leg_positions"] = {
                "leg_pick_distance_mm": float(self.leg_pick_distance_var.get()),
                "leg_pick_marker_size_mm": float(self.leg_pick_marker_size_var.get()),
            }
        if stage_index >= 8:
            payload["die_regions"] = {
                "die_region_span_percent": float(self.die_region_span_percent_var.get()),
                "die_region_depth_mm": float(self.die_region_depth_var.get()),
                "die_region_offset_mm": float(self.die_region_offset_var.get()),
                "die_pick_region": self.die_pick_region_var.get().strip().title() or "Top",
                "die_pick_section_index": int(self.die_pick_section_index_var.get()),
                "die_pick_position_percent": float(self.die_pick_position_percent_var.get()),
                "die_pick_marker_size_mm": float(self.die_pick_marker_size_var.get()),
            }
        if stage_index >= 9:
            payload["bond_arcs"] = {
                "arc_height_mm": float(self.arc_height_var.get()),
                "arc_xy_noise_mm": float(self.arc_xy_noise_var.get()),
                "wire_arc_point_spacing_mm": float(self.wire_arc_point_spacing_var.get()),
            }
        if stage_index >= 10:
            payload["ball_bond_formation"] = {
                "ball_bond_diameter_mm": float(self.ball_bond_diameter_var.get()),
                "ball_bond_length_mm": float(self.ball_bond_length_var.get()),
                "ball_bond_revolution_steps": int(self.ball_bond_revolution_steps_var.get()),
            }
        if stage_index >= 11:
            payload["bond_wire_tube"] = {
                "wire_diameter_mm": float(self.wire_diameter_var.get()),
                "wire_rise_z_mm": float(self.wire_rise_z_var.get()),
                "wire_arc_point_spacing_mm": float(self.wire_arc_point_spacing_var.get()),
                "wire_tube_side_count": int(self.wire_tube_side_count_var.get()),
            }
        if stage_index >= 12:
            payload["wedge_bond_ending"] = {
                "wedge_bond_length_mm": float(self.wedge_bond_length_var.get()),
                "wedge_bond_width_mm": float(self.wedge_bond_width_var.get()),
                "wedge_bond_thickness_mm": float(self.wedge_bond_thickness_var.get()),
                "wedge_approach_run_mm": float(self.wedge_approach_run_var.get()),
            }
        return payload

    def _autosave_stage_jsons(self) -> None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROJECTS_ROOT_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_current_project_dir()
        self._project_snapshot_path().write_text(json.dumps(self._project_payload(), indent=2), encoding="utf-8")
        for stage_index, stage_title in enumerate(self.step_titles):
            filename = f"stage_{stage_index + 1}_{_safe_stage_slug(stage_title)}.json"
            stage_path = self._project_stage_dir() / filename
            stage_path.write_text(json.dumps(self._stage_payload(stage_index), indent=2), encoding="utf-8")

    def _export_combined_stl(self) -> None:
        try:
            payload = self._project_payload()
            meshes = collect_export_meshes(payload)
            if not meshes:
                raise ValueError("No solid meshes are available to export at the current step.")
            merged_mesh = trimesh.util.concatenate(meshes)
            self._ensure_current_project_dir()
            default_name = f"{_safe_stage_slug(self.project_name_var.get() or self.current_project_dir.name)}_combined.stl"
            target_path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Export Combined STL",
                initialdir=str(self.current_project_dir),
                initialfile=default_name,
                defaultextension=".stl",
                filetypes=[("STL files", "*.stl")],
            )
            if not target_path:
                self.status_var.set("Combined STL export cancelled.")
                return
            merged_mesh.export(target_path)
            self.status_var.set(f"Combined STL exported: {Path(target_path).name}")
        except Exception as exc:
            messagebox.showerror("Export Combined STL Failed", str(exc), parent=self.root)
            self.status_var.set(f"Combined STL export failed: {exc}")

    def _ensure_viewer_running(self) -> bool:
        if self.viewer_process is not None and self.viewer_process.poll() is None:
            return False
        self.viewer_process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--viewer-bridge",
                str(self.bridge_path),
            ],
            cwd=str(REPO_ROOT),
        )
        return True

    def _push_payload(self, launch_if_missing: bool = False) -> None:
        try:
            project_payload = self._project_payload()
            write_bridge_payload(self.bridge_path, project_payload)
            self._autosave_stage_jsons()
            if launch_if_missing:
                launched = self._ensure_viewer_running()
                if launched:
                    self.status_var.set("3D preview launched.")
        except Exception as exc:
            self.status_var.set(f"Viewer sync failed: {exc}")

    def _start_viewer(self) -> None:
        try:
            self._push_payload()
            launched = self._ensure_viewer_running()
            if launched:
                self.status_var.set("3D preview launched.")
            else:
                self.status_var.set("3D preview already running. Updating data instead.")
        except Exception as exc:
            messagebox.showerror("Viewer Launch Failed", str(exc), parent=self.root)
            self.status_var.set(f"Viewer launch failed: {exc}")

    def _on_close(self) -> None:
        if self.viewer_process is not None and self.viewer_process.poll() is None:
            self.viewer_process.terminate()
        if self.bridge_path.exists():
            try:
                self.bridge_path.unlink()
            except OSError:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step-by-step IC lead generator with 2D sketching and 3D placement preview.")
    parser.add_argument("--viewer-bridge", type=Path, default=None, help="Internal: viewer bridge JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.viewer_bridge is not None:
        viewer = IcLeadViewer(args.viewer_bridge.expanduser().resolve())
        viewer.run()
        return 0
    app = IcChipGeneratorApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
