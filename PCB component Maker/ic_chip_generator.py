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
from tkinter import colorchooser, filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
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
SIMULATION_PART_GAP_MM = 0.001
SNAP_ANGLE_THRESHOLD_DEG = 12.0
SNAP_HOVER_DELAY_MS = 2000
POINT_AXIS_SNAP_THRESHOLD_PX = 10.0
PATH_ANGLE_SNAP_THRESHOLD_DEG = 3.0
LEADFRAME_MIRROR_VALIDATION_THRESHOLD_PX = 12.0
LEADFRAME_KEEPOUT_SCALE = 1.1
BOOLEAN_CLEANUP_EPSILON_MM = 0.005


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
        payload = json.loads(bridge_path.read_text(encoding="utf-8-sig"))
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


def _simplify_path_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    simplified: list[tuple[float, float]] = []
    for point in points:
        if not simplified or point != simplified[-1]:
            simplified.append(point)
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


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 3:
        return (0.0, 0.0)
    signed_area = _signed_area(points)
    if abs(signed_area) <= 1e-9:
        avg_x = sum(point[0] for point in points) / len(points)
        avg_y = sum(point[1] for point in points) / len(points)
        return (avg_x, avg_y)
    factor = 1.0 / (6.0 * signed_area)
    centroid_x = 0.0
    centroid_y = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        cross = (x1 * y2) - (x2 * y1)
        centroid_x += (x1 + x2) * cross
        centroid_y += (y1 + y2) * cross
    return (centroid_x * factor, centroid_y * factor)


def _leadframe_shapes_from_payload(payload: dict) -> list[list[tuple[float, float]]]:
    shapes_payload = payload.get("leadframe_profiles_points_mm", [])
    shapes: list[list[tuple[float, float]]] = []
    if isinstance(shapes_payload, list):
        for shape in shapes_payload:
            if not isinstance(shape, list):
                continue
            points = [
                tuple(point)
                for point in shape
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            if points:
                shapes.append(points)
    if shapes:
        return shapes
    legacy_points = [
        tuple(point)
        for point in payload.get("leadframe_profile_points_mm", [])
        if isinstance(point, list | tuple) and len(point) == 2
    ]
    return [legacy_points] if legacy_points else []


def _leadframe_profiles_from_payload(payload: dict) -> list[LeadProfile]:
    profiles_payload = payload.get("leadframe_profiles", [])
    profiles: list[LeadProfile] = []
    if isinstance(profiles_payload, list):
        for profile_payload in profiles_payload:
            if not isinstance(profile_payload, dict):
                continue
            points = [
                tuple(point)
                for point in profile_payload.get("points_mm", [])
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            if not points:
                continue
            profiles.append(LeadProfile(points_mm=points, closed=bool(profile_payload.get("closed", False))))
    if profiles:
        return profiles
    return [LeadProfile(points_mm=shape, closed=True) for shape in _leadframe_shapes_from_payload(payload)]


def _combined_leadframe_centroid(shapes: list[list[tuple[float, float]]]) -> tuple[float, float]:
    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0
    for points in shapes:
        if len(points) < 3:
            continue
        area = abs(_signed_area(points))
        if area <= 1e-9:
            continue
        centroid_x, centroid_y = _polygon_centroid(points)
        weighted_x += centroid_x * area
        weighted_y += centroid_y * area
        total_weight += area
    if total_weight > 1e-9:
        return (weighted_x / total_weight, weighted_y / total_weight)
    flattened = [point for shape in shapes for point in shape]
    if flattened:
        return (
            sum(point[0] for point in flattened) / len(flattened),
            sum(point[1] for point in flattened) / len(flattened),
        )
    return (0.0, 0.0)


def _lead_profile_join_distance_mm(profile_points: list[tuple[float, float]]) -> float:
    if len(profile_points) >= 2:
        return max(0.01, math.dist(profile_points[0], profile_points[-1]))
    if profile_points:
        x_values = [point[0] for point in profile_points]
        y_values = [point[1] for point in profile_points]
        return max(0.01, min(max(x_values) - min(x_values), max(y_values) - min(y_values), 0.2))
    return 0.01


def _normalize_2d(vector_xy: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector_xy[0], vector_xy[1])
    if length <= 1e-12:
        return None
    return (vector_xy[0] / length, vector_xy[1] / length)


def _left_normal_2d(direction_xy: tuple[float, float]) -> tuple[float, float]:
    return (-direction_xy[1], direction_xy[0])


def _line_intersection_2d(
    point_a: tuple[float, float],
    direction_a: tuple[float, float],
    point_b: tuple[float, float],
    direction_b: tuple[float, float],
) -> tuple[float, float] | None:
    denominator = (direction_a[0] * direction_b[1]) - (direction_a[1] * direction_b[0])
    if abs(denominator) <= 1e-12:
        return None
    delta_x = point_b[0] - point_a[0]
    delta_y = point_b[1] - point_a[1]
    factor_a = ((delta_x * direction_b[1]) - (delta_y * direction_b[0])) / denominator
    return (
        point_a[0] + (direction_a[0] * factor_a),
        point_a[1] + (direction_a[1] * factor_a),
    )


def _build_stroked_path_polygon(points_mm: list[tuple[float, float]], width_mm: float) -> list[tuple[float, float]]:
    simplified_points = _simplify_path_points(points_mm)
    if len(simplified_points) < 2:
        raise ValueError("Path needs at least 2 unique points.")
    half_width_mm = width_mm / 2.0
    segment_directions: list[tuple[float, float]] = []
    segment_normals: list[tuple[float, float]] = []
    for start_point_mm, end_point_mm in zip(simplified_points[:-1], simplified_points[1:]):
        direction_xy = _normalize_2d((end_point_mm[0] - start_point_mm[0], end_point_mm[1] - start_point_mm[1]))
        if direction_xy is None:
            continue
        segment_directions.append(direction_xy)
        segment_normals.append(_left_normal_2d(direction_xy))
    if not segment_directions:
        raise ValueError("Path segments are degenerate.")

    left_points: list[tuple[float, float]] = [(
        simplified_points[0][0] + (segment_normals[0][0] * half_width_mm),
        simplified_points[0][1] + (segment_normals[0][1] * half_width_mm),
    )]
    right_points: list[tuple[float, float]] = [(
        simplified_points[0][0] - (segment_normals[0][0] * half_width_mm),
        simplified_points[0][1] - (segment_normals[0][1] * half_width_mm),
    )]

    for vertex_index in range(1, len(simplified_points) - 1):
        point_mm = simplified_points[vertex_index]
        prev_direction = segment_directions[vertex_index - 1]
        next_direction = segment_directions[vertex_index]
        prev_normal = segment_normals[vertex_index - 1]
        next_normal = segment_normals[vertex_index]

        left_intersection = _line_intersection_2d(
            (point_mm[0] + (prev_normal[0] * half_width_mm), point_mm[1] + (prev_normal[1] * half_width_mm)),
            prev_direction,
            (point_mm[0] + (next_normal[0] * half_width_mm), point_mm[1] + (next_normal[1] * half_width_mm)),
            next_direction,
        )
        if left_intersection is None:
            average_left = _normalize_2d((prev_normal[0] + next_normal[0], prev_normal[1] + next_normal[1]))
            if average_left is None:
                average_left = prev_normal
            left_intersection = (
                point_mm[0] + (average_left[0] * half_width_mm),
                point_mm[1] + (average_left[1] * half_width_mm),
            )
        left_points.append(left_intersection)

        right_intersection = _line_intersection_2d(
            (point_mm[0] - (prev_normal[0] * half_width_mm), point_mm[1] - (prev_normal[1] * half_width_mm)),
            prev_direction,
            (point_mm[0] - (next_normal[0] * half_width_mm), point_mm[1] - (next_normal[1] * half_width_mm)),
            next_direction,
        )
        if right_intersection is None:
            average_right = _normalize_2d((-(prev_normal[0] + next_normal[0]), -(prev_normal[1] + next_normal[1])))
            if average_right is None:
                average_right = (-prev_normal[0], -prev_normal[1])
            right_intersection = (
                point_mm[0] + (average_right[0] * half_width_mm),
                point_mm[1] + (average_right[1] * half_width_mm),
            )
        right_points.append(right_intersection)

    last_normal = segment_normals[-1]
    left_points.append((
        simplified_points[-1][0] + (last_normal[0] * half_width_mm),
        simplified_points[-1][1] + (last_normal[1] * half_width_mm),
    ))
    right_points.append((
        simplified_points[-1][0] - (last_normal[0] * half_width_mm),
        simplified_points[-1][1] - (last_normal[1] * half_width_mm),
    ))
    return left_points + list(reversed(right_points))


def _build_path_segment_mesh(
    start_point_mm: tuple[float, float],
    end_point_mm: tuple[float, float],
    width_mm: float,
    height_mm: float,
    z_center_mm: float,
) -> trimesh.Trimesh | None:
    dx = end_point_mm[0] - start_point_mm[0]
    dy = end_point_mm[1] - start_point_mm[1]
    length_mm = math.hypot(dx, dy)
    if length_mm <= 1e-9:
        return None
    mesh = trimesh.creation.box(extents=(length_mm, width_mm, height_mm))
    angle_rad = math.atan2(dy, dx)
    transform = matrix_multiply(
        translation_transform(
            (start_point_mm[0] + end_point_mm[0]) / 2.0,
            (start_point_mm[1] + end_point_mm[1]) / 2.0,
            z_center_mm,
        ),
        rotation_matrix_xyz((0.0, 0.0, math.degrees(angle_rad))),
    )
    mesh.apply_transform(transform)
    return mesh


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
    lead_offset_mm = float(payload.get("lead_offset_mm", 0.0)) + SIMULATION_PART_GAP_MM

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
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    body_width_mm = float(payload.get("body_width_mm", 0.0))
    body_depth_mm = float(payload.get("body_depth_mm", 0.0))
    requested_width_mm = max(0.01, float(payload.get("die_leadframe_width_mm", max(body_width_mm * 0.8, 0.01))))
    requested_depth_mm = max(0.01, float(payload.get("die_leadframe_depth_mm", max(body_depth_mm * 0.8, 0.01))))
    thickness_mm = max(0.01, float(payload.get("die_leadframe_thickness_mm", 0.08)))
    center_mode = str(payload.get("die_leadframe_center_mode", "region_centroid")).strip().lower()
    custom_center_x_mm = float(payload.get("die_leadframe_center_x_mm", 0.0))
    custom_center_y_mm = float(payload.get("die_leadframe_center_y_mm", 0.0))
    if body_width_mm <= 0.0 or body_depth_mm <= 0.0:
        return None, {}
    leadframe_shapes = _leadframe_shapes_from_payload(payload)
    region_center_x_mm, region_center_y_mm = _combined_leadframe_centroid(leadframe_shapes)
    if center_mode == "custom_point":
        center_x_mm = custom_center_x_mm
        center_y_mm = custom_center_y_mm
    else:
        center_x_mm = region_center_x_mm
        center_y_mm = region_center_y_mm

    mesh = trimesh.creation.box(
        extents=(requested_width_mm, requested_depth_mm, thickness_mm),
    )
    mesh.apply_translation([center_x_mm, center_y_mm, body_height_mm + SIMULATION_PART_GAP_MM + (thickness_mm / 2.0)])
    return mesh, {
        "final_width_mm": requested_width_mm,
        "final_depth_mm": requested_depth_mm,
        "thickness_mm": thickness_mm,
        "center_mode": center_mode,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "source_region_shape_count": len(leadframe_shapes),
    }


def build_sketched_leadframe_meshes(payload: dict) -> list[trimesh.Trimesh]:
    profile_points = [
        tuple(point)
        for point in payload.get("profile_points_mm", [])
        if isinstance(point, list | tuple) and len(point) == 2
    ]
    leadframe_profiles = _leadframe_profiles_from_payload(payload)
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    path_width_mm = max(0.01, float(payload.get("leadframe_path_width_mm", 0.3)))
    if not profile_points or not leadframe_profiles or body_height_mm <= 0.0:
        return []

    try:
        _body_mesh, side_meshes = build_ic_meshes(payload)
    except Exception:
        side_meshes = []
    extrusion_height_mm = max(0.01, float(payload.get("leadframe_path_thickness_mm", 1.0)))
    # Keep the lead-frame paths slightly above the package body so they do not share triangles.
    z_offset_mm = body_height_mm + SIMULATION_PART_GAP_MM
    meshes: list[trimesh.Trimesh] = []
    z_center_mm = z_offset_mm + (extrusion_height_mm / 2.0)
    for profile in leadframe_profiles:
        if profile.closed and len(profile.points_mm) >= 3:
            try:
                mesh = extrude_closed_polygon(profile.points_mm, extrusion_height_mm)
            except Exception:
                continue
            mesh.apply_translation([0.0, 0.0, z_offset_mm])
            meshes.append(mesh)
            continue
        if len(profile.points_mm) < 2:
            continue
        try:
            stroked_polygon_points = _build_stroked_path_polygon(profile.points_mm, path_width_mm)
            mesh = extrude_closed_polygon(stroked_polygon_points, extrusion_height_mm)
        except Exception:
            segment_meshes: list[trimesh.Trimesh] = []
            for start_point_mm, end_point_mm in zip(profile.points_mm[:-1], profile.points_mm[1:]):
                segment_mesh = _build_path_segment_mesh(start_point_mm, end_point_mm, path_width_mm, extrusion_height_mm, z_center_mm)
                if segment_mesh is not None:
                    segment_meshes.append(segment_mesh)
            if segment_meshes:
                meshes.append(trimesh.util.concatenate(segment_meshes))
            continue
        mesh.apply_translation([0.0, 0.0, z_offset_mm])
        meshes.append(mesh)
    return meshes


def build_combined_lead_system_mesh(
    payload: dict,
    side_meshes: list[tuple[str, trimesh.Trimesh]],
) -> tuple[trimesh.Trimesh | None, list[trimesh.Trimesh]]:
    sketched_leadframe_meshes = build_sketched_leadframe_meshes(payload)
    lead_meshes = [mesh for _side_name, mesh in side_meshes]
    combined_parts = lead_meshes + sketched_leadframe_meshes
    if not combined_parts:
        return None, sketched_leadframe_meshes
    return trimesh.util.concatenate(combined_parts), sketched_leadframe_meshes


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
    mesh.apply_translation([center_x, center_y, top_z + SIMULATION_PART_GAP_MM + (thickness_mm / 2.0)])
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


def build_leadframe_path_end_markers(payload: dict) -> list[dict]:
    leadframe_profiles = _leadframe_profiles_from_payload(payload)
    body_height_mm = float(payload.get("body_height_mm", 0.0))
    thickness_mm = max(0.01, float(payload.get("leadframe_path_thickness_mm", 1.0)))
    pick_distance_mm = float(payload.get("leg_pick_distance_mm", 0.2))
    marker_size_mm = max(0.02, float(payload.get("leg_pick_marker_size_mm", 0.08)))
    marker_thickness_mm = min(marker_size_mm * 0.25, 0.02)
    if body_height_mm <= 0.0:
        return []

    anchor_z = body_height_mm + SIMULATION_PART_GAP_MM + thickness_mm + COMPONENT_CLEARANCE_MM
    markers: list[dict] = []
    for profile in leadframe_profiles:
        if profile.closed or len(profile.points_mm) < 2:
            continue
        end_x, end_y = profile.points_mm[-1]
        previous_x, previous_y = profile.points_mm[-2]
        start_x, start_y = profile.points_mm[0]
        dx = end_x - start_x
        dy = end_y - start_y
        segment_dx = previous_x - end_x
        segment_dy = previous_y - end_y
        segment_length = math.hypot(segment_dx, segment_dy)
        if segment_length > 1e-9:
            offset_distance_mm = max(-segment_length, min(segment_length, pick_distance_mm))
            target_x = end_x + ((segment_dx / segment_length) * offset_distance_mm)
            target_y = end_y + ((segment_dy / segment_length) * offset_distance_mm)
        else:
            target_x = end_x
            target_y = end_y
        if abs(dx) >= abs(dy):
            side_name = "Right" if dx >= 0.0 else "Left"
        else:
            side_name = "Top" if dy >= 0.0 else "Bottom"
        marker = trimesh.creation.box(extents=(marker_size_mm, marker_size_mm, marker_thickness_mm))
        marker.apply_translation([target_x, target_y, anchor_z - (COMPONENT_CLEARANCE_MM / 2.0)])
        markers.append(
            {
                "side_name": side_name,
                "mesh": marker,
                "center_xy": (target_x, target_y),
                "anchor_xyz": (target_x, target_y, anchor_z),
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

    region_counts = {
        "Top": max(0, int(float(payload.get("die_region_top_count", 0)))),
        "Bottom": max(0, int(float(payload.get("die_region_bottom_count", 0)))),
        "Left": max(0, int(float(payload.get("die_region_left_count", 0)))),
        "Right": max(0, int(float(payload.get("die_region_right_count", 0)))),
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

    add_segmented_regions("Top", region_counts.get("Top", 0), horizontal_span, top_bottom_depth)
    add_segmented_regions("Bottom", region_counts.get("Bottom", 0), horizontal_span, top_bottom_depth)
    add_segmented_regions("Left", region_counts.get("Left", 0), vertical_span, left_right_depth)
    add_segmented_regions("Right", region_counts.get("Right", 0), vertical_span, left_right_depth)

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
        "side_region_counts": region_counts,
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


def _pair_nearest_leg_regions(leg_markers: list[dict], die_regions: list[dict]) -> list[tuple[dict, dict]]:
    if not leg_markers or not die_regions:
        return []

    remaining_regions = list(die_regions)
    pairs: list[tuple[dict, dict]] = []
    for leg_data in leg_markers:
        leg_x, leg_y = leg_data.get("center_xy", (0.0, 0.0))
        nearest_index = min(
            range(len(remaining_regions)),
            key=lambda index: math.dist((leg_x, leg_y), remaining_regions[index].get("center_xy", (0.0, 0.0))),
        )
        region_data = remaining_regions.pop(nearest_index)
        pairs.append((leg_data, region_data))
        if not remaining_regions:
            break
    return pairs


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

    def point_distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
        return math.dist(point_a, point_b)

    def tj(ti: float, point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
        return ti + max(point_distance(point_a, point_b), 1e-9) ** 0.5

    def interpolate(
        point_a: tuple[float, float, float],
        point_b: tuple[float, float, float],
        ta: float,
        tb: float,
        t_value: float,
    ) -> tuple[float, float, float]:
        if abs(tb - ta) <= 1e-9:
            return point_a
        blend_a = (tb - t_value) / (tb - ta)
        blend_b = (t_value - ta) / (tb - ta)
        return (
            (point_a[0] * blend_a) + (point_b[0] * blend_b),
            (point_a[1] * blend_a) + (point_b[1] * blend_b),
            (point_a[2] * blend_a) + (point_b[2] * blend_b),
        )

    extended_points = [control_points[0], *control_points, control_points[-1]]
    segment_count = len(control_points) - 1
    spline_points: list[list[float]] = []

    for segment_index in range(segment_count):
        p0 = extended_points[segment_index]
        p1 = extended_points[segment_index + 1]
        p2 = extended_points[segment_index + 2]
        p3 = extended_points[segment_index + 3]

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        segment_samples = max(2, int(round(sample_count / segment_count)))
        if segment_index == segment_count - 1:
            segment_range = range(segment_samples + 1)
        else:
            segment_range = range(segment_samples)

        for local_index in segment_range:
            t_value = t1 + ((t2 - t1) * (local_index / segment_samples))
            a1 = interpolate(p0, p1, t0, t1, t_value)
            a2 = interpolate(p1, p2, t1, t2, t_value)
            a3 = interpolate(p2, p3, t2, t3, t_value)
            b1 = interpolate(a1, a2, t0, t2, t_value)
            b2 = interpolate(a2, a3, t1, t3, t_value)
            point = interpolate(b1, b2, t1, t2, t_value)
            spline_points.append([point[0], point[1], point[2]])

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


def _vector_dot(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float]) -> float:
    return (a_vec[0] * b_vec[0]) + (a_vec[1] * b_vec[1]) + (a_vec[2] * b_vec[2])


def _vector_lerp(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float], t_value: float) -> tuple[float, float, float]:
    return (
        a_vec[0] + ((b_vec[0] - a_vec[0]) * t_value),
        a_vec[1] + ((b_vec[1] - a_vec[1]) * t_value),
        a_vec[2] + ((b_vec[2] - a_vec[2]) * t_value),
    )


def _choose_frame_reference(tangent: tuple[float, float, float]) -> tuple[float, float, float]:
    for candidate in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        if _vector_length(_vector_cross(candidate, tangent)) > 1e-6:
            return candidate
    return (1.0, 0.0, 0.0)


def _build_swept_tube_mesh(
    points: list[tuple[float, float, float]],
    radius_mm: float,
    side_count: int,
) -> trimesh.Trimesh | None:
    if len(points) < 2:
        return None

    tangents: list[tuple[float, float, float]] = []
    for index in range(len(points)):
        if index == 0:
            tangent = _vector_normalize(_vector_sub(points[1], points[0]))
        elif index == len(points) - 1:
            tangent = _vector_normalize(_vector_sub(points[-1], points[-2]))
        else:
            tangent = _vector_normalize(_vector_sub(points[index + 1], points[index - 1]))
        tangents.append(tangent)

    sections: list[list[tuple[float, float, float]]] = []
    lateral: tuple[float, float, float] | None = None
    vertical: tuple[float, float, float] | None = None
    for center, tangent in zip(points, tangents):
        if lateral is None or vertical is None:
            reference = _choose_frame_reference(tangent)
            lateral = _vector_normalize(_vector_cross(reference, tangent))
            vertical = _vector_normalize(_vector_cross(tangent, lateral))
        else:
            projected_lateral = _vector_sub(lateral, _vector_scale(tangent, _vector_dot(lateral, tangent)))
            if _vector_length(projected_lateral) <= 1e-6:
                projected_vertical = _vector_sub(vertical, _vector_scale(tangent, _vector_dot(vertical, tangent)))
                if _vector_length(projected_vertical) > 1e-6:
                    vertical = _vector_normalize(projected_vertical)
                    projected_lateral = _vector_cross(vertical, tangent)
                else:
                    reference = _choose_frame_reference(tangent)
                    projected_lateral = _vector_cross(reference, tangent)
            lateral = _vector_normalize(projected_lateral)
            vertical = _vector_normalize(_vector_cross(tangent, lateral))

        section: list[tuple[float, float, float]] = []
        for side_index in range(side_count):
            angle = (2.0 * math.pi * side_index) / side_count
            offset = _vector_add(
                _vector_scale(lateral, math.cos(angle) * radius_mm),
                _vector_scale(vertical, math.sin(angle) * radius_mm),
            )
            section.append(_vector_add(center, offset))
        sections.append(section)

    vertices: list[list[float]] = []
    for section in sections:
        for vertex in section:
            vertices.append([vertex[0], vertex[1], vertex[2]])

    faces: list[list[int]] = []
    section_size = side_count
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
    vertices.append(list(points[0]))
    for vertex_index in range(section_size):
        next_vertex_index = (vertex_index + 1) % section_size
        faces.append([start_center_index, next_vertex_index, vertex_index])

    end_center_index = len(vertices)
    vertices.append(list(points[-1]))
    end_base_index = (len(sections) - 1) * section_size
    for vertex_index in range(section_size):
        next_vertex_index = (vertex_index + 1) % section_size
        faces.append([end_center_index, end_base_index + vertex_index, end_base_index + next_vertex_index])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_connection_paths(
    payload: dict,
    leg_markers: list[dict],
    die_regions: list[dict],
    ball_bond_meshes: list[dict] | None = None,
) -> list[dict]:
    resolved_leg_markers = build_leadframe_path_end_markers(payload) or _clockwise_indexed_items(leg_markers)
    paired_anchors = _pair_nearest_leg_regions(resolved_leg_markers, die_regions)
    if not paired_anchors:
        return []

    arc_height_mm = max(0.0, float(payload.get("arc_height_mm", 0.5)))
    arc_xy_noise_mm = max(0.0, float(payload.get("arc_xy_noise_mm", 0.0)))
    point_spacing_mm = max(0.01, float(payload.get("wire_arc_point_spacing_mm", 0.08)))
    wire_rise_z_mm = max(0.01, float(payload.get("wire_rise_z_mm", 0.12)))
    wedge_approach_run_mm = max(0.02, float(payload.get("wedge_approach_run_mm", 0.18)))
    indexed_ball_bonds = ball_bond_meshes or []
    arcs: list[dict] = []

    for index, (leg_data, die_data) in enumerate(paired_anchors):
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
        rise_control = (start_x, start_y, mid_z)
        landing_height = end_z + min(max(wire_rise_z_mm * 0.15, 0.01), max(arc_height_mm * 0.2, 0.01))
        landing_control = (landing_x, landing_y, landing_height)
        approximate_length = (
            math.dist(start, rise_point)
            + math.dist(rise_point, rise_control)
            + math.dist(rise_control, landing_control)
            + math.dist(landing_control, end)
        )
        sample_count = max(12, int(math.ceil(approximate_length / point_spacing_mm)))

        rise_length = math.dist(start, rise_point)
        rise_sample_count = max(2, int(math.ceil(rise_length / point_spacing_mm)))
        curve_sample_count = max(8, sample_count - rise_sample_count + 1)

        rise_points = [
            [
                start_x,
                start_y,
                start_z + ((wire_rise_z_mm * sample_index) / rise_sample_count),
            ]
            for sample_index in range(rise_sample_count + 1)
        ]
        curve_points = _sample_cubic_bezier(rise_point, rise_control, landing_control, end, curve_sample_count)
        points = rise_points[:-1] + curve_points

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
    resolved_leg_markers = build_leadframe_path_end_markers(payload) or _clockwise_indexed_items(leg_markers)
    paired_anchors = _pair_nearest_leg_regions(resolved_leg_markers, die_regions)
    if not paired_anchors:
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
    for index, (_leg_data, die_data) in enumerate(paired_anchors):
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
        points = [tuple(point) for point in _trim_polyline_from_end(path_data.get("points", []), trim_end_distance_mm)]
        if len(points) < 2:
            continue
        swept_mesh = _build_swept_tube_mesh(points, wire_radius_mm, tube_side_count)
        if swept_mesh is None:
            continue
        tube_meshes.append(
            {
                "mesh": swept_mesh,
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


def build_selected_bond_assemblies(
    current_step_index: int,
    ball_bond_meshes: list[dict],
    ball_bond_wire_meshes: list[dict],
    ball_bond_terminal_meshes: list[dict],
) -> list[dict]:
    if current_step_index >= 13:
        return ball_bond_terminal_meshes
    if current_step_index >= 12:
        return ball_bond_wire_meshes
    if current_step_index >= 11:
        return ball_bond_meshes
    return []


def _expand_mesh_from_centroid(mesh: trimesh.Trimesh, offset_mm: float) -> trimesh.Trimesh:
    expanded = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    if offset_mm <= 1e-9:
        return expanded

    min_corner = expanded.bounds[0]
    max_corner = expanded.bounds[1]
    extents = max_corner - min_corner
    if len(extents) != 3 or float(np.max(extents)) <= 1e-9:
        return expanded

    centroid = expanded.centroid
    centered_vertices = expanded.vertices - centroid
    scale_factors = np.ones(3, dtype=float)
    for axis_index in range(3):
        axis_extent = float(extents[axis_index])
        if axis_extent <= 1e-9:
            continue
        scale_factors[axis_index] = (axis_extent + (2.0 * offset_mm)) / axis_extent
    expanded.vertices = centroid + (centered_vertices * scale_factors)
    try:
        expanded.remove_unreferenced_vertices()
        expanded.merge_vertices()
        expanded.fix_normals()
    except Exception:
        pass
    return expanded


def _solid_mesh_components(mesh: trimesh.Trimesh, volume_epsilon: float = 1e-9) -> list[trimesh.Trimesh]:
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = [mesh]

    solid_components: list[trimesh.Trimesh] = []
    for component in components:
        if not isinstance(component, trimesh.Trimesh) or len(component.faces) == 0:
            continue
        candidate = trimesh.Trimesh(vertices=component.vertices.copy(), faces=component.faces.copy(), process=False)
        try:
            candidate.remove_unreferenced_vertices()
            candidate.merge_vertices()
            candidate.fix_normals()
        except Exception:
            pass
        try:
            is_solid = bool(candidate.is_watertight and abs(float(candidate.volume)) > volume_epsilon)
        except Exception:
            is_solid = bool(candidate.is_watertight)
        if is_solid:
            solid_components.append(candidate)
    return solid_components


def _subtract_meshes(
    subject_mesh: trimesh.Trimesh,
    cutter_meshes: list[tuple[str, trimesh.Trimesh]],
) -> tuple[trimesh.Trimesh, list[str], dict[str, str]]:
    result_mesh = trimesh.Trimesh(vertices=subject_mesh.vertices.copy(), faces=subject_mesh.faces.copy(), process=False)
    warnings: list[str] = []
    status_map: dict[str, str] = {}
    for cutter_name, cutter_mesh in cutter_meshes:
        solid_cutters = _solid_mesh_components(cutter_mesh)
        if not solid_cutters:
            warnings.append(f"Boolean subtract skipped for {cutter_name}: cutter is not a closed solid volume.")
            status_map[cutter_name] = "skipped: not a closed solid volume"
            continue
        cutter_success = False
        for component_index, solid_cutter in enumerate(solid_cutters, start=1):
            try:
                difference_mesh = trimesh.boolean.difference([result_mesh, solid_cutter])
            except Exception as exc:
                suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                warnings.append(f"Boolean subtract failed for {cutter_name}{suffix}: {exc}")
                continue
            if difference_mesh is None:
                suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                warnings.append(f"Boolean subtract returned no mesh for {cutter_name}{suffix}.")
                continue
            if isinstance(difference_mesh, list):
                valid_meshes = [mesh for mesh in difference_mesh if isinstance(mesh, trimesh.Trimesh)]
                if not valid_meshes:
                    suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                    warnings.append(f"Boolean subtract produced no valid mesh for {cutter_name}{suffix}.")
                    continue
                result_mesh = trimesh.util.concatenate(valid_meshes)
            else:
                result_mesh = difference_mesh
            residual_faces_a, residual_faces_b = _find_intersecting_face_indices(result_mesh, solid_cutter, stop_at_first=True)
            residual_intersection = bool(residual_faces_a and residual_faces_b)
            if residual_intersection:
                try:
                    cleanup_cutter = _expand_mesh_from_centroid(solid_cutter, BOOLEAN_CLEANUP_EPSILON_MM)
                    cleanup_difference = trimesh.boolean.difference([result_mesh, cleanup_cutter])
                    if isinstance(cleanup_difference, list):
                        cleanup_meshes = [mesh for mesh in cleanup_difference if isinstance(mesh, trimesh.Trimesh)]
                        if cleanup_meshes:
                            result_mesh = trimesh.util.concatenate(cleanup_meshes)
                    elif isinstance(cleanup_difference, trimesh.Trimesh):
                        result_mesh = cleanup_difference
                    residual_faces_a, residual_faces_b = _find_intersecting_face_indices(result_mesh, solid_cutter, stop_at_first=True)
                    residual_intersection = bool(residual_faces_a and residual_faces_b)
                    if residual_intersection:
                        suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                        warnings.append(
                            f"Residual overlap remained after subtracting {cutter_name}{suffix}, even after cleanup epsilon."
                        )
                    else:
                        suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                        warnings.append(
                            f"Residual overlap for {cutter_name}{suffix} was cleaned with an extra {BOOLEAN_CLEANUP_EPSILON_MM:.3f} mm expansion."
                        )
                except Exception as exc:
                    suffix = f" component {component_index}" if len(solid_cutters) > 1 else ""
                    warnings.append(f"Cleanup subtract failed for {cutter_name}{suffix}: {exc}")
            cutter_success = True
        if cutter_success:
            status_map[cutter_name] = "subtracted"
        elif cutter_name not in status_map:
            status_map[cutter_name] = "failed"
    return result_mesh, warnings, status_map


def build_encapsulation_meshes(
    payload: dict,
    body_mesh: trimesh.Trimesh | None,
    side_meshes: list[tuple[str, trimesh.Trimesh]],
    sketched_leadframe_meshes: list[trimesh.Trimesh],
    die_leadframe_mesh: trimesh.Trimesh | None,
    silicon_die_mesh: trimesh.Trimesh | None,
    bond_assembly_meshes: list[dict],
) -> tuple[dict[str, trimesh.Trimesh | None], list[str], dict[str, dict[str, str]]]:
    if body_mesh is None:
        return {"base": None, "top": None}, ["Encapsulation requires the step 6 base body."], {"base": {}, "top": {}}

    body_width_mm = max(0.01, float(payload.get("body_width_mm", 0.0)))
    body_depth_mm = max(0.01, float(payload.get("body_depth_mm", 0.0)))
    body_height_mm = max(0.01, float(payload.get("body_height_mm", 0.0)))
    encapsulation_height_mm = max(0.01, float(payload.get("encapsulation_height_mm", body_height_mm)))
    clearance_mm = max(0.0, float(payload.get("simulation_clearance_mm", 0.001)))

    top_mesh = trimesh.creation.box(extents=(body_width_mm, body_depth_mm, encapsulation_height_mm))
    top_mesh.apply_translation([0.0, 0.0, body_height_mm + (encapsulation_height_mm / 2.0)])

    cutter_sources: list[tuple[str, trimesh.Trimesh | None]] = []
    for lead_index, (_side_name, lead_mesh) in enumerate(side_meshes, start=1):
        cutter_sources.append((f"Lead System {lead_index}", lead_mesh))
    for path_index, path_mesh in enumerate(sketched_leadframe_meshes, start=1):
        cutter_sources.append((f"Lead Path {path_index}", path_mesh))
    cutter_sources.extend(
        [
            ("Die Leadframe", die_leadframe_mesh),
            ("Silicon Die", silicon_die_mesh),
        ]
    )
    for bond_index, bond_data in enumerate(bond_assembly_meshes, start=1):
        bond_mesh = bond_data.get("mesh")
        if isinstance(bond_mesh, trimesh.Trimesh):
            cutter_sources.append((f"Bond Assembly {bond_index}", bond_mesh))

    expanded_cutters: list[tuple[str, trimesh.Trimesh]] = []
    warnings: list[str] = []
    for cutter_name, cutter_mesh in cutter_sources:
        if cutter_mesh is None:
            continue
        try:
            expanded_cutters.append((cutter_name, _expand_mesh_from_centroid(cutter_mesh, clearance_mm)))
        except Exception as exc:
            warnings.append(f"Failed to enlarge {cutter_name}: {exc}")

    base_result = trimesh.Trimesh(vertices=body_mesh.vertices.copy(), faces=body_mesh.faces.copy(), process=False)
    top_result = trimesh.Trimesh(vertices=top_mesh.vertices.copy(), faces=top_mesh.faces.copy(), process=False)
    subtraction_status = {"base": {}, "top": {}}
    if expanded_cutters:
        base_result, base_warnings, base_status = _subtract_meshes(base_result, expanded_cutters)
        top_result, top_warnings, top_status = _subtract_meshes(top_result, expanded_cutters)
        warnings.extend(base_warnings)
        warnings.extend(top_warnings)
        subtraction_status = {"base": base_status, "top": top_status}

    return {"base": base_result, "top": top_result}, warnings, subtraction_status


def _bounds_overlap(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, tolerance_mm: float = 1e-6) -> bool:
    min_a, max_a = mesh_a.bounds
    min_b, max_b = mesh_b.bounds
    overlap_x = min(max_a[0], max_b[0]) - max(min_a[0], min_b[0])
    overlap_y = min(max_a[1], max_b[1]) - max(min_a[1], min_b[1])
    overlap_z = min(max_a[2], max_b[2]) - max(min_a[2], min_b[2])
    return overlap_x > tolerance_mm and overlap_y > tolerance_mm and overlap_z > tolerance_mm


def _project_points_to_2d(points: np.ndarray, drop_axis: int) -> np.ndarray:
    return np.delete(points, drop_axis, axis=1)


def _orientation_2d(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, epsilon: float = 1e-9) -> int:
    value = ((point_b[0] - point_a[0]) * (point_c[1] - point_a[1])) - ((point_b[1] - point_a[1]) * (point_c[0] - point_a[0]))
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _on_segment_2d(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, epsilon: float = 1e-9) -> bool:
    return (
        min(point_a[0], point_c[0]) - epsilon <= point_b[0] <= max(point_a[0], point_c[0]) + epsilon
        and min(point_a[1], point_c[1]) - epsilon <= point_b[1] <= max(point_a[1], point_c[1]) + epsilon
    )


def _segments_intersect_2d(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
    epsilon: float = 1e-9,
) -> bool:
    orient_1 = _orientation_2d(start_a, end_a, start_b, epsilon)
    orient_2 = _orientation_2d(start_a, end_a, end_b, epsilon)
    orient_3 = _orientation_2d(start_b, end_b, start_a, epsilon)
    orient_4 = _orientation_2d(start_b, end_b, end_a, epsilon)

    if orient_1 != orient_2 and orient_3 != orient_4:
        return True
    if orient_1 == 0 and _on_segment_2d(start_a, start_b, end_a, epsilon):
        return True
    if orient_2 == 0 and _on_segment_2d(start_a, end_b, end_a, epsilon):
        return True
    if orient_3 == 0 and _on_segment_2d(start_b, start_a, end_b, epsilon):
        return True
    if orient_4 == 0 and _on_segment_2d(start_b, end_a, end_b, epsilon):
        return True
    return False


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, epsilon: float = 1e-9) -> bool:
    orient_1 = _orientation_2d(triangle[0], triangle[1], point, epsilon)
    orient_2 = _orientation_2d(triangle[1], triangle[2], point, epsilon)
    orient_3 = _orientation_2d(triangle[2], triangle[0], point, epsilon)
    has_negative = orient_1 < 0 or orient_2 < 0 or orient_3 < 0
    has_positive = orient_1 > 0 or orient_2 > 0 or orient_3 > 0
    return not (has_negative and has_positive)


def _coplanar_triangles_intersect(triangle_a: np.ndarray, triangle_b: np.ndarray, normal: np.ndarray, epsilon: float = 1e-9) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    projected_a = _project_points_to_2d(triangle_a, drop_axis)
    projected_b = _project_points_to_2d(triangle_b, drop_axis)
    for edge_index_a in range(3):
        start_a = projected_a[edge_index_a]
        end_a = projected_a[(edge_index_a + 1) % 3]
        for edge_index_b in range(3):
            start_b = projected_b[edge_index_b]
            end_b = projected_b[(edge_index_b + 1) % 3]
            if _segments_intersect_2d(start_a, end_a, start_b, end_b, epsilon):
                return True
    return _point_in_triangle_2d(projected_a[0], projected_b, epsilon) or _point_in_triangle_2d(projected_b[0], projected_a, epsilon)


def _segment_triangle_intersect(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    triangle: np.ndarray,
    epsilon: float = 1e-9,
) -> bool:
    direction = segment_end - segment_start
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    p_vec = np.cross(direction, edge_2)
    determinant = float(np.dot(edge_1, p_vec))
    if abs(determinant) <= epsilon:
        return False
    inverse_determinant = 1.0 / determinant
    t_vec = segment_start - triangle[0]
    u_value = float(np.dot(t_vec, p_vec)) * inverse_determinant
    if u_value < -epsilon or u_value > 1.0 + epsilon:
        return False
    q_vec = np.cross(t_vec, edge_1)
    v_value = float(np.dot(direction, q_vec)) * inverse_determinant
    if v_value < -epsilon or (u_value + v_value) > 1.0 + epsilon:
        return False
    t_value = float(np.dot(edge_2, q_vec)) * inverse_determinant
    return -epsilon <= t_value <= 1.0 + epsilon


def _triangles_intersect(triangle_a: np.ndarray, triangle_b: np.ndarray, epsilon: float = 1e-9) -> bool:
    normal_a = np.cross(triangle_a[1] - triangle_a[0], triangle_a[2] - triangle_a[0])
    normal_b = np.cross(triangle_b[1] - triangle_b[0], triangle_b[2] - triangle_b[0])
    if np.linalg.norm(normal_a) <= epsilon or np.linalg.norm(normal_b) <= epsilon:
        return False

    plane_distances_b = np.dot(triangle_b - triangle_a[0], normal_a)
    plane_distances_a = np.dot(triangle_a - triangle_b[0], normal_b)
    if np.all(plane_distances_b > epsilon) or np.all(plane_distances_b < -epsilon):
        return False
    if np.all(plane_distances_a > epsilon) or np.all(plane_distances_a < -epsilon):
        return False

    normals_cross = np.cross(normal_a, normal_b)
    if np.linalg.norm(normals_cross) <= epsilon and np.all(np.abs(plane_distances_b) <= epsilon):
        return _coplanar_triangles_intersect(triangle_a, triangle_b, normal_a, epsilon)

    for edge_index in range(3):
        if _segment_triangle_intersect(triangle_a[edge_index], triangle_a[(edge_index + 1) % 3], triangle_b, epsilon):
            return True
        if _segment_triangle_intersect(triangle_b[edge_index], triangle_b[(edge_index + 1) % 3], triangle_a, epsilon):
            return True
    return False


def _triangle_bounds_overlap(bounds_a: np.ndarray, bounds_b: np.ndarray, epsilon: float = 1e-9) -> bool:
    return not (
        bounds_a[3] < (bounds_b[0] - epsilon)
        or bounds_b[3] < (bounds_a[0] - epsilon)
        or bounds_a[4] < (bounds_b[1] - epsilon)
        or bounds_b[4] < (bounds_a[1] - epsilon)
        or bounds_a[5] < (bounds_b[2] - epsilon)
        or bounds_b[5] < (bounds_a[2] - epsilon)
    )


def _find_intersecting_face_indices(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    stop_at_first: bool = False,
) -> tuple[set[int], set[int]]:
    if not _bounds_overlap(mesh_a, mesh_b):
        return set(), set()

    triangles_a = mesh_a.triangles
    triangles_b = mesh_b.triangles
    bounds_a = np.hstack((triangles_a.min(axis=1), triangles_a.max(axis=1)))
    bounds_b = np.hstack((triangles_b.min(axis=1), triangles_b.max(axis=1)))

    intersecting_a: set[int] = set()
    intersecting_b: set[int] = set()
    for index_a, triangle_a in enumerate(triangles_a):
        candidate_mask = np.array([_triangle_bounds_overlap(bounds_a[index_a], bound_b) for bound_b in bounds_b], dtype=bool)
        candidate_indices = np.nonzero(candidate_mask)[0]
        for index_b in candidate_indices:
            if _triangles_intersect(triangle_a, triangles_b[index_b]):
                intersecting_a.add(index_a)
                intersecting_b.add(index_b)
                if stop_at_first:
                    return intersecting_a, intersecting_b
    return intersecting_a, intersecting_b


def _mesh_from_face_indices(mesh: trimesh.Trimesh, face_indices: set[int]) -> trimesh.Trimesh | None:
    if not face_indices:
        return None
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[list(face_indices)] = True
    try:
        submesh = mesh.submesh([mask], append=True, repair=False)
    except Exception:
        return None
    return submesh if isinstance(submesh, trimesh.Trimesh) and len(submesh.faces) > 0 else None


def build_intersection_report(
    named_meshes: list[tuple[str, trimesh.Trimesh | None]],
    subtraction_status: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    valid_meshes = [(name, mesh) for name, mesh in named_meshes if isinstance(mesh, trimesh.Trimesh)]
    ignored_pairs = {
        frozenset({"Encapsulation Base", "Encapsulation Top"}),
    }
    intersections: list[dict[str, str]] = []
    for first_index in range(len(valid_meshes)):
        first_name, first_mesh = valid_meshes[first_index]
        for second_index in range(first_index + 1, len(valid_meshes)):
            second_name, second_mesh = valid_meshes[second_index]
            if frozenset({first_name, second_name}) in ignored_pairs:
                continue
            face_indices_a, face_indices_b = _find_intersecting_face_indices(first_mesh, second_mesh, stop_at_first=False)
            if face_indices_a and face_indices_b:
                extra_details: list[str] = [
                    "method: triangle",
                    f"faces: {len(face_indices_a)} vs {len(face_indices_b)}",
                ]
                if subtraction_status:
                    if first_name == "Encapsulation Top":
                        extra_details.append(f"top subtract {second_name}: {subtraction_status.get('top', {}).get(second_name, 'n/a')}")
                    elif second_name == "Encapsulation Top":
                        extra_details.append(f"top subtract {first_name}: {subtraction_status.get('top', {}).get(first_name, 'n/a')}")
                    if first_name == "Encapsulation Base":
                        extra_details.append(f"base subtract {second_name}: {subtraction_status.get('base', {}).get(second_name, 'n/a')}")
                    elif second_name == "Encapsulation Base":
                        extra_details.append(f"base subtract {first_name}: {subtraction_status.get('base', {}).get(first_name, 'n/a')}")
                intersections.append(
                    {
                        "pair": f"{first_name} intersects {second_name}",
                        "details": "; ".join(extra_details),
                    }
                )
    return intersections


def build_step16_separated_meshes(
    payload: dict,
    named_meshes: list[tuple[str, trimesh.Trimesh | None]],
) -> list[tuple[str, trimesh.Trimesh | None]]:
    return named_meshes


def _extract_faces_inside_other(
    source_mesh: trimesh.Trimesh,
    other_mesh: trimesh.Trimesh,
    tolerance_mm: float = 1e-6,
) -> trimesh.Trimesh | None:
    try:
        signed = trimesh.proximity.signed_distance(other_mesh, source_mesh.triangles_center)
    except Exception:
        return None
    if signed is None or len(signed) != len(source_mesh.faces):
        return None
    face_mask = np.asarray(signed) >= (-tolerance_mm)
    if not np.any(face_mask):
        return None
    try:
        return source_mesh.submesh([face_mask], append=True, repair=False)
    except Exception:
        return None


def build_focus_intersection_meshes(
    named_mesh_map: dict[str, trimesh.Trimesh],
    selected_pairs: set[str],
) -> list[dict[str, object]]:
    focus_meshes: list[dict[str, object]] = []
    for pair_label in selected_pairs:
        if " intersects " not in pair_label:
            continue
        left_name, right_name = [item.strip() for item in pair_label.split(" intersects ", 1)]
        left_mesh = named_mesh_map.get(left_name)
        right_mesh = named_mesh_map.get(right_name)
        if not isinstance(left_mesh, trimesh.Trimesh) or not isinstance(right_mesh, trimesh.Trimesh):
            continue

        left_face_indices, right_face_indices = _find_intersecting_face_indices(left_mesh, right_mesh, stop_at_first=False)
        left_focus_mesh = _mesh_from_face_indices(left_mesh, left_face_indices)
        right_focus_mesh = _mesh_from_face_indices(right_mesh, right_face_indices)
        focus_parts = [
            mesh
            for mesh in (left_focus_mesh, right_focus_mesh)
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0
        ]
        if not focus_parts:
            continue
        intersection_mesh = trimesh.util.concatenate(focus_parts)
        focus_meshes.append(
            {
                "pair": pair_label,
                "mesh": intersection_mesh,
            }
        )
    return focus_meshes


def build_scene_geometry(payload: dict) -> dict:
    current_step_index = int(payload.get("current_step_index", 0))
    body_mesh, side_meshes = build_ic_meshes(payload)
    combined_lead_system_mesh, sketched_leadframe_meshes = build_combined_lead_system_mesh(payload, side_meshes)
    die_leadframe_mesh, die_info = build_die_leadframe_mesh(payload, side_meshes)
    silicon_die_mesh, silicon_die_info = build_silicon_die_mesh(payload, die_leadframe_mesh, die_info)
    leg_pick_markers = build_leg_pick_markers(payload, side_meshes)
    wire_target_markers = build_leadframe_path_end_markers(payload)
    die_regions, die_pick_marker, die_region_info = build_die_region_meshes_and_pick(payload, silicon_die_mesh, silicon_die_info)
    ball_bond_meshes = build_ball_bond_meshes(payload, leg_pick_markers, die_regions)
    connection_paths = build_connection_paths(payload, leg_pick_markers, die_regions, ball_bond_meshes)
    tube_connection_meshes = build_tube_connection_meshes(payload, connection_paths)
    integrated_terminal_meshes = build_integrated_wire_terminal_meshes(payload, connection_paths)
    ball_bond_wire_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, tube_connection_meshes)
    ball_bond_terminal_meshes = build_ball_bond_wire_assembly_meshes(ball_bond_meshes, integrated_terminal_meshes)
    selected_bond_assemblies = build_selected_bond_assemblies(
        current_step_index,
        ball_bond_meshes,
        ball_bond_wire_meshes,
        ball_bond_terminal_meshes,
    )
    encapsulation_meshes = {"base": None, "top": None}
    encapsulation_warnings: list[str] = []
    encapsulation_subtraction_status = {"base": {}, "top": {}}
    if current_step_index >= 14:
        encapsulation_meshes, encapsulation_warnings, encapsulation_subtraction_status = build_encapsulation_meshes(
            payload,
            body_mesh,
            side_meshes,
            sketched_leadframe_meshes,
            die_leadframe_mesh,
            silicon_die_mesh,
            ball_bond_terminal_meshes,
        )
    named_intersection_meshes = [
        ("Lead System", combined_lead_system_mesh),
        ("Die Leadframe", die_leadframe_mesh),
        ("Silicon Die", silicon_die_mesh),
        (
            "Bond Assembly",
            trimesh.util.concatenate([item["mesh"] for item in ball_bond_terminal_meshes])
            if ball_bond_terminal_meshes
            else None,
        ),
        ("Encapsulation Base", encapsulation_meshes.get("base")),
        ("Encapsulation Top", encapsulation_meshes.get("top")),
    ]
    step16_named_meshes = build_step16_separated_meshes(payload, named_intersection_meshes) if current_step_index >= 15 else named_intersection_meshes
    intersections = build_intersection_report(step16_named_meshes, encapsulation_subtraction_status) if current_step_index >= 15 else []
    return {
        "current_step_index": current_step_index,
        "body_mesh": body_mesh,
        "side_meshes": side_meshes,
        "combined_lead_system_mesh": combined_lead_system_mesh,
        "sketched_leadframe_meshes": sketched_leadframe_meshes,
        "die_leadframe_mesh": die_leadframe_mesh,
        "die_info": die_info,
        "silicon_die_mesh": silicon_die_mesh,
        "silicon_die_info": silicon_die_info,
        "leg_pick_markers": leg_pick_markers,
        "wire_target_markers": wire_target_markers,
        "die_regions": die_regions,
        "die_pick_marker": die_pick_marker,
        "die_region_info": die_region_info,
        "ball_bond_meshes": ball_bond_meshes,
        "connection_paths": connection_paths,
        "tube_connection_meshes": tube_connection_meshes,
        "integrated_terminal_meshes": integrated_terminal_meshes,
        "ball_bond_wire_meshes": ball_bond_wire_meshes,
        "ball_bond_terminal_meshes": ball_bond_terminal_meshes,
        "selected_bond_assemblies": selected_bond_assemblies,
        "encapsulation_meshes": encapsulation_meshes,
        "encapsulation_warnings": encapsulation_warnings,
        "encapsulation_subtraction_status": encapsulation_subtraction_status,
        "step16_named_meshes": step16_named_meshes,
        "intersection_pairs": intersections,
    }


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

    geometry = build_scene_geometry(payload)
    body_mesh = geometry["body_mesh"]
    side_meshes = geometry["side_meshes"]
    combined_lead_system_mesh = geometry["combined_lead_system_mesh"]
    die_leadframe_mesh = geometry["die_leadframe_mesh"]
    silicon_die_mesh = geometry["silicon_die_mesh"]
    selected_bond_assemblies = geometry["selected_bond_assemblies"]
    encapsulation_meshes = geometry["encapsulation_meshes"]

    if current_step_index >= 5 and body_mesh is not None:
        meshes.append(encapsulation_meshes["base"] if current_step_index >= 14 and encapsulation_meshes.get("base") is not None else body_mesh)
    if combined_lead_system_mesh is not None:
        meshes.append(combined_lead_system_mesh)
    if current_step_index >= 5 and die_leadframe_mesh is not None:
        meshes.append(die_leadframe_mesh)
    if current_step_index >= 7 and silicon_die_mesh is not None:
        meshes.append(silicon_die_mesh)
    if current_step_index >= 11:
        meshes.extend(item["mesh"] for item in selected_bond_assemblies)
    if current_step_index >= 14 and encapsulation_meshes.get("top") is not None:
        meshes.append(encapsulation_meshes["top"])

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
            tuple(
                (
                    tuple(tuple(point) for point in profile.get("points_mm", [])),
                    bool(profile.get("closed", False)),
                )
                for profile in payload.get("leadframe_profiles", [])
                if isinstance(profile, dict)
            ),
            payload.get("leg_length_mm", 0.0),
            payload.get("lead_offset_mm", 0.0),
            payload.get("die_leadframe_width_mm", 0.0),
            payload.get("die_leadframe_depth_mm", 0.0),
            payload.get("die_leadframe_thickness_mm", 0.08),
            payload.get("leadframe_path_width_mm", 0.3),
            payload.get("leadframe_path_thickness_mm", 1.0),
            payload.get("die_leadframe_center_mode", "region_centroid"),
            payload.get("die_leadframe_center_x_mm", 0.0),
            payload.get("die_leadframe_center_y_mm", 0.0),
            payload.get("silicon_die_width_mm", 0.0),
            payload.get("silicon_die_depth_mm", 0.0),
            payload.get("silicon_die_thickness_mm", 0.12),
            payload.get("leg_pick_distance_mm", 0.2),
            payload.get("leg_pick_marker_size_mm", 0.08),
            payload.get("die_region_span_percent", 70.0),
            payload.get("die_region_depth_mm", 0.15),
            payload.get("die_region_offset_mm", 0.05),
            payload.get("die_region_top_count", 4),
            payload.get("die_region_bottom_count", 4),
            payload.get("die_region_left_count", 0),
            payload.get("die_region_right_count", 0),
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
            payload.get("encapsulation_height_mm", 1.6),
            payload.get("simulation_clearance_mm", 0.001),
            payload.get("step16_split_gap_mm", 0.01),
            payload.get("show_step16_lead_system", True),
            payload.get("show_step16_die_leadframe", True),
            payload.get("show_step16_silicon_die", True),
            payload.get("show_step16_bond_assembly", True),
            payload.get("show_step16_encapsulation_base", True),
            payload.get("show_step16_encapsulation_top", True),
            payload.get("step16_lead_system_color", "#c58a34"),
            payload.get("step16_die_leadframe_color", "#8e3f2b"),
            payload.get("step16_silicon_die_color", "#232323"),
            payload.get("step16_bond_assembly_color", "#2563eb"),
            payload.get("step16_encapsulation_base_color", DEFAULT_BODY_COLOR),
            payload.get("step16_encapsulation_top_color", "#6f7b83"),
            tuple(str(item) for item in payload.get("selected_intersection_pairs", [])),
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
                geometry = build_scene_geometry(payload)
                body_mesh = geometry["body_mesh"]
                side_meshes = geometry["side_meshes"]
                combined_lead_system_mesh = geometry["combined_lead_system_mesh"]
                sketched_leadframe_meshes = geometry["sketched_leadframe_meshes"]
                body_plane_mesh = build_body_plane_mesh(payload)
                die_leadframe_mesh = geometry["die_leadframe_mesh"]
                die_info = geometry["die_info"]
                silicon_die_mesh = geometry["silicon_die_mesh"]
                silicon_die_info = geometry["silicon_die_info"]
                wire_target_markers = geometry["wire_target_markers"]
                die_regions = geometry["die_regions"]
                die_pick_marker = geometry["die_pick_marker"]
                die_region_info = geometry["die_region_info"]
                ball_bond_meshes = geometry["ball_bond_meshes"]
                connection_paths = geometry["connection_paths"]
                ball_bond_wire_meshes = geometry["ball_bond_wire_meshes"]
                ball_bond_terminal_meshes = geometry["ball_bond_terminal_meshes"]
                selected_bond_assemblies = geometry["selected_bond_assemblies"]
                encapsulation_meshes = geometry["encapsulation_meshes"]
                encapsulation_warnings = geometry["encapsulation_warnings"]
                step16_named_meshes = geometry["step16_named_meshes"]
                intersection_pairs = geometry["intersection_pairs"]
                selected_intersection_pairs = {
                    str(item).strip()
                    for item in payload.get("selected_intersection_pairs", [])
                    if str(item).strip()
                }
                step16_visibility = {
                    "Lead System": bool(payload.get("show_step16_lead_system", True)),
                    "Die Leadframe": bool(payload.get("show_step16_die_leadframe", True)),
                    "Silicon Die": bool(payload.get("show_step16_silicon_die", True)),
                    "Bond Assembly": bool(payload.get("show_step16_bond_assembly", True)),
                    "Encapsulation Base": bool(payload.get("show_step16_encapsulation_base", True)),
                    "Encapsulation Top": bool(payload.get("show_step16_encapsulation_top", True)),
                }
                step16_colors = {
                    "Lead System": str(payload.get("step16_lead_system_color", "#c58a34")),
                    "Die Leadframe": str(payload.get("step16_die_leadframe_color", "#8e3f2b")),
                    "Silicon Die": str(payload.get("step16_silicon_die_color", "#232323")),
                    "Bond Assembly": str(payload.get("step16_bond_assembly_color", "#2563eb")),
                    "Encapsulation Base": str(payload.get("step16_encapsulation_base_color", DEFAULT_BODY_COLOR)),
                    "Encapsulation Top": str(payload.get("step16_encapsulation_top_color", "#6f7b83")),
                }
                step16_mesh_map = {name: mesh for name, mesh in step16_named_meshes if isinstance(mesh, trimesh.Trimesh)}
                focus_intersection_meshes = build_focus_intersection_meshes(step16_mesh_map, selected_intersection_pairs) if current_step_index >= 15 else []
                focus_only_mode = current_step_index >= 15 and bool(selected_intersection_pairs)
                focused_component_names: set[str] = set()
                for pair_label in selected_intersection_pairs:
                    if " intersects " not in pair_label:
                        continue
                    left_name, right_name = pair_label.split(" intersects ", 1)
                    focused_component_names.add(left_name.strip())
                    focused_component_names.add(right_name.strip())
                def step16_component_visible(component_name: str) -> bool:
                    if current_step_index < 15:
                        return True
                    if not step16_visibility.get(component_name, True):
                        return False
                    return (not focused_component_names) or (component_name in focused_component_names)
                if self.show_helper_objects:
                    for axis_name, axis_mesh, axis_color in build_axis_meshes(payload):
                        axis_actor = Mesh([axis_mesh.vertices.tolist(), axis_mesh.faces.tolist()]).c(axis_color).alpha(1.0)
                        axis_actor.info = f"{axis_name} Axis"
                        self.actors.append(axis_actor)
                        self.plotter += axis_actor
                if self.show_helper_objects and current_step_index >= 2 and body_plane_mesh is not None and not focus_only_mode:
                    plane_actor = Mesh([body_plane_mesh.vertices.tolist(), body_plane_mesh.faces.tolist()]).c("#d9c9ad").alpha(1.0)
                    plane_actor.info = "Body Placement Plane"
                    self.actors.append(plane_actor)
                    self.plotter += plane_actor
                visible_body_mesh = encapsulation_meshes["base"] if current_step_index >= 14 and encapsulation_meshes.get("base") is not None else body_mesh
                if current_step_index >= 15:
                    visible_body_mesh = step16_mesh_map.get("Encapsulation Base", visible_body_mesh)
                if current_step_index >= 5 and visible_body_mesh is not None and not focus_only_mode:
                    if step16_component_visible("Encapsulation Base"):
                        body_actor = Mesh([visible_body_mesh.vertices.tolist(), visible_body_mesh.faces.tolist()]).c(step16_colors["Encapsulation Base"] if current_step_index >= 15 else DEFAULT_BODY_COLOR).alpha(1.0)
                        body_actor.info = "Encapsulation Base" if current_step_index >= 14 else "Package Body"
                        self.actors.append(body_actor)
                        self.plotter += body_actor
                if combined_lead_system_mesh is not None and not focus_only_mode:
                    visible_lead_system_mesh = step16_mesh_map.get("Lead System", combined_lead_system_mesh) if current_step_index >= 15 else combined_lead_system_mesh
                    if step16_component_visible("Lead System"):
                        lead_system_actor = Mesh([visible_lead_system_mesh.vertices.tolist(), visible_lead_system_mesh.faces.tolist()]).c(step16_colors["Lead System"] if current_step_index >= 15 else "#c58a34").alpha(1.0)
                        lead_system_actor.info = "Lead System"
                        self.actors.append(lead_system_actor)
                        self.plotter += lead_system_actor
                body_status = "ready" if (current_step_index >= 5 and body_mesh is not None) else "hidden until final placement"
                plane_status = "xy placement plane shown" if body_plane_mesh is not None else "xy placement plane waiting"
                if current_step_index >= 14 and encapsulation_meshes.get("top") is not None and not focus_only_mode:
                    top_mesh = step16_mesh_map.get("Encapsulation Top", encapsulation_meshes["top"]) if current_step_index >= 15 else encapsulation_meshes["top"]
                    if step16_component_visible("Encapsulation Top"):
                        top_actor = Mesh([top_mesh.vertices.tolist(), top_mesh.faces.tolist()]).c(step16_colors["Encapsulation Top"] if current_step_index >= 15 else "#6f7b83").alpha(1.0)
                        top_actor.info = "Encapsulation Top"
                        self.actors.append(top_actor)
                        self.plotter += top_actor
                if current_step_index >= 5 and die_leadframe_mesh is not None and not focus_only_mode:
                    visible_die_leadframe_mesh = step16_mesh_map.get("Die Leadframe", die_leadframe_mesh) if current_step_index >= 15 else die_leadframe_mesh
                    if step16_component_visible("Die Leadframe"):
                        die_actor = Mesh([visible_die_leadframe_mesh.vertices.tolist(), visible_die_leadframe_mesh.faces.tolist()]).c(step16_colors["Die Leadframe"] if current_step_index >= 15 else "#8e3f2b").alpha(1.0)
                        die_actor.info = "Leadframe"
                        self.actors.append(die_actor)
                        self.plotter += die_actor
                if current_step_index >= 7 and silicon_die_mesh is not None and not focus_only_mode:
                    visible_silicon_die_mesh = step16_mesh_map.get("Silicon Die", silicon_die_mesh) if current_step_index >= 15 else silicon_die_mesh
                    if step16_component_visible("Silicon Die"):
                        silicon_actor = Mesh([visible_silicon_die_mesh.vertices.tolist(), visible_silicon_die_mesh.faces.tolist()]).c(step16_colors["Silicon Die"] if current_step_index >= 15 else "#232323").alpha(1.0)
                        silicon_actor.info = "Silicon Die"
                        self.actors.append(silicon_actor)
                        self.plotter += silicon_actor
                if self.show_helper_objects and current_step_index >= 8 and not focus_only_mode:
                    for target_index, marker_data in enumerate(wire_target_markers, start=1):
                        marker_mesh = marker_data.get("mesh")
                        if marker_mesh is None:
                            continue
                        marker_actor = Mesh([marker_mesh.vertices.tolist(), marker_mesh.faces.tolist()]).c("#da4f2f").alpha(1.0)
                        marker_actor.info = f"Wire Target {target_index}\nSide: {marker_data['side_name']}"
                        self.actors.append(marker_actor)
                        self.plotter += marker_actor
                if self.show_helper_objects and current_step_index >= 9 and not focus_only_mode:
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
                if self.show_helper_objects and 10 <= current_step_index < 12 and not focus_only_mode:
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
                if focus_only_mode:
                    focus_palette = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#8b5cf6"]
                    for focus_index, focus_data in enumerate(focus_intersection_meshes, start=1):
                        focus_mesh = focus_data["mesh"]
                        focus_color = focus_palette[(focus_index - 1) % len(focus_palette)]
                        focus_actor = Mesh([focus_mesh.vertices.tolist(), focus_mesh.faces.tolist()]).c(focus_color).alpha(1.0)
                        focus_actor.info = f"Intersection Triangles\n{focus_data['pair']}"
                        self.actors.append(focus_actor)
                        self.plotter += focus_actor
                elif current_step_index >= 11:
                    if current_step_index >= 15:
                        if step16_component_visible("Bond Assembly") and step16_mesh_map.get("Bond Assembly") is not None:
                            assembly_mesh = step16_mesh_map["Bond Assembly"]
                            assembly_actor = Mesh([assembly_mesh.vertices.tolist(), assembly_mesh.faces.tolist()]).c(step16_colors["Bond Assembly"]).alpha(1.0)
                            assembly_actor.info = "Bond Assembly"
                            self.actors.append(assembly_actor)
                            self.plotter += assembly_actor
                    else:
                        assembly_source = selected_bond_assemblies
                        assembly_label = (
                            "Integrated Ball Bond Terminal"
                            if current_step_index >= 13
                            else "Ball Bond + Wire"
                            if current_step_index >= 12
                            else "Ball Bond"
                        )
                        assembly_color = "#2563eb" if current_step_index >= 12 else "#f59e0b"
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
                    if current_step_index >= 7 and silicon_die_mesh is not None
                    else "silicon die waiting"
                )
                leg_pick_status = f"wire targets {len(wire_target_markers)}" if current_step_index >= 8 and wire_target_markers else "wire targets waiting"
                die_region_status = (
                    f"die regions {die_region_info.get('span_percent', 0.0):.0f}% span"
                    if current_step_index >= 9 and die_regions
                    else "die regions waiting"
                )
                arc_status = f"arcs {len(connection_paths)}" if current_step_index >= 10 and connection_paths else "arcs waiting"
                bond_status = f"ball bonds {len(ball_bond_meshes)}" if current_step_index >= 11 and ball_bond_meshes else "ball bonds waiting"
                tube_status = f"bond-wire assemblies {len(ball_bond_wire_meshes)}" if current_step_index >= 12 and ball_bond_wire_meshes else "bond-wire assemblies waiting"
                wedge_status = f"integrated terminals {len(ball_bond_terminal_meshes)}" if current_step_index >= 13 and ball_bond_terminal_meshes else "integrated terminals waiting"
                encapsulation_status = (
                    f"ready ({len(encapsulation_warnings)} warning(s))"
                    if current_step_index >= 14 and encapsulation_meshes.get("top") is not None
                    else "waiting"
                )
                intersection_status = (
                    "no intersections"
                    if current_step_index >= 15 and not intersection_pairs
                    else f"{len(intersection_pairs)} intersecting pair(s)"
                    if current_step_index >= 15
                    else "waiting"
                )
                focus_status = (
                    f"{len(selected_intersection_pairs)} focus pair(s), {len(focus_intersection_meshes)} intersection mesh(es)"
                    if current_step_index >= 15 and selected_intersection_pairs
                    else "all visible pairs"
                    if current_step_index >= 15
                    else "n/a"
                )
                summary = (
                    f"Lead System: {len(side_meshes)} leads + {len(sketched_leadframe_meshes)} path mesh(es)  Body: {body_status}  Plane: {plane_status}  "
                    f"Leadframe: {die_status}  Die: {silicon_status}  Leg Picks: {leg_pick_status}  Regions: {die_region_status}  Arcs: {arc_status}  "
                    f"Bonds: {bond_status}  Tubes: {tube_status}  Wedges: {wedge_status}  Encapsulation: {encapsulation_status}  Step 16: {intersection_status}  Focus: {focus_status}"
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
        self.leadframe_profiles: list[LeadProfile] = []
        self.leadframe_current_points_px: list[tuple[float, float]] = []
        self.preview_cursor_px: tuple[float, float] | None = None
        self.dragging_vertex_index: int | tuple[int, int] | None = None
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
            "4. Die Leadframe",
            "5. Lead Frame Designing",
            "6. Overall 3D Placement",
            "7. Lead Offset",
            "8. Silicon Die",
            "9. Leg Positions",
            "10. Die Regions",
            "11. Bond Arcs",
            "12. Ball Bond Formation",
            "13. Bond Wire Tube",
            "14. Wedge Bond Ending",
            "15. Encapsulation",
            "16. Intersection Check",
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
        self.die_leadframe_width_var = tk.DoubleVar(value=8.0)
        self.die_leadframe_depth_var = tk.DoubleVar(value=8.0)
        self.die_leadframe_thickness_var = tk.DoubleVar(value=0.08)
        self.leadframe_path_width_var = tk.DoubleVar(value=0.3)
        self.leadframe_path_thickness_var = tk.DoubleVar(value=1.0)
        self.die_leadframe_center_mode_var = tk.StringVar(value="region_centroid")
        self.die_leadframe_center_x_var = tk.DoubleVar(value=0.0)
        self.die_leadframe_center_y_var = tk.DoubleVar(value=0.0)
        self.die_leadframe_pick_active = False
        self.silicon_die_width_var = tk.DoubleVar(value=2.5)
        self.silicon_die_depth_var = tk.DoubleVar(value=2.5)
        self.silicon_die_thickness_var = tk.DoubleVar(value=0.12)
        self.leg_pick_distance_var = tk.DoubleVar(value=0.2)
        self.leg_pick_marker_size_var = tk.DoubleVar(value=0.08)
        self.die_region_span_percent_var = tk.DoubleVar(value=70.0)
        self.die_region_depth_var = tk.DoubleVar(value=0.15)
        self.die_region_offset_var = tk.DoubleVar(value=0.05)
        self.die_region_top_count_var = tk.IntVar(value=4)
        self.die_region_bottom_count_var = tk.IntVar(value=4)
        self.die_region_left_count_var = tk.IntVar(value=0)
        self.die_region_right_count_var = tk.IntVar(value=0)
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
        self.encapsulation_height_var = tk.DoubleVar(value=1.6)
        self.simulation_clearance_var = tk.DoubleVar(value=0.001)
        self.step16_split_gap_var = tk.DoubleVar(value=0.01)
        self.show_step16_lead_system_var = tk.BooleanVar(value=True)
        self.show_step16_die_leadframe_var = tk.BooleanVar(value=True)
        self.show_step16_silicon_die_var = tk.BooleanVar(value=True)
        self.show_step16_bond_assembly_var = tk.BooleanVar(value=True)
        self.show_step16_encapsulation_base_var = tk.BooleanVar(value=True)
        self.show_step16_encapsulation_top_var = tk.BooleanVar(value=True)
        self.step16_lead_system_color_var = tk.StringVar(value="#c58a34")
        self.step16_die_leadframe_color_var = tk.StringVar(value="#8e3f2b")
        self.step16_silicon_die_color_var = tk.StringVar(value="#232323")
        self.step16_bond_assembly_color_var = tk.StringVar(value="#2563eb")
        self.step16_encapsulation_base_color_var = tk.StringVar(value=DEFAULT_BODY_COLOR)
        self.step16_encapsulation_top_color_var = tk.StringVar(value="#6f7b83")
        self.intersection_report_var = tk.StringVar(value="Step 16 will list any remaining intersecting parts.")
        self.intersection_pair_vars: dict[str, tk.BooleanVar] = {}
        self.intersection_pairs_container: ttk.Frame | None = None
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
        self.step_frames.append(self._build_die_leadframe_step(left))
        self.step_frames.append(self._build_leadframe_design_step(left))
        self.step_frames.append(self._build_overall_step(left))
        self.step_frames.append(self._build_lead_offset_step(left))
        self.step_frames.append(self._build_silicon_die_step(left))
        self.step_frames.append(self._build_leg_positions_step(left))
        self.step_frames.append(self._build_die_regions_step(left))
        self.step_frames.append(self._build_bond_arcs_step(left))
        self.step_frames.append(self._build_ball_bond_step(left))
        self.step_frames.append(self._build_bond_wire_tube_step(left))
        self.step_frames.append(self._build_wedge_bond_step(left))
        self.step_frames.append(self._build_encapsulation_step(left))
        self.step_frames.append(self._build_intersection_check_step(left))

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
        self.root.bind("<Control-z>", self._on_control_undo)
        self.root.bind("<Control-Z>", self._on_control_undo)
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
            self.die_leadframe_width_var,
            self.die_leadframe_depth_var,
            self.die_leadframe_thickness_var,
            self.leadframe_path_width_var,
            self.leadframe_path_thickness_var,
            self.die_leadframe_center_mode_var,
            self.die_leadframe_center_x_var,
            self.die_leadframe_center_y_var,
            self.silicon_die_width_var,
            self.silicon_die_depth_var,
            self.silicon_die_thickness_var,
            self.leg_pick_distance_var,
            self.leg_pick_marker_size_var,
            self.die_region_span_percent_var,
            self.die_region_depth_var,
            self.die_region_offset_var,
            self.die_region_top_count_var,
            self.die_region_bottom_count_var,
            self.die_region_left_count_var,
            self.die_region_right_count_var,
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
            self.encapsulation_height_var,
            self.simulation_clearance_var,
            self.step16_split_gap_var,
            self.show_step16_lead_system_var,
            self.show_step16_die_leadframe_var,
            self.show_step16_silicon_die_var,
            self.show_step16_bond_assembly_var,
            self.show_step16_encapsulation_base_var,
            self.show_step16_encapsulation_top_var,
            self.step16_lead_system_color_var,
            self.step16_die_leadframe_color_var,
            self.step16_silicon_die_color_var,
            self.step16_bond_assembly_color_var,
            self.step16_encapsulation_base_color_var,
            self.step16_encapsulation_top_color_var,
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

    def _is_leadframe_canvas_context(self) -> bool:
        return self.step_index in {3, 4}

    def _is_leadframe_design_step(self) -> bool:
        return self.step_index == 4

    def _canvas_step_active(self) -> bool:
        return self.step_index in {0, 3, 4}

    def _active_profile(self) -> LeadProfile:
        return self.profile

    def _active_draft_points(self) -> list[tuple[float, float]]:
        return self.leadframe_current_points_px if self._is_leadframe_canvas_context() else self.current_points_px

    def _leadframe_shapes(self) -> list[LeadProfile]:
        return self.leadframe_profiles

    def _replace_active_profile(self, profile: LeadProfile) -> None:
        self.profile = profile

    def _clear_active_draft_points(self) -> None:
        if self._is_leadframe_canvas_context():
            self.leadframe_current_points_px.clear()
        else:
            self.current_points_px.clear()

    def _canvas_step_title(self) -> str:
        if self.step_index == 3:
            return "Die Leadframe Placement"
        if self._is_leadframe_canvas_context():
            return "Lead Frame Sketch"
        return "Lead Profile Sketch"

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
        payload = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
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
            self.leadframe_profiles = _leadframe_profiles_from_payload(payload)
            self.current_points_px = [
                tuple(point)
                for point in payload.get("draft_points_px", [])
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            self.leadframe_current_points_px = [
                tuple(point)
                for point in payload.get("leadframe_draft_points_px", [])
                if isinstance(point, list | tuple) and len(point) == 2
            ]
            body_width_value = _coerce_float(payload.get("body_width_mm", 10.0), 10.0)
            body_depth_value = _coerce_float(payload.get("body_depth_mm", 10.0), 10.0)
            self.distance_target_var.set(_coerce_float(payload.get("distance_target_mm", 1.0), 1.0))
            self.lead_offset_var.set(_coerce_float(payload.get("lead_offset_mm", 0.0), 0.0))
            legacy_ratio_percent = _coerce_float(payload.get("die_leadframe_ratio_percent", 80.0), 80.0)
            self.die_leadframe_width_var.set(
                _coerce_float(payload.get("die_leadframe_width_mm", body_width_value * (legacy_ratio_percent / 100.0)), body_width_value * 0.8)
            )
            self.die_leadframe_depth_var.set(
                _coerce_float(payload.get("die_leadframe_depth_mm", body_depth_value * (legacy_ratio_percent / 100.0)), body_depth_value * 0.8)
            )
            self.die_leadframe_thickness_var.set(_coerce_float(payload.get("die_leadframe_thickness_mm", 0.08), 0.08))
            self.leadframe_path_width_var.set(_coerce_float(payload.get("leadframe_path_width_mm", 0.3), 0.3))
            self.leadframe_path_thickness_var.set(_coerce_float(payload.get("leadframe_path_thickness_mm", 1.0), 1.0))
            center_mode = str(payload.get("die_leadframe_center_mode", "region_centroid")).strip().lower()
            self.die_leadframe_center_mode_var.set(center_mode if center_mode in {"region_centroid", "custom_point"} else "region_centroid")
            self.die_leadframe_center_x_var.set(_coerce_float(payload.get("die_leadframe_center_x_mm", 0.0), 0.0))
            self.die_leadframe_center_y_var.set(_coerce_float(payload.get("die_leadframe_center_y_mm", 0.0), 0.0))
            self.silicon_die_width_var.set(_coerce_float(payload.get("silicon_die_width_mm", 2.5), 2.5))
            self.silicon_die_depth_var.set(_coerce_float(payload.get("silicon_die_depth_mm", 2.5), 2.5))
            self.silicon_die_thickness_var.set(_coerce_float(payload.get("silicon_die_thickness_mm", 0.12), 0.12))
            self.leg_pick_distance_var.set(_coerce_float(payload.get("leg_pick_distance_mm", 0.2), 0.2))
            self.leg_pick_marker_size_var.set(_coerce_float(payload.get("leg_pick_marker_size_mm", 0.08), 0.08))
            self.die_region_span_percent_var.set(_coerce_float(payload.get("die_region_span_percent", 70.0), 70.0))
            self.die_region_depth_var.set(_coerce_float(payload.get("die_region_depth_mm", 0.15), 0.15))
            self.die_region_offset_var.set(_coerce_float(payload.get("die_region_offset_mm", 0.05), 0.05))
            self.die_region_top_count_var.set(_coerce_int(payload.get("die_region_top_count", payload.get("top_count", 4)), 4))
            self.die_region_bottom_count_var.set(_coerce_int(payload.get("die_region_bottom_count", payload.get("bottom_count", 4)), 4))
            self.die_region_left_count_var.set(_coerce_int(payload.get("die_region_left_count", payload.get("left_count", 0)), 0))
            self.die_region_right_count_var.set(_coerce_int(payload.get("die_region_right_count", payload.get("right_count", 0)), 0))
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
            self.encapsulation_height_var.set(_coerce_float(payload.get("encapsulation_height_mm", self.body_height_var.get()), self.body_height_var.get()))
            self.simulation_clearance_var.set(_coerce_float(payload.get("simulation_clearance_mm", 0.001), 0.001))
            self.step16_split_gap_var.set(_coerce_float(payload.get("step16_split_gap_mm", 0.01), 0.01))
            self.show_step16_lead_system_var.set(bool(payload.get("show_step16_lead_system", True)))
            self.show_step16_die_leadframe_var.set(bool(payload.get("show_step16_die_leadframe", True)))
            self.show_step16_silicon_die_var.set(bool(payload.get("show_step16_silicon_die", True)))
            self.show_step16_bond_assembly_var.set(bool(payload.get("show_step16_bond_assembly", True)))
            self.show_step16_encapsulation_base_var.set(bool(payload.get("show_step16_encapsulation_base", True)))
            self.show_step16_encapsulation_top_var.set(bool(payload.get("show_step16_encapsulation_top", True)))
            self.step16_lead_system_color_var.set(str(payload.get("step16_lead_system_color", "#c58a34")))
            self.step16_die_leadframe_color_var.set(str(payload.get("step16_die_leadframe_color", "#8e3f2b")))
            self.step16_silicon_die_color_var.set(str(payload.get("step16_silicon_die_color", "#232323")))
            self.step16_bond_assembly_color_var.set(str(payload.get("step16_bond_assembly_color", "#2563eb")))
            self.step16_encapsulation_base_color_var.set(str(payload.get("step16_encapsulation_base_color", DEFAULT_BODY_COLOR)))
            self.step16_encapsulation_top_color_var.set(str(payload.get("step16_encapsulation_top_color", "#6f7b83")))
            selected_intersection_pairs = [
                str(item).strip()
                for item in payload.get("selected_intersection_pairs", [])
                if str(item).strip()
            ]
            self._refresh_intersection_pair_checkboxes([], selected_intersection_pairs)
            self.distance_pick_active = bool(payload.get("distance_pick_active", False))
            self.distance_point_indices = [
                _coerce_int(index, -1)
                for index in payload.get("distance_point_indices", [])
                if 0 <= _coerce_int(index, -1) < len(self.profile.points_mm)
            ]

            self.leg_length_var.set(_coerce_float(payload.get("leg_length_mm", 4.0), 4.0))
            self.body_width_var.set(body_width_value)
            self.body_depth_var.set(body_depth_value)
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
        ttk.Label(frame, text="Lead Frame Layer Thickness (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.leadframe_path_thickness_var).pack(fill="x", pady=(2, 0))
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

    def _build_leadframe_design_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 5: Lead Frame Designing", padding=10)
        ttk.Label(
            frame,
            text="Use the 2D canvas to draw lead-frame paths. Start from the orange leg guides or snap to any point along the die leadframe square edges, then control the metal width in the XY plane here.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Lead Frame Path Width In XY (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.leadframe_path_width_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Undo Point", command=self._undo_point).pack(fill="x", pady=(10, 0))
        ttk.Button(frame, text="Finish Path", command=self._finish_closed_shape).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="Mirror Horizontal", command=self._mirror_leadframe_paths_horizontal).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="Mirror Vertical", command=self._mirror_leadframe_paths_vertical).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="Clear Lead Frame Paths", command=self._clear_profile).pack(fill="x", pady=(8, 0))
        return frame

    def _build_overall_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 6: Overall 3D Placement", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 7: Lead Offset", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 4: Die Leadframe", padding=10)
        ttk.Label(
            frame,
            text="Create the die leadframe rectangle using explicit width and height. Place it either at the region centroid or at a custom 2D point picked on the canvas.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Leadframe Width / Height (mm)").pack(anchor="w", pady=(10, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self.die_leadframe_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(row, textvariable=self.die_leadframe_depth_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(frame, text="Frame Thickness (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_leadframe_thickness_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Center Mode").pack(anchor="w", pady=(10, 0))
        ttk.Combobox(
            frame,
            textvariable=self.die_leadframe_center_mode_var,
            values=("region_centroid", "custom_point"),
            state="readonly",
        ).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Pick Custom Point On Canvas", command=self._toggle_die_leadframe_pick_mode).pack(fill="x", pady=(10, 0))
        ttk.Label(frame, text="Custom Center X / Y (mm)").pack(anchor="w", pady=(10, 0))
        center_row = ttk.Frame(frame)
        center_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(center_row, textvariable=self.die_leadframe_center_x_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(center_row, textvariable=self.die_leadframe_center_y_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_silicon_die_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 8: Silicon Die", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 9: Leg Positions", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 10: Die Regions", padding=10)
        ttk.Label(
            frame,
            text="Build 2D edge regions on the silicon die. Set how many regions each side should have, hide a side by setting it to 0, and offset the bands inward from the die edge.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Region Span Percent").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_span_percent_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Region Band Depth (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_depth_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Region Offset From Die Edge (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.die_region_offset_var).pack(fill="x", pady=(2, 0))
        count_box = ttk.LabelFrame(frame, text="Regions Per Side", padding=8)
        count_box.pack(fill="x", pady=(10, 0))
        ttk.Label(count_box, text="Top").pack(anchor="w")
        ttk.Entry(count_box, textvariable=self.die_region_top_count_var).pack(fill="x", pady=(2, 6))
        ttk.Label(count_box, text="Bottom").pack(anchor="w")
        ttk.Entry(count_box, textvariable=self.die_region_bottom_count_var).pack(fill="x", pady=(2, 6))
        ttk.Label(count_box, text="Left").pack(anchor="w")
        ttk.Entry(count_box, textvariable=self.die_region_left_count_var).pack(fill="x", pady=(2, 6))
        ttk.Label(count_box, text="Right").pack(anchor="w")
        ttk.Entry(count_box, textvariable=self.die_region_right_count_var).pack(fill="x", pady=(2, 0))
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
        frame = ttk.LabelFrame(parent, text="Step 11: Bond Arcs", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 12: Ball Bond Formation", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 13: Bond Wire Tube", padding=10)
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
        frame = ttk.LabelFrame(parent, text="Step 14: Wedge Bond Ending", padding=10)
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

    def _build_encapsulation_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 15: Encapsulation", padding=10)
        ttk.Label(
            frame,
            text="Create a top encapsulation cuboid using the same width and depth as the step 6 base. For simulation, the tool enlarges the internal parts, subtracts them from the base and top, and previews the resulting non-intersecting package solids.",
            wraplength=330,
        ).pack(anchor="w")
        ttk.Label(frame, text="Top Encapsulation Height (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.encapsulation_height_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame, text="Simulation Clearance / Enlargement (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.simulation_clearance_var).pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="Update Preview Data", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        return frame

    def _build_intersection_check_step(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Step 16: Intersection Check", padding=10)
        ttk.Label(
            frame,
            text="Check the simulation-ready solids for remaining triangle intersections after the encapsulation subtraction. Any intersecting part pairs will be listed below.",
            wraplength=330,
        ).pack(anchor="w")
        visibility_box = ttk.LabelFrame(frame, text="Visible Components", padding=8)
        visibility_box.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(visibility_box, text="Lead System", variable=self.show_step16_lead_system_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Die Leadframe", variable=self.show_step16_die_leadframe_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Silicon Die", variable=self.show_step16_silicon_die_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Bond Assembly", variable=self.show_step16_bond_assembly_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Encapsulation Base", variable=self.show_step16_encapsulation_base_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Encapsulation Top", variable=self.show_step16_encapsulation_top_var).pack(anchor="w")
        color_box = ttk.LabelFrame(frame, text="Part Colors", padding=8)
        color_box.pack(fill="x", pady=(10, 0))
        self._build_step16_color_row(color_box, "Lead System", self.step16_lead_system_color_var)
        self._build_step16_color_row(color_box, "Die Leadframe", self.step16_die_leadframe_color_var)
        self._build_step16_color_row(color_box, "Silicon Die", self.step16_silicon_die_color_var)
        self._build_step16_color_row(color_box, "Bond Assembly", self.step16_bond_assembly_color_var)
        self._build_step16_color_row(color_box, "Encapsulation Base", self.step16_encapsulation_base_color_var)
        self._build_step16_color_row(color_box, "Encapsulation Top", self.step16_encapsulation_top_color_var)
        pair_box = ttk.LabelFrame(frame, text="Focus Intersections", padding=8)
        pair_box.pack(fill="x", pady=(10, 0))
        self.intersection_pairs_container = ttk.Frame(pair_box)
        self.intersection_pairs_container.pack(fill="x")
        ttk.Label(self.intersection_pairs_container, text="Refresh step 16 to list selectable pairs.", wraplength=300, justify="left").pack(anchor="w")
        ttk.Label(frame, textvariable=self.intersection_report_var, wraplength=330, justify="left").pack(anchor="w", pady=(10, 0))
        ttk.Button(frame, text="Refresh Intersection Report", command=lambda: self._push_payload(launch_if_missing=True)).pack(fill="x", pady=(10, 0))
        ttk.Button(frame, text="Export Combined STL", command=self._export_combined_stl).pack(fill="x", pady=(8, 0))
        return frame

    def _build_step16_color_row(self, parent: ttk.Frame, label_text: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text=label_text).pack(side="left")
        ttk.Button(
            row,
            text="Choose...",
            command=lambda var=variable, title=label_text: self._choose_step16_color(var, title),
        ).pack(side="right")

    def _choose_step16_color(self, variable: tk.StringVar, title: str) -> None:
        initial_color = variable.get().strip() or "#ffffff"
        _rgb, hex_color = colorchooser.askcolor(color=initial_color, title=f"Choose {title} Color", parent=self.root)
        if not hex_color:
            return
        variable.set(hex_color)
        self._push_payload(launch_if_missing=True)

    def _show_step(self) -> None:
        self.step_label.configure(text=self.step_titles[self.step_index])
        for index, frame in enumerate(self.step_frames):
            if index == self.step_index:
                frame.pack(fill="x", pady=(0, 12))
            else:
                frame.pack_forget()
        if self._canvas_step_active():
            self.canvas_title_label.configure(text=self._canvas_step_title())
            self.canvas_title_label.grid()
            self.canvas.grid()
        else:
            self.canvas_title_label.grid_remove()
            self.canvas.grid_remove()

    def _selected_intersection_pairs(self) -> list[str]:
        return [pair_label for pair_label, variable in self.intersection_pair_vars.items() if bool(variable.get())]

    def _refresh_intersection_pair_checkboxes(self, intersections: list[dict[str, str]], selected_pairs: list[str] | None = None) -> None:
        if self.intersection_pairs_container is None:
            return
        selected_set = set(selected_pairs or self._selected_intersection_pairs())
        for child in self.intersection_pairs_container.winfo_children():
            child.destroy()
        self.intersection_pair_vars.clear()
        if not intersections:
            ttk.Label(
                self.intersection_pairs_container,
                text="No intersecting pairs are currently available for focus.",
                wraplength=300,
                justify="left",
            ).pack(anchor="w")
            return
        for item in intersections:
            pair_label = str(item.get("pair", "")).strip()
            if not pair_label:
                continue
            variable = tk.BooleanVar(value=pair_label in selected_set)
            variable.trace_add("write", self._schedule_preview_refresh)
            self.intersection_pair_vars[pair_label] = variable
            ttk.Checkbutton(self.intersection_pairs_container, text=pair_label, variable=variable).pack(anchor="w")

    def _previous_step(self) -> None:
        self.step_index = max(0, self.step_index - 1)
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.")

    def _next_step(self) -> None:
        trim_message = ""
        if self.step_index == 4:
            trimmed_count = self._sanitize_leadframe_paths_against_keepout()
            if trimmed_count > 0:
                trim_message = f" Trimmed {trimmed_count} path(s) against the center keep-out."
        self.step_index = min(len(self.step_titles) - 1, self.step_index + 1)
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.{trim_message}")
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

    def _leadframe_contact_guide_data(self) -> list[dict[str, object]]:
        body_width_mm = max(0.5, float(self.body_width_var.get()))
        body_depth_mm = max(0.5, float(self.body_depth_var.get()))
        guide_half_length_mm = max(0.12, min(body_width_mm, body_depth_mm) * 0.035)
        region_half_thickness_mm = max(0.08, guide_half_length_mm * 0.55)
        guide_data: list[dict[str, object]] = []
        rows = [
            ("Top", self.top_count_var.get(), self.top_pitch_var.get()),
            ("Bottom", self.bottom_count_var.get(), self.bottom_pitch_var.get()),
            ("Left", self.left_count_var.get(), self.left_pitch_var.get()),
            ("Right", self.right_count_var.get(), self.right_pitch_var.get()),
        ]
        for side_name, count, pitch_mm in rows:
            count = max(0, int(count))
            pitch_mm = float(pitch_mm)
            if count <= 0 or pitch_mm <= 0.0:
                continue
            spread = (count - 1) * pitch_mm
            for index in range(count):
                position = (index * pitch_mm) - (spread / 2.0)
                if side_name == "Top":
                    center_mm = (position, body_depth_mm / 2.0)
                    start_mm = (center_mm[0] - guide_half_length_mm, center_mm[1])
                    end_mm = (center_mm[0] + guide_half_length_mm, center_mm[1])
                    region_points_mm = [
                        (start_mm[0], center_mm[1] - region_half_thickness_mm),
                        (end_mm[0], center_mm[1] - region_half_thickness_mm),
                        (end_mm[0], center_mm[1] + region_half_thickness_mm),
                        (start_mm[0], center_mm[1] + region_half_thickness_mm),
                    ]
                elif side_name == "Bottom":
                    center_mm = (position, -body_depth_mm / 2.0)
                    start_mm = (center_mm[0] - guide_half_length_mm, center_mm[1])
                    end_mm = (center_mm[0] + guide_half_length_mm, center_mm[1])
                    region_points_mm = [
                        (start_mm[0], center_mm[1] - region_half_thickness_mm),
                        (end_mm[0], center_mm[1] - region_half_thickness_mm),
                        (end_mm[0], center_mm[1] + region_half_thickness_mm),
                        (start_mm[0], center_mm[1] + region_half_thickness_mm),
                    ]
                elif side_name == "Left":
                    center_mm = (-body_width_mm / 2.0, position)
                    start_mm = (center_mm[0], center_mm[1] - guide_half_length_mm)
                    end_mm = (center_mm[0], center_mm[1] + guide_half_length_mm)
                    region_points_mm = [
                        (center_mm[0] - region_half_thickness_mm, start_mm[1]),
                        (center_mm[0] + region_half_thickness_mm, start_mm[1]),
                        (center_mm[0] + region_half_thickness_mm, end_mm[1]),
                        (center_mm[0] - region_half_thickness_mm, end_mm[1]),
                    ]
                else:
                    center_mm = (body_width_mm / 2.0, position)
                    start_mm = (center_mm[0], center_mm[1] - guide_half_length_mm)
                    end_mm = (center_mm[0], center_mm[1] + guide_half_length_mm)
                    region_points_mm = [
                        (center_mm[0] - region_half_thickness_mm, start_mm[1]),
                        (center_mm[0] + region_half_thickness_mm, start_mm[1]),
                        (center_mm[0] + region_half_thickness_mm, end_mm[1]),
                        (center_mm[0] - region_half_thickness_mm, end_mm[1]),
                    ]
                guide_data.append(
                    {
                        "side_name": side_name,
                        "index": index + 1,
                        "orientation": "horizontal" if side_name in {"Top", "Bottom"} else "vertical",
                        "start_mm": start_mm,
                        "end_mm": end_mm,
                        "center_mm": center_mm,
                        "region_points_mm": region_points_mm,
                    }
                )
        return guide_data

    def _die_leadframe_corner_points_mm(self) -> list[tuple[float, float]]:
        try:
            payload = self._project_payload()
            _body_mesh, side_meshes = build_ic_meshes(payload)
            die_leadframe_mesh, _die_info = build_die_leadframe_mesh(payload, side_meshes)
        except Exception:
            return []
        if die_leadframe_mesh is None:
            return []
        min_corner = die_leadframe_mesh.bounds[0].tolist()
        max_corner = die_leadframe_mesh.bounds[1].tolist()
        return [
            (min_corner[0], min_corner[1]),
            (min_corner[0], max_corner[1]),
            (max_corner[0], max_corner[1]),
            (max_corner[0], min_corner[1]),
        ]

    def _die_leadframe_keepout_corners_mm(self) -> list[tuple[float, float]]:
        die_corners_mm = self._die_leadframe_corner_points_mm()
        if len(die_corners_mm) != 4:
            return []
        min_x = min(point[0] for point in die_corners_mm)
        max_x = max(point[0] for point in die_corners_mm)
        min_y = min(point[1] for point in die_corners_mm)
        max_y = max(point[1] for point in die_corners_mm)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_width = ((max_x - min_x) * LEADFRAME_KEEPOUT_SCALE) / 2.0
        half_height = ((max_y - min_y) * LEADFRAME_KEEPOUT_SCALE) / 2.0
        return [
            (center_x - half_width, center_y - half_height),
            (center_x - half_width, center_y + half_height),
            (center_x + half_width, center_y + half_height),
            (center_x + half_width, center_y - half_height),
        ]

    def _leadframe_reference_anchors(self) -> list[dict[str, object]]:
        anchors: list[dict[str, object]] = []
        for guide in self._leadframe_contact_guide_data():
            anchors.append(
                {
                    "kind": "leg_guide_center",
                    "side_name": guide["side_name"],
                    "index": guide["index"],
                    "role": "center",
                    "point_mm": tuple(guide["center_mm"]),
                }
            )
            anchors.append(
                {
                    "kind": "leg_guide",
                    "side_name": guide["side_name"],
                    "index": guide["index"],
                    "role": "start",
                    "point_mm": tuple(guide["start_mm"]),
                }
            )
            anchors.append(
                {
                    "kind": "leg_guide",
                    "side_name": guide["side_name"],
                    "index": guide["index"],
                    "role": "end",
                    "point_mm": tuple(guide["end_mm"]),
                }
            )
        for corner_index, point_mm in enumerate(self._die_leadframe_corner_points_mm(), start=1):
            anchors.append(
                {
                    "kind": "die_corner",
                    "corner_index": corner_index,
                    "role": "corner",
                    "point_mm": point_mm,
                }
            )
        return anchors

    def _closest_point_on_segment_mm(
        self,
        point_mm: tuple[float, float],
        start_mm: tuple[float, float],
        end_mm: tuple[float, float],
    ) -> tuple[float, float]:
        segment_dx = end_mm[0] - start_mm[0]
        segment_dy = end_mm[1] - start_mm[1]
        segment_length_sq = (segment_dx * segment_dx) + (segment_dy * segment_dy)
        if segment_length_sq <= 1e-12:
            return start_mm
        t_value = (
            ((point_mm[0] - start_mm[0]) * segment_dx) + ((point_mm[1] - start_mm[1]) * segment_dy)
        ) / segment_length_sq
        clamped_t = max(0.0, min(1.0, t_value))
        return (
            start_mm[0] + (segment_dx * clamped_t),
            start_mm[1] + (segment_dy * clamped_t),
        )

    def _find_die_leadframe_edge_hit(self, event_x: float, event_y: float) -> dict[str, object] | None:
        die_corners_mm = self._die_leadframe_corner_points_mm()
        if len(die_corners_mm) != 4:
            return None
        point_mm = self._canvas_to_world((event_x, event_y))
        threshold_px = 8.0
        closest_hit: dict[str, object] | None = None
        closest_distance_px = float("inf")
        edge_pairs = [
            (die_corners_mm[0], die_corners_mm[1], "left"),
            (die_corners_mm[1], die_corners_mm[2], "top"),
            (die_corners_mm[2], die_corners_mm[3], "right"),
            (die_corners_mm[3], die_corners_mm[0], "bottom"),
        ]
        for start_mm, end_mm, edge_name in edge_pairs:
            closest_point_mm = self._closest_point_on_segment_mm(point_mm, start_mm, end_mm)
            closest_point_px = self._world_to_canvas(closest_point_mm)
            distance_px = math.dist(closest_point_px, (event_x, event_y))
            if distance_px <= threshold_px and distance_px < closest_distance_px:
                closest_distance_px = distance_px
                closest_hit = {
                    "kind": "die_edge",
                    "edge_name": edge_name,
                    "role": "edge_point",
                    "point_mm": closest_point_mm,
                    "edge_start_mm": start_mm,
                    "edge_end_mm": end_mm,
                }
        return closest_hit

    def _ray_segment_intersection_mm(
        self,
        ray_start_mm: tuple[float, float],
        ray_direction_mm: tuple[float, float],
        segment_start_mm: tuple[float, float],
        segment_end_mm: tuple[float, float],
    ) -> tuple[float, float] | None:
        rx, ry = ray_direction_mm
        sx = segment_end_mm[0] - segment_start_mm[0]
        sy = segment_end_mm[1] - segment_start_mm[1]
        denominator = (rx * sy) - (ry * sx)
        if abs(denominator) <= 1e-12:
            return None
        dx = segment_start_mm[0] - ray_start_mm[0]
        dy = segment_start_mm[1] - ray_start_mm[1]
        ray_t = ((dx * sy) - (dy * sx)) / denominator
        segment_t = ((dx * ry) - (dy * rx)) / denominator
        if ray_t < 0.0 or segment_t < 0.0 or segment_t > 1.0:
            return None
        return (
            ray_start_mm[0] + (ray_t * rx),
            ray_start_mm[1] + (ray_t * ry),
        )

    def _point_inside_axis_aligned_box(
        self,
        point_mm: tuple[float, float],
        box_corners_mm: list[tuple[float, float]],
    ) -> bool:
        if len(box_corners_mm) != 4:
            return False
        min_x = min(point[0] for point in box_corners_mm)
        max_x = max(point[0] for point in box_corners_mm)
        min_y = min(point[1] for point in box_corners_mm)
        max_y = max(point[1] for point in box_corners_mm)
        return min_x <= point_mm[0] <= max_x and min_y <= point_mm[1] <= max_y

    def _segment_box_intersections_mm(
        self,
        start_mm: tuple[float, float],
        end_mm: tuple[float, float],
        box_corners_mm: list[tuple[float, float]],
    ) -> list[tuple[float, tuple[float, float]]]:
        if len(box_corners_mm) != 4:
            return []
        min_x = min(point[0] for point in box_corners_mm)
        max_x = max(point[0] for point in box_corners_mm)
        min_y = min(point[1] for point in box_corners_mm)
        max_y = max(point[1] for point in box_corners_mm)
        dx = end_mm[0] - start_mm[0]
        dy = end_mm[1] - start_mm[1]
        candidates: list[tuple[float, tuple[float, float]]] = []
        if abs(dx) > 1e-12:
            for edge_x in (min_x, max_x):
                t_value = (edge_x - start_mm[0]) / dx
                if 0.0 <= t_value <= 1.0:
                    y_value = start_mm[1] + (t_value * dy)
                    if min_y - 1e-9 <= y_value <= max_y + 1e-9:
                        candidates.append((t_value, (edge_x, y_value)))
        if abs(dy) > 1e-12:
            for edge_y in (min_y, max_y):
                t_value = (edge_y - start_mm[1]) / dy
                if 0.0 <= t_value <= 1.0:
                    x_value = start_mm[0] + (t_value * dx)
                    if min_x - 1e-9 <= x_value <= max_x + 1e-9:
                        candidates.append((t_value, (x_value, edge_y)))
        deduped: list[tuple[float, tuple[float, float]]] = []
        for t_value, point_mm in sorted(candidates, key=lambda item: item[0]):
            if not deduped or math.dist(point_mm, deduped[-1][1]) > 1e-8:
                deduped.append((t_value, point_mm))
        return deduped

    def _trim_path_against_keepout(
        self,
        path_points_mm: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        keepout_corners_mm = self._die_leadframe_keepout_corners_mm()
        if len(keepout_corners_mm) != 4 or len(path_points_mm) < 2:
            return path_points_mm
        if self._point_inside_axis_aligned_box(path_points_mm[0], keepout_corners_mm):
            return []
        trimmed_points: list[tuple[float, float]] = [path_points_mm[0]]
        for start_mm, end_mm in zip(path_points_mm[:-1], path_points_mm[1:]):
            start_inside = self._point_inside_axis_aligned_box(start_mm, keepout_corners_mm)
            end_inside = self._point_inside_axis_aligned_box(end_mm, keepout_corners_mm)
            intersections = self._segment_box_intersections_mm(start_mm, end_mm, keepout_corners_mm)
            if not start_inside and not end_inside and not intersections:
                trimmed_points.append(end_mm)
                continue
            if not start_inside and (end_inside or intersections):
                if intersections:
                    trimmed_points.append(intersections[0][1])
                return _simplify_path_points(trimmed_points)
            if start_inside:
                return _simplify_path_points(trimmed_points)
        return _simplify_path_points(trimmed_points)

    def _sanitize_leadframe_paths_against_keepout(self) -> int:
        sanitized_profiles: list[LeadProfile] = []
        trimmed_count = 0
        for profile in self.leadframe_profiles:
            if profile.closed:
                sanitized_profiles.append(profile)
                continue
            trimmed_points_mm = self._trim_path_against_keepout(profile.points_mm)
            if len(trimmed_points_mm) >= 2:
                if trimmed_points_mm != profile.points_mm:
                    trimmed_count += 1
                sanitized_profiles.append(LeadProfile(points_mm=trimmed_points_mm, closed=False))
            else:
                trimmed_count += 1
        self.leadframe_profiles = sanitized_profiles
        return trimmed_count

    def _angled_path_snap_to_die_edge(
        self,
        anchor_px: tuple[float, float],
        cursor_px: tuple[float, float],
        edge_hit: dict[str, object],
    ) -> tuple[float, float] | None:
        snapped_path = self._snapped_cursor_for_45_degree_path(anchor_px, cursor_px)
        if snapped_path is None:
            return None
        anchor_mm = self._canvas_to_world(anchor_px)
        snapped_target_mm = self._canvas_to_world(snapped_path[0])
        direction_mm = (
            snapped_target_mm[0] - anchor_mm[0],
            snapped_target_mm[1] - anchor_mm[1],
        )
        if math.hypot(direction_mm[0], direction_mm[1]) <= 1e-9:
            return None
        edge_start_mm = tuple(edge_hit["edge_start_mm"])
        edge_end_mm = tuple(edge_hit["edge_end_mm"])
        intersection_mm = self._ray_segment_intersection_mm(anchor_mm, direction_mm, edge_start_mm, edge_end_mm)
        if intersection_mm is None:
            return None
        return self._world_to_canvas(intersection_mm)

    def _distance_point_to_die_leadframe_edge_mm(self, point_mm: tuple[float, float]) -> float | None:
        die_corners_mm = self._die_leadframe_corner_points_mm()
        if len(die_corners_mm) != 4:
            return None
        edge_pairs = [
            (die_corners_mm[0], die_corners_mm[1]),
            (die_corners_mm[1], die_corners_mm[2]),
            (die_corners_mm[2], die_corners_mm[3]),
            (die_corners_mm[3], die_corners_mm[0]),
        ]
        distances_mm: list[float] = []
        for start_mm, end_mm in edge_pairs:
            closest_point_mm = self._closest_point_on_segment_mm(point_mm, start_mm, end_mm)
            distances_mm.append(math.dist(point_mm, closest_point_mm))
        return min(distances_mm) if distances_mm else None

    def _die_leadframe_box_center_mm(self) -> tuple[float, float] | None:
        die_corners_mm = self._die_leadframe_corner_points_mm()
        if len(die_corners_mm) != 4:
            return None
        min_x = min(point[0] for point in die_corners_mm)
        max_x = max(point[0] for point in die_corners_mm)
        min_y = min(point[1] for point in die_corners_mm)
        max_y = max(point[1] for point in die_corners_mm)
        return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

    def _leadframe_path_endpoints_valid(self, path_points_mm: list[tuple[float, float]]) -> tuple[bool, str]:
        if len(path_points_mm) < 2:
            return False, "Path needs at least 2 points."
        threshold_mm = LEADFRAME_MIRROR_VALIDATION_THRESHOLD_PX / max(self.scale_px_per_mm, 1e-9)
        start_point_mm = path_points_mm[0]
        end_point_mm = path_points_mm[-1]

        start_anchors_mm = [
            tuple(anchor["point_mm"])
            for anchor in self._leadframe_reference_anchors()
            if anchor.get("kind") == "leg_guide_center"
        ]
        if not start_anchors_mm:
            return False, "No orange start markers are available for validation."
        start_distance_mm = min(math.dist(start_point_mm, anchor_point_mm) for anchor_point_mm in start_anchors_mm)
        if start_distance_mm > threshold_mm:
            return False, f"Mirrored start point is too far from an orange start marker ({start_distance_mm:.3f} mm)."

        end_distance_mm = self._distance_point_to_die_leadframe_edge_mm(end_point_mm)
        if end_distance_mm is None:
            return False, "Die leadframe square is not available for end-point validation."
        if end_distance_mm > threshold_mm:
            return False, f"Mirrored end point is too far from the die leadframe edge ({end_distance_mm:.3f} mm)."
        return True, ""

    def _mirror_leadframe_paths(self, axis_name: str) -> None:
        open_profiles = [profile for profile in self.leadframe_profiles if not profile.closed]
        if not open_profiles:
            self.status_var.set("No saved open lead frame paths are available to mirror.")
            return
        center_mm = self._die_leadframe_box_center_mm()
        if center_mm is None:
            self.status_var.set("Die leadframe square is not available for mirroring.")
            return
        mirrored_profiles: list[LeadProfile] = []
        for profile_index, source_profile in enumerate(open_profiles, start=1):
            mirrored_points_mm: list[tuple[float, float]] = []
            for point_x_mm, point_y_mm in source_profile.points_mm:
                if axis_name == "horizontal":
                    mirrored_points_mm.append((point_x_mm, (2.0 * center_mm[1]) - point_y_mm))
                else:
                    mirrored_points_mm.append((((2.0 * center_mm[0]) - point_x_mm), point_y_mm))
            is_valid, message = self._leadframe_path_endpoints_valid(mirrored_points_mm)
            if not is_valid:
                self.status_var.set(f"Mirror {axis_name} blocked on path {profile_index}: {message}")
                self._redraw_canvas()
                return
            mirrored_profiles.append(LeadProfile(points_mm=mirrored_points_mm, closed=False))
        self.leadframe_profiles.extend(mirrored_profiles)
        self.status_var.set(f"Mirrored {len(mirrored_profiles)} lead frame path(s) across the {axis_name} axis.")
        self._redraw_canvas()
        self._push_payload(launch_if_missing=True)

    def _mirror_leadframe_paths_horizontal(self) -> None:
        self._mirror_leadframe_paths("horizontal")

    def _mirror_leadframe_paths_vertical(self) -> None:
        self._mirror_leadframe_paths("vertical")

    def _die_leadframe_region_centroid_mm(self) -> tuple[float, float]:
        return _combined_leadframe_centroid([profile.points_mm for profile in self.leadframe_profiles])

    def _find_leadframe_anchor_hit(self, event_x: float, event_y: float) -> dict[str, object] | None:
        threshold_px = 8.0
        closest_anchor: dict[str, object] | None = None
        closest_distance_px = float("inf")
        for anchor in self._leadframe_reference_anchors():
            point_px = self._world_to_canvas(tuple(anchor["point_mm"]))
            distance_px = math.dist(point_px, (event_x, event_y))
            if distance_px <= threshold_px and distance_px < closest_distance_px:
                closest_anchor = anchor
                closest_distance_px = distance_px
        if closest_anchor is not None:
            return closest_anchor
        return None

    def _toggle_die_leadframe_pick_mode(self) -> None:
        self.die_leadframe_center_mode_var.set("custom_point")
        self.die_leadframe_pick_active = not self.die_leadframe_pick_active
        if self.die_leadframe_pick_active:
            self.status_var.set("Die leadframe custom-point pick is ON. Click the Step 4 canvas to place the center.")
        else:
            self.status_var.set("Die leadframe custom-point pick is OFF.")
        self._redraw_canvas()

    def _draw_leadframe_design_guide(self) -> None:
        if self.step_index not in {3, 4}:
            return
        body_width_mm = max(0.5, float(self.body_width_var.get()))
        body_depth_mm = max(0.5, float(self.body_depth_var.get()))
        left_top = self._world_to_canvas((-body_width_mm / 2.0, body_depth_mm / 2.0))
        right_bottom = self._world_to_canvas((body_width_mm / 2.0, -body_depth_mm / 2.0))
        self.canvas.create_rectangle(
            left_top[0],
            left_top[1],
            right_bottom[0],
            right_bottom[1],
            fill="#241f1c",
            outline="#3a3028",
            width=2,
        )

        die_corners_mm = self._die_leadframe_corner_points_mm()
        if len(die_corners_mm) == 4:
            flat_corner_points: list[float] = []
            for point_mm in die_corners_mm:
                point_px = self._world_to_canvas(point_mm)
                flat_corner_points.extend([point_px[0], point_px[1]])
            self.canvas.create_polygon(
                *flat_corner_points,
                outline="#b91c1c",
                width=2,
                dash=(6, 4),
                fill="",
            )
            for corner_index, point_mm in enumerate(die_corners_mm, start=1):
                point_px = self._world_to_canvas(point_mm)
                self.canvas.create_oval(
                    point_px[0] - 5,
                    point_px[1] - 5,
                    point_px[0] + 5,
                    point_px[1] + 5,
                    fill="#b91c1c",
                    outline="",
                )
                self.canvas.create_text(
                    point_px[0] + 14,
                    point_px[1] - 12,
                    text=f"C{corner_index}",
                    fill="#7f1d1d",
                    font=("Segoe UI", 9, "bold"),
                )

        keepout_corners_mm = self._die_leadframe_keepout_corners_mm()
        if len(keepout_corners_mm) == 4:
            keepout_flat_points: list[float] = []
            for point_mm in keepout_corners_mm:
                point_px = self._world_to_canvas(point_mm)
                keepout_flat_points.extend([point_px[0], point_px[1]])
            self.canvas.create_polygon(
                *keepout_flat_points,
                outline="#ef4444",
                width=2,
                dash=(3, 3),
                fill="",
            )
            keepout_center_px = self._world_to_canvas((
                sum(point[0] for point in keepout_corners_mm) / 4.0,
                sum(point[1] for point in keepout_corners_mm) / 4.0,
            ))
            self.canvas.create_text(
                keepout_center_px[0],
                keepout_center_px[1] - 18,
                text="Keep-Out +10%",
                fill="#ef4444",
                font=("Segoe UI", 8, "bold"),
            )

        region_center_mm = self._die_leadframe_region_centroid_mm()
        region_center_px = self._world_to_canvas(region_center_mm)
        self.canvas.create_oval(
            region_center_px[0] - 5,
            region_center_px[1] - 5,
            region_center_px[0] + 5,
            region_center_px[1] + 5,
            fill="#15803d",
            outline="",
        )
        self.canvas.create_text(
            region_center_px[0] + 16,
            region_center_px[1] - 12,
            text="Centroid",
            fill="#166534",
            font=("Segoe UI", 9, "bold"),
        )

        custom_center_mm = (float(self.die_leadframe_center_x_var.get()), float(self.die_leadframe_center_y_var.get()))
        custom_center_px = self._world_to_canvas(custom_center_mm)
        custom_color = "#2563eb" if self.die_leadframe_center_mode_var.get() == "custom_point" else "#60a5fa"
        self.canvas.create_line(custom_center_px[0] - 8, custom_center_px[1], custom_center_px[0] + 8, custom_center_px[1], fill=custom_color, width=2)
        self.canvas.create_line(custom_center_px[0], custom_center_px[1] - 8, custom_center_px[0], custom_center_px[1] + 8, fill=custom_color, width=2)
        self.canvas.create_text(
            custom_center_px[0] + 18,
            custom_center_px[1] + 14,
            text="Custom",
            fill=custom_color,
            font=("Segoe UI", 9, "bold"),
        )

        guide_color = "#ff8a2a"
        for guide in self._leadframe_contact_guide_data():
            region_flat_points: list[float] = []
            for point_mm in guide["region_points_mm"]:
                point_px = self._world_to_canvas(tuple(point_mm))
                region_flat_points.extend([point_px[0], point_px[1]])
            self.canvas.create_polygon(
                *region_flat_points,
                fill="#f59e0b",
                outline="#f59e0b",
                stipple="gray25",
                width=1,
            )
            start_px = self._world_to_canvas(tuple(guide["start_mm"]))
            end_px = self._world_to_canvas(tuple(guide["end_mm"]))
            center_px = self._world_to_canvas(tuple(guide["center_mm"]))
            self.canvas.create_line(
                start_px[0],
                start_px[1],
                end_px[0],
                end_px[1],
                fill=guide_color,
                width=6,
                capstyle="round",
            )
            for point_px in (start_px, end_px):
                self.canvas.create_oval(
                    point_px[0] - 4,
                    point_px[1] - 4,
                    point_px[0] + 4,
                    point_px[1] + 4,
                    fill=guide_color,
                    outline="",
                )
            self.canvas.create_oval(
                center_px[0] - 5,
                center_px[1] - 5,
                center_px[0] + 5,
                center_px[1] + 5,
                fill="#fde68a",
                outline="#f97316",
                width=2,
            )
            label_x = (start_px[0] + end_px[0]) / 2.0
            label_y = (start_px[1] + end_px[1]) / 2.0
            self.canvas.create_text(
                label_x + 16,
                label_y,
                text=f"{str(guide['side_name'])[0]}{guide['index']}",
                fill="#fed7aa",
                font=("Segoe UI", 8, "bold"),
            )
            self.canvas.create_text(
                center_px[0] + 14,
                center_px[1] + 12,
                text="Start",
                fill="#fde68a",
                font=("Segoe UI", 8, "bold"),
            )

    def _find_vertex_hit(self, event_x: float, event_y: float) -> int | tuple[int, int] | None:
        if self._is_leadframe_canvas_context():
            for shape_index, profile in enumerate(self.leadframe_profiles):
                for point_index, point in enumerate(profile.points_mm):
                    px, py = self._world_to_canvas(point)
                    if abs(px - event_x) <= 7 and abs(py - event_y) <= 7:
                        return (shape_index, point_index)
            return None
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

    def _snapped_cursor_for_45_degree_path(
        self,
        anchor_px: tuple[float, float],
        cursor_px: tuple[float, float],
    ) -> tuple[tuple[float, float], float] | None:
        dx = cursor_px[0] - anchor_px[0]
        dy = cursor_px[1] - anchor_px[1]
        radial_distance_px = math.hypot(dx, dy)
        if radial_distance_px <= 1e-9:
            return None
        angle_deg = math.degrees(math.atan2(dy, dx))
        snapped_angle_deg = round(angle_deg / 45.0) * 45.0
        angle_delta_deg = abs(((angle_deg - snapped_angle_deg) + 180.0) % 360.0 - 180.0)
        if angle_delta_deg > PATH_ANGLE_SNAP_THRESHOLD_DEG:
            return None
        snapped_angle_rad = math.radians(snapped_angle_deg)
        return (
            (
                anchor_px[0] + (math.cos(snapped_angle_rad) * radial_distance_px),
                anchor_px[1] + (math.sin(snapped_angle_rad) * radial_distance_px),
            ),
            snapped_angle_deg,
        )

    def _snap_dragged_point_to_other_vertices(
        self,
        candidate_mm: tuple[float, float],
        dragged_index: int | tuple[int, int],
    ) -> tuple[float, float]:
        threshold_mm = POINT_AXIS_SNAP_THRESHOLD_PX / max(self.scale_px_per_mm, 1e-9)
        best_x: float | None = None
        best_y: float | None = None
        best_x_delta = float("inf")
        best_y_delta = float("inf")
        if self._is_leadframe_canvas_context():
            for shape_index, profile in enumerate(self.leadframe_profiles):
                for point_index, point_mm in enumerate(profile.points_mm):
                    if dragged_index == (shape_index, point_index):
                        continue
                    delta_x = abs(point_mm[0] - candidate_mm[0])
                    delta_y = abs(point_mm[1] - candidate_mm[1])
                    if delta_x <= threshold_mm and delta_x < best_x_delta:
                        best_x = point_mm[0]
                        best_x_delta = delta_x
                    if delta_y <= threshold_mm and delta_y < best_y_delta:
                        best_y = point_mm[1]
                        best_y_delta = delta_y
        else:
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
        if not self._canvas_step_active():
            return
        if self.step_index == 3:
            if self.die_leadframe_pick_active:
                picked_point_mm = self._canvas_to_world((event.x, event.y))
                self.die_leadframe_center_x_var.set(picked_point_mm[0])
                self.die_leadframe_center_y_var.set(picked_point_mm[1])
                self.die_leadframe_center_mode_var.set("custom_point")
                self.die_leadframe_pick_active = False
                self.status_var.set(
                    f"Die leadframe custom center set to ({picked_point_mm[0]:.3f}, {picked_point_mm[1]:.3f}) mm."
                )
                self._redraw_canvas()
                self._push_payload(launch_if_missing=True)
                return
            self.status_var.set("Die leadframe reference view active. Move to Step 5 to sketch the lead frame.")
            return
        active_profile = self._active_profile()
        active_draft = self._active_draft_points()
        hit_index = self._find_vertex_hit(event.x, event.y)
        if not self._is_leadframe_design_step() and self.distance_pick_active:
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
                point_a = active_profile.points_mm[self.distance_point_indices[0]]
                point_b = active_profile.points_mm[self.distance_point_indices[1]]
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
        anchor_hit = None
        if self._is_leadframe_design_step():
            anchor_hit = self._find_leadframe_anchor_hit(event.x, event.y)
            if anchor_hit is None:
                anchor_hit = self._find_die_leadframe_edge_hit(event.x, event.y)
        anchor_point_mm = tuple(anchor_hit["point_mm"]) if anchor_hit is not None else None
        point_px = self._world_to_canvas(anchor_point_mm) if anchor_point_mm is not None else (event.x, event.y)
        if (
            anchor_hit is not None
            and anchor_hit.get("kind") == "die_edge"
            and active_draft
        ):
            angled_edge_snap_px = self._angled_path_snap_to_die_edge(active_draft[-1], (event.x, event.y), anchor_hit)
            if angled_edge_snap_px is not None:
                point_px = angled_edge_snap_px
        elif anchor_point_mm is None and self.snapped_preview_cursor_px is not None:
            point_px = self.snapped_preview_cursor_px
        active_draft.append(point_px)
        self._clear_snap_state()
        if anchor_hit is not None and anchor_hit.get("kind") == "leg_guide_center":
            self.status_var.set(
                f"Snapped to {anchor_hit['side_name']} leg {anchor_hit['index']} center start point. Draft points: {len(active_draft)}."
            )
        elif anchor_hit is not None and anchor_hit.get("kind") == "leg_guide":
            self.status_var.set(
                f"Snapped to {anchor_hit['side_name']} leg {anchor_hit['index']} {anchor_hit['role']} point. Draft points: {len(active_draft)}."
            )
        elif anchor_hit is not None and anchor_hit.get("kind") == "die_edge":
            self.status_var.set(
                f"Snapped to die leadframe {anchor_hit['edge_name']} edge. Draft points: {len(active_draft)}."
            )
        elif anchor_hit is not None:
            self.status_var.set(f"Lead frame anchor selected. Draft points: {len(active_draft)}.")
        else:
            self.status_var.set(f"Draft points: {len(active_draft)}. Finish the outline when ready.")
        self._redraw_canvas()

    def _on_canvas_drag_motion(self, event) -> None:
        if self.dragging_vertex_index is None:
            return
        if self._is_leadframe_canvas_context():
            if not isinstance(self.dragging_vertex_index, tuple):
                return
            shape_index, point_index = self.dragging_vertex_index
            if 0 <= shape_index < len(self.leadframe_profiles) and 0 <= point_index < len(self.leadframe_profiles[shape_index].points_mm):
                candidate_mm = self._canvas_to_world((event.x, event.y))
                snapped_mm = self._snap_dragged_point_to_other_vertices(candidate_mm, self.dragging_vertex_index)
                self.leadframe_profiles[shape_index].points_mm[point_index] = snapped_mm
                self._redraw_canvas()
                self._push_payload()
            return
        active_profile = self._active_profile()
        if isinstance(self.dragging_vertex_index, int) and 0 <= self.dragging_vertex_index < len(active_profile.points_mm):
            candidate_mm = self._canvas_to_world((event.x, event.y))
            snapped_mm = self._snap_dragged_point_to_other_vertices(candidate_mm, self.dragging_vertex_index)
            active_profile.points_mm[self.dragging_vertex_index] = snapped_mm
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
        if self.step_index == 3:
            self._clear_snap_state()
            self._redraw_canvas()
            return
        active_draft = self._active_draft_points()
        anchor_hit = None
        if self._is_leadframe_design_step():
            anchor_hit = self._find_leadframe_anchor_hit(event.x, event.y)
            if anchor_hit is None:
                anchor_hit = self._find_die_leadframe_edge_hit(event.x, event.y)
        if anchor_hit is not None:
            self._clear_snap_state(keep_lock=True)
            if anchor_hit.get("kind") == "die_edge" and active_draft:
                angled_edge_snap_px = self._angled_path_snap_to_die_edge(active_draft[-1], self.preview_cursor_px, anchor_hit)
                self.snapped_preview_cursor_px = (
                    angled_edge_snap_px
                    if angled_edge_snap_px is not None
                    else self._world_to_canvas(tuple(anchor_hit["point_mm"]))
                )
            else:
                self.snapped_preview_cursor_px = self._world_to_canvas(tuple(anchor_hit["point_mm"]))
            self._redraw_canvas()
            return
        if self.dragging_vertex_index is not None or self.is_panning_canvas:
            self._clear_snap_state()
            self._redraw_canvas()
            return
        if not active_draft:
            self._clear_snap_state()
            self._redraw_canvas()
            return

        anchor_px = active_draft[-1]
        if self._is_leadframe_design_step():
            snapped_path = self._snapped_cursor_for_45_degree_path(anchor_px, self.preview_cursor_px)
            if snapped_path is None:
                self._clear_snap_state()
            else:
                self._clear_snap_state(keep_lock=True)
                self.snapped_preview_cursor_px = snapped_path[0]
        else:
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

    def _on_control_undo(self, _event=None):
        if not self._canvas_step_active():
            return None
        self._undo_point()
        return "break"

    def _undo_point(self) -> None:
        if self.step_index == 3:
            self.status_var.set("Die leadframe reference view active. Sketch edits happen in Step 5.")
            return
        active_draft = self._active_draft_points()
        if active_draft:
            active_draft.pop()
            self._clear_snap_state()
            self.status_var.set(f"Draft points: {len(active_draft)}.")
            self._redraw_canvas()
            return
        if self._is_leadframe_canvas_context():
            if self.leadframe_profiles:
                self.leadframe_profiles.pop()
                self.status_var.set("Removed last saved lead frame shape.")
                self._redraw_canvas()
                self._push_payload()
            return
        active_profile = self._active_profile()
        if active_profile.points_mm:
            active_profile.points_mm.pop()
            if not self._is_leadframe_canvas_context():
                self.distance_point_indices = [index for index in self.distance_point_indices if index < len(active_profile.points_mm)]
            self.status_var.set("Removed last saved sketch point.")
            self._redraw_canvas()
            self._push_payload()

    def _finish_closed_shape(self) -> None:
        if self.step_index == 3:
            self.status_var.set("Die leadframe reference view active. Sketch edits happen in Step 5.")
            return
        active_draft = self._active_draft_points()
        if self._is_leadframe_canvas_context():
            if len(active_draft) < 2:
                messagebox.showerror("Path Incomplete", "Draw at least 2 points before finishing the path.", parent=self.root)
                return
            path_points_mm = [self._canvas_to_world(point_px) for point_px in active_draft]
            simplified_path = _simplify_path_points(path_points_mm)
            if len(simplified_path) < 2:
                messagebox.showerror("Invalid Path", "The path needs at least 2 unique points.", parent=self.root)
                return
            self.leadframe_profiles.append(LeadProfile(points_mm=simplified_path, closed=False))
            self._clear_active_draft_points()
            self._clear_snap_state()
            self.status_var.set("Lead frame path saved.")
            self._redraw_canvas()
            self._push_payload()
            return
        if len(active_draft) < 3:
            messagebox.showerror("Shape Incomplete", "Draw at least 3 points before closing the shape.", parent=self.root)
            return
        points_mm = [self._canvas_to_world(point_px) for point_px in active_draft]
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
        self._replace_active_profile(LeadProfile(points_mm=simplified, closed=True))
        self._clear_active_draft_points()
        self._clear_snap_state()
        self.distance_pick_active = False
        self.distance_point_indices.clear()
        self.status_var.set("Lead profile saved. Move to Step 2 to set the extrusion length.")
        self._redraw_canvas()
        self._push_payload()

    def _clear_profile(self) -> None:
        if self.step_index == 3:
            self.status_var.set("Die leadframe reference view active. Sketch edits happen in Step 5.")
            return
        if self._is_leadframe_canvas_context():
            self.leadframe_profiles.clear()
        else:
            self._replace_active_profile(LeadProfile(points_mm=[]))
        self._clear_active_draft_points()
        self._clear_snap_state()
        self.drag_snap_x_mm = None
        self.drag_snap_y_mm = None
        if not self._is_leadframe_canvas_context():
            self.distance_pick_active = False
            self.distance_point_indices.clear()
            self.status_var.set("Lead profile cleared.")
        else:
            self.status_var.set("Lead frame sketch cleared.")
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
        if self._is_leadframe_canvas_context():
            if not self.leadframe_profiles:
                return
            for shape_index, profile in enumerate(self.leadframe_profiles, start=1):
                if len(profile.points_mm) < 2:
                    continue
                flat_points: list[float] = []
                for point_mm in profile.points_mm:
                    x_px, y_px = self._world_to_canvas(point_mm)
                    flat_points.extend([x_px, y_px])
                if profile.closed and len(profile.points_mm) >= 3:
                    self.canvas.create_polygon(
                        *flat_points,
                        fill="#d9c4a1",
                        outline="#6f5132",
                        width=2,
                        stipple="gray25",
                    )
                else:
                    self.canvas.create_line(*flat_points, fill="#6f5132", width=3)
                for point_index, point_mm in enumerate(profile.points_mm, start=1):
                    x_px, y_px = self._world_to_canvas(point_mm)
                    self.canvas.create_oval(x_px - 5, y_px - 5, x_px + 5, y_px + 5, fill="#2d241f", outline="", width=2)
                    self.canvas.create_text(
                        x_px + 12,
                        y_px - 12,
                        text=f"{shape_index}.{point_index}",
                        fill="#2d241f",
                        font=("Segoe UI", 8, "bold"),
                    )
        else:
            active_profile = self._active_profile()
            if len(active_profile.points_mm) < 2:
                return
            flat_points: list[float] = []
            for point_mm in active_profile.points_mm:
                x_px, y_px = self._world_to_canvas(point_mm)
                flat_points.extend([x_px, y_px])
            self.canvas.create_polygon(
                *flat_points,
                fill="#d9c4a1",
                outline="#6f5132",
                width=2,
                stipple="gray25",
            )
            for index, point_mm in enumerate(active_profile.points_mm):
                x_px, y_px = self._world_to_canvas(point_mm)
                point_fill = "#2d241f"
                point_outline = ""
                if index in self.distance_point_indices:
                    point_fill = "#3f6b5b"
                    point_outline = "#dceee6"
                self.canvas.create_oval(x_px - 5, y_px - 5, x_px + 5, y_px + 5, fill=point_fill, outline=point_outline, width=2)
                self.canvas.create_text(x_px + 12, y_px - 12, text=str(index + 1), fill="#2d241f", font=("Segoe UI", 9, "bold"))

        if not self._is_leadframe_canvas_context() and len(self.distance_point_indices) == 2:
            first_point = active_profile.points_mm[self.distance_point_indices[0]]
            second_point = active_profile.points_mm[self.distance_point_indices[1]]
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
        active_draft = self._active_draft_points()
        if not active_draft:
            if self.snapped_preview_cursor_px is not None:
                self.canvas.create_oval(
                    self.snapped_preview_cursor_px[0] - 5,
                    self.snapped_preview_cursor_px[1] - 5,
                    self.snapped_preview_cursor_px[0] + 5,
                    self.snapped_preview_cursor_px[1] + 5,
                    fill="#3f6b5b",
                    outline="",
                )
            return
        flat_points: list[float] = []
        for x_px, y_px in active_draft:
            flat_points.extend([x_px, y_px])
            self.canvas.create_oval(x_px - 4, y_px - 4, x_px + 4, y_px + 4, fill="#7d4b2f", outline="")
        if len(flat_points) >= 4:
            self.canvas.create_line(*flat_points, fill="#7d4b2f", width=2)
        preview_point = self.snapped_preview_cursor_px if self.snapped_preview_cursor_px is not None else self.preview_cursor_px
        if preview_point is not None:
            last_x, last_y = active_draft[-1]
            line_color = "#3f6b5b" if self.snapped_preview_cursor_px is not None else "#7d4b2f"
            self.canvas.create_line(last_x, last_y, preview_point[0], preview_point[1], fill=line_color, dash=(5, 4), width=2)
            if self.snapped_preview_cursor_px is not None:
                self.canvas.create_oval(preview_point[0] - 4, preview_point[1] - 4, preview_point[0] + 4, preview_point[1] + 4, fill=line_color, outline="")

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_stage_three_side_guide()
        self._draw_leadframe_design_guide()
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
                "leadframe_profile_points_mm": [list(point) for point in (self.leadframe_profiles[0].points_mm if self.leadframe_profiles else [])],
                "leadframe_profiles_points_mm": [
                    [list(point) for point in profile.points_mm]
                    for profile in self.leadframe_profiles
                ],
                "leadframe_profiles": [
                    {
                        "points_mm": [list(point) for point in profile.points_mm],
                        "closed": profile.closed,
                    }
                    for profile in self.leadframe_profiles
                ],
                "leadframe_draft_points_px": [list(point) for point in self.leadframe_current_points_px],
                "distance_target_mm": float(self.distance_target_var.get()),
                "lead_offset_mm": float(self.lead_offset_var.get()),
                "die_leadframe_width_mm": float(self.die_leadframe_width_var.get()),
                "die_leadframe_depth_mm": float(self.die_leadframe_depth_var.get()),
                "die_leadframe_thickness_mm": float(self.die_leadframe_thickness_var.get()),
                "leadframe_path_width_mm": float(self.leadframe_path_width_var.get()),
                "leadframe_path_thickness_mm": float(self.leadframe_path_thickness_var.get()),
                "die_leadframe_center_mode": self.die_leadframe_center_mode_var.get().strip().lower() or "region_centroid",
                "die_leadframe_center_x_mm": float(self.die_leadframe_center_x_var.get()),
                "die_leadframe_center_y_mm": float(self.die_leadframe_center_y_var.get()),
                "silicon_die_width_mm": float(self.silicon_die_width_var.get()),
                "silicon_die_depth_mm": float(self.silicon_die_depth_var.get()),
                "silicon_die_thickness_mm": float(self.silicon_die_thickness_var.get()),
                "leg_pick_distance_mm": float(self.leg_pick_distance_var.get()),
                "leg_pick_marker_size_mm": float(self.leg_pick_marker_size_var.get()),
                "die_region_span_percent": float(self.die_region_span_percent_var.get()),
                "die_region_depth_mm": float(self.die_region_depth_var.get()),
                "die_region_offset_mm": float(self.die_region_offset_var.get()),
                "die_region_top_count": int(self.die_region_top_count_var.get()),
                "die_region_bottom_count": int(self.die_region_bottom_count_var.get()),
                "die_region_left_count": int(self.die_region_left_count_var.get()),
                "die_region_right_count": int(self.die_region_right_count_var.get()),
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
                "encapsulation_height_mm": float(self.encapsulation_height_var.get()),
                "simulation_clearance_mm": float(self.simulation_clearance_var.get()),
                "step16_split_gap_mm": float(self.step16_split_gap_var.get()),
                "show_step16_lead_system": bool(self.show_step16_lead_system_var.get()),
                "show_step16_die_leadframe": bool(self.show_step16_die_leadframe_var.get()),
                "show_step16_silicon_die": bool(self.show_step16_silicon_die_var.get()),
                "show_step16_bond_assembly": bool(self.show_step16_bond_assembly_var.get()),
                "show_step16_encapsulation_base": bool(self.show_step16_encapsulation_base_var.get()),
                "show_step16_encapsulation_top": bool(self.show_step16_encapsulation_top_var.get()),
                "step16_lead_system_color": self.step16_lead_system_color_var.get().strip() or "#c58a34",
                "step16_die_leadframe_color": self.step16_die_leadframe_color_var.get().strip() or "#8e3f2b",
                "step16_silicon_die_color": self.step16_silicon_die_color_var.get().strip() or "#232323",
                "step16_bond_assembly_color": self.step16_bond_assembly_color_var.get().strip() or "#2563eb",
                "step16_encapsulation_base_color": self.step16_encapsulation_base_color_var.get().strip() or DEFAULT_BODY_COLOR,
                "step16_encapsulation_top_color": self.step16_encapsulation_top_color_var.get().strip() or "#6f7b83",
                "selected_intersection_pairs": self._selected_intersection_pairs(),
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
            payload["die_leadframe"] = {
                "die_leadframe_width_mm": float(self.die_leadframe_width_var.get()),
                "die_leadframe_depth_mm": float(self.die_leadframe_depth_var.get()),
                "die_leadframe_thickness_mm": float(self.die_leadframe_thickness_var.get()),
                "leadframe_path_width_mm": float(self.leadframe_path_width_var.get()),
                "leadframe_path_thickness_mm": float(self.leadframe_path_thickness_var.get()),
                "die_leadframe_center_mode": self.die_leadframe_center_mode_var.get().strip().lower() or "region_centroid",
                "die_leadframe_center_x_mm": float(self.die_leadframe_center_x_var.get()),
                "die_leadframe_center_y_mm": float(self.die_leadframe_center_y_var.get()),
            }
        if stage_index >= 4:
            payload["lead_frame_design"] = {
                "leadframe_profile_points_mm": [list(point) for point in (self.leadframe_profiles[0].points_mm if self.leadframe_profiles else [])],
                "leadframe_profiles_points_mm": [
                    [list(point) for point in profile.points_mm]
                    for profile in self.leadframe_profiles
                ],
                "leadframe_profiles": [
                    {
                        "points_mm": [list(point) for point in profile.points_mm],
                        "closed": profile.closed,
                    }
                    for profile in self.leadframe_profiles
                ],
                "leadframe_draft_points_px": [list(point) for point in self.leadframe_current_points_px],
            }
        if stage_index >= 5:
            payload["overall_3d_placement"] = {
                "body_width_mm": float(self.body_width_var.get()),
                "body_depth_mm": float(self.body_depth_var.get()),
                "body_height_mm": float(self.body_height_var.get()),
            }
        if stage_index >= 6:
            payload["lead_offset"] = {
                "lead_offset_mm": float(self.lead_offset_var.get()),
            }
        if stage_index >= 7:
            payload["silicon_die"] = {
                "silicon_die_width_mm": float(self.silicon_die_width_var.get()),
                "silicon_die_depth_mm": float(self.silicon_die_depth_var.get()),
                "silicon_die_thickness_mm": float(self.silicon_die_thickness_var.get()),
            }
        if stage_index >= 8:
            payload["leg_positions"] = {
                "leg_pick_distance_mm": float(self.leg_pick_distance_var.get()),
                "leg_pick_marker_size_mm": float(self.leg_pick_marker_size_var.get()),
            }
        if stage_index >= 9:
            payload["die_regions"] = {
                "die_region_span_percent": float(self.die_region_span_percent_var.get()),
                "die_region_depth_mm": float(self.die_region_depth_var.get()),
                "die_region_offset_mm": float(self.die_region_offset_var.get()),
                "die_region_top_count": int(self.die_region_top_count_var.get()),
                "die_region_bottom_count": int(self.die_region_bottom_count_var.get()),
                "die_region_left_count": int(self.die_region_left_count_var.get()),
                "die_region_right_count": int(self.die_region_right_count_var.get()),
                "die_pick_region": self.die_pick_region_var.get().strip().title() or "Top",
                "die_pick_section_index": int(self.die_pick_section_index_var.get()),
                "die_pick_position_percent": float(self.die_pick_position_percent_var.get()),
                "die_pick_marker_size_mm": float(self.die_pick_marker_size_var.get()),
            }
        if stage_index >= 10:
            payload["bond_arcs"] = {
                "arc_height_mm": float(self.arc_height_var.get()),
                "arc_xy_noise_mm": float(self.arc_xy_noise_var.get()),
                "wire_arc_point_spacing_mm": float(self.wire_arc_point_spacing_var.get()),
            }
        if stage_index >= 11:
            payload["ball_bond_formation"] = {
                "ball_bond_diameter_mm": float(self.ball_bond_diameter_var.get()),
                "ball_bond_length_mm": float(self.ball_bond_length_var.get()),
                "ball_bond_revolution_steps": int(self.ball_bond_revolution_steps_var.get()),
            }
        if stage_index >= 12:
            payload["bond_wire_tube"] = {
                "wire_diameter_mm": float(self.wire_diameter_var.get()),
                "wire_rise_z_mm": float(self.wire_rise_z_var.get()),
                "wire_arc_point_spacing_mm": float(self.wire_arc_point_spacing_var.get()),
                "wire_tube_side_count": int(self.wire_tube_side_count_var.get()),
            }
        if stage_index >= 13:
            payload["wedge_bond_ending"] = {
                "wedge_bond_length_mm": float(self.wedge_bond_length_var.get()),
                "wedge_bond_width_mm": float(self.wedge_bond_width_var.get()),
                "wedge_bond_thickness_mm": float(self.wedge_bond_thickness_var.get()),
                "wedge_approach_run_mm": float(self.wedge_approach_run_var.get()),
            }
        if stage_index >= 14:
            payload["encapsulation"] = {
                "encapsulation_height_mm": float(self.encapsulation_height_var.get()),
                "simulation_clearance_mm": float(self.simulation_clearance_var.get()),
            }
        if stage_index >= 15:
            payload["intersection_check"] = {
                "step16_split_gap_mm": float(self.step16_split_gap_var.get()),
                "show_step16_lead_system": bool(self.show_step16_lead_system_var.get()),
                "show_step16_die_leadframe": bool(self.show_step16_die_leadframe_var.get()),
                "show_step16_silicon_die": bool(self.show_step16_silicon_die_var.get()),
                "show_step16_bond_assembly": bool(self.show_step16_bond_assembly_var.get()),
                "show_step16_encapsulation_base": bool(self.show_step16_encapsulation_base_var.get()),
                "show_step16_encapsulation_top": bool(self.show_step16_encapsulation_top_var.get()),
                "step16_lead_system_color": self.step16_lead_system_color_var.get().strip() or "#c58a34",
                "step16_die_leadframe_color": self.step16_die_leadframe_color_var.get().strip() or "#8e3f2b",
                "step16_silicon_die_color": self.step16_silicon_die_color_var.get().strip() or "#232323",
                "step16_bond_assembly_color": self.step16_bond_assembly_color_var.get().strip() or "#2563eb",
                "step16_encapsulation_base_color": self.step16_encapsulation_base_color_var.get().strip() or DEFAULT_BODY_COLOR,
                "step16_encapsulation_top_color": self.step16_encapsulation_top_color_var.get().strip() or "#6f7b83",
                "selected_intersection_pairs": self._selected_intersection_pairs(),
                "intersection_report": self.intersection_report_var.get(),
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
            if self.step_index >= 15:
                geometry = build_scene_geometry(project_payload)
                intersections = geometry.get("intersection_pairs", [])
                warnings = geometry.get("encapsulation_warnings", [])
                selected_pairs = self._selected_intersection_pairs()
                available_pair_labels = [str(item.get("pair", "")).strip() for item in intersections if str(item.get("pair", "")).strip()]
                selected_pairs = [pair_label for pair_label in selected_pairs if pair_label in available_pair_labels]
                self._refresh_intersection_pair_checkboxes(intersections, selected_pairs)
                if intersections:
                    report_lines = ["Intersecting parts found:"]
                    for item in intersections:
                        pair_text = str(item.get("pair", "Unknown intersection"))
                        details_text = str(item.get("details", "")).strip()
                        report_lines.append(f"- {pair_text}")
                        if details_text:
                            report_lines.append(f"  {details_text}")
                else:
                    report_lines = ["No intersections detected between the simulation solids."]
                if warnings:
                    report_lines.append("")
                    report_lines.append("Warnings:")
                    report_lines.extend(f"- {item}" for item in warnings)
                self.intersection_report_var.set("\n".join(report_lines))
                project_payload = self._project_payload()
            elif self.step_index >= 14:
                geometry = build_scene_geometry(project_payload)
                warnings = geometry.get("encapsulation_warnings", [])
                if warnings:
                    self.intersection_report_var.set("Encapsulation warnings:\n" + "\n".join(f"- {item}" for item in warnings))
                else:
                    self.intersection_report_var.set("Step 16 will list any remaining intersecting parts.")
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
