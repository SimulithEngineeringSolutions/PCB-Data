from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import json
import numpy as np
import trimesh
from vedo import Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "ic_chip_generator_New"
PROJECTS_ROOT_DIR = DEFAULT_OUTPUT_DIR / "projects"
DEFAULT_CANVAS_WIDTH = 980
DEFAULT_CANVAS_HEIGHT = 700
DEFAULT_SCALE_PX_PER_MM = 24.0
CANVAS_BACKGROUND = "#f7f0e4"
CANVAS_GRID = "#e6dac7"
CANVAS_AXIS = "#c7b39b"
VIEWER_PAYLOAD_NAME = "step2_viewer_payload.json"
DIE_CLEARANCE_MM = 0.001
LEADFRAME_CONTACT_CLEARANCE_MM = 0.001


def _safe_stage_slug(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")
    return slug or "stage"


def _payload_float(payload: dict, key: str, fallback: float) -> float:
    try:
        return float(payload.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _payload_int(payload: dict, key: str, fallback: int) -> int:
    try:
        return int(payload.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def read_viewer_payload(payload_path: Path) -> dict:
    if not payload_path.exists():
        return {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_2d(vector_xy: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector_xy[0], vector_xy[1])
    if length <= 1e-12:
        return None
    return (vector_xy[0] / length, vector_xy[1] / length)


def _left_normal_2d(direction_xy: tuple[float, float]) -> tuple[float, float]:
    return (-direction_xy[1], direction_xy[0])


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


def _oriented_triangle_indices(
    vertices_xyz: list[list[float]],
    index_a: int,
    index_b: int,
    index_c: int,
    reference_vector_xyz: np.ndarray | None,
) -> list[int]:
    if reference_vector_xyz is None or float(np.linalg.norm(reference_vector_xyz)) <= 1e-12:
        return [index_a, index_b, index_c]
    point_a = np.asarray(vertices_xyz[index_a], dtype=float)
    point_b = np.asarray(vertices_xyz[index_b], dtype=float)
    point_c = np.asarray(vertices_xyz[index_c], dtype=float)
    triangle_normal = np.cross(point_b - point_a, point_c - point_a)
    if float(np.linalg.norm(triangle_normal)) <= 1e-12:
        return [index_a, index_b, index_c]
    if float(np.dot(triangle_normal, reference_vector_xyz)) < 0.0:
        return [index_a, index_c, index_b]
    return [index_a, index_b, index_c]


def _make_face_winding_consistent(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    faces = np.asarray(mesh.faces, dtype=int).copy()
    edge_to_faces: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}

    for face_index, (index_a, index_b, index_c) in enumerate(faces):
        directed_edges = ((index_a, index_b), (index_b, index_c), (index_c, index_a))
        for edge_start, edge_end in directed_edges:
            edge_key = (min(edge_start, edge_end), max(edge_start, edge_end))
            edge_to_faces.setdefault(edge_key, []).append((face_index, (edge_start, edge_end)))

    visited: set[int] = set()
    for seed_face_index in range(len(faces)):
        if seed_face_index in visited:
            continue
        stack = [seed_face_index]
        visited.add(seed_face_index)
        while stack:
            current_face_index = stack.pop()
            current_face = faces[current_face_index]
            current_directed_edges = (
                (current_face[0], current_face[1]),
                (current_face[1], current_face[2]),
                (current_face[2], current_face[0]),
            )
            for current_edge_start, current_edge_end in current_directed_edges:
                edge_key = (min(current_edge_start, current_edge_end), max(current_edge_start, current_edge_end))
                for neighbor_face_index, neighbor_directed_edge in edge_to_faces.get(edge_key, []):
                    if neighbor_face_index == current_face_index:
                        continue
                    if neighbor_face_index not in visited:
                        if neighbor_directed_edge == (current_edge_start, current_edge_end):
                            faces[neighbor_face_index, 1], faces[neighbor_face_index, 2] = (
                                faces[neighbor_face_index, 2],
                                faces[neighbor_face_index, 1],
                            )
                        visited.add(neighbor_face_index)
                        stack.append(neighbor_face_index)
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=float).copy(), faces=faces, process=False)


def _closest_point_and_tangent_on_polyline(
    point_xyz: tuple[float, float, float],
    polyline_xyz: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(polyline_xyz) < 2:
        point_array = np.asarray(point_xyz, dtype=float)
        return point_array, np.asarray((1.0, 0.0, 0.0), dtype=float)

    point_array = np.asarray(point_xyz, dtype=float)
    best_point = np.asarray(polyline_xyz[0], dtype=float)
    best_tangent = np.asarray((1.0, 0.0, 0.0), dtype=float)
    best_distance_sq = float("inf")

    for start_xyz, end_xyz in zip(polyline_xyz[:-1], polyline_xyz[1:]):
        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-12:
            continue
        t_value = float(np.dot(point_array - start, segment) / segment_length_sq)
        t_value = max(0.0, min(1.0, t_value))
        candidate = start + (segment * t_value)
        distance_sq = float(np.dot(point_array - candidate, point_array - candidate))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_point = candidate
            best_tangent = segment / math.sqrt(segment_length_sq)
    return best_point, best_tangent


def _orient_faces_by_reference_vectors(
    mesh: trimesh.Trimesh,
    reference_vectors_by_face: list[np.ndarray | None],
) -> trimesh.Trimesh:
    if len(mesh.faces) == 0 or len(reference_vectors_by_face) != len(mesh.faces):
        return mesh
    corrected_faces = np.asarray(mesh.faces, dtype=int).copy()
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    for face_index, reference_vector in enumerate(reference_vectors_by_face):
        if reference_vector is None:
            continue
        vector_length = float(np.linalg.norm(reference_vector))
        if vector_length <= 1e-12:
            continue
        if float(np.dot(face_normals[face_index], reference_vector)) < 0.0:
            corrected_faces[face_index, 1], corrected_faces[face_index, 2] = corrected_faces[face_index, 2], corrected_faces[face_index, 1]
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=float).copy(), faces=corrected_faces, process=False)


def _flip_all_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    flipped_faces = np.asarray(mesh.faces, dtype=int).copy()
    flipped_faces[:, [1, 2]] = flipped_faces[:, [2, 1]]
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=float).copy(), faces=flipped_faces, process=False)


def _whole_mesh_reference_score(mesh: trimesh.Trimesh, reference_vectors_by_face: list[np.ndarray | None]) -> float:
    if len(mesh.faces) == 0 or len(reference_vectors_by_face) != len(mesh.faces):
        return 0.0
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    score = 0.0
    weight = 0.0
    for face_normal, reference_vector in zip(face_normals, reference_vectors_by_face):
        if reference_vector is None:
            continue
        vector_length = float(np.linalg.norm(reference_vector))
        if vector_length <= 1e-12:
            continue
        score += float(np.dot(face_normal, reference_vector / vector_length))
        weight += 1.0
    if weight <= 1e-12:
        return 0.0
    return score / weight


def _orient_tube_faces_from_anchor_path(
    mesh: trimesh.Trimesh,
    anchor_path_xyz: list[tuple[float, float, float]],
    *,
    body_face_end: int,
    start_cap_face_range: tuple[int, int] | None = None,
    end_cap_face_range: tuple[int, int] | None = None,
) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    mesh = _make_face_winding_consistent(mesh)
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    reference_vectors: list[np.ndarray | None] = [None] * len(mesh.faces)

    for face_index in range(min(body_face_end, len(mesh.faces))):
        nearest_point, _nearest_tangent = _closest_point_and_tangent_on_polyline(tuple(face_centers[face_index]), anchor_path_xyz)
        reference_vectors[face_index] = face_centers[face_index] - nearest_point

    if start_cap_face_range is not None and len(anchor_path_xyz) >= 2:
        start_tangent = np.asarray(_vector_normalize(_vector_sub(anchor_path_xyz[1], anchor_path_xyz[0])), dtype=float)
        for face_index in range(start_cap_face_range[0], min(start_cap_face_range[1], len(mesh.faces))):
            reference_vectors[face_index] = -start_tangent

    if end_cap_face_range is not None and len(anchor_path_xyz) >= 2:
        end_tangent = np.asarray(_vector_normalize(_vector_sub(anchor_path_xyz[-1], anchor_path_xyz[-2])), dtype=float)
        for face_index in range(end_cap_face_range[0], min(end_cap_face_range[1], len(mesh.faces))):
            reference_vectors[face_index] = end_tangent

    if _whole_mesh_reference_score(mesh, reference_vectors) < 0.0:
        mesh = _flip_all_faces(mesh)
    return mesh


def _orient_ball_faces_from_centroid(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) == 0:
        return mesh
    mesh = _make_face_winding_consistent(mesh)
    try:
        centroid = np.asarray(mesh.centroid, dtype=float)
    except Exception:
        centroid = np.mean(np.asarray(mesh.vertices, dtype=float), axis=0)
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    reference_vectors = [face_center - centroid for face_center in face_centers]
    if _whole_mesh_reference_score(mesh, reference_vectors) < 0.0:
        mesh = _flip_all_faces(mesh)
    return mesh


def _choose_frame_reference(tangent: tuple[float, float, float]) -> tuple[float, float, float]:
    for candidate in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        if _vector_length(_vector_cross(candidate, tangent)) > 1e-6:
            return candidate
    return (1.0, 0.0, 0.0)


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
    return simplified


def _triangulate_polygon(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    simplified = _simplify_profile_points(points)
    if len(simplified) < 3:
        raise ValueError("Polygon needs at least 3 unique points.")
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
            raise ValueError("Failed to triangulate stroked path polygon.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _extrude_closed_polygon(points_mm: list[tuple[float, float]], height_mm: float) -> trimesh.Trimesh:
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
    return _ensure_outward_normals(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))


def _ensure_outward_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh is None or len(mesh.faces) == 0:
        return mesh

    def orient_single(component_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        candidate = component_mesh.copy()
        try:
            candidate.fix_normals(multibody=False)
        except TypeError:
            try:
                candidate.fix_normals()
            except Exception:
                pass
        except Exception:
            pass

        try:
            if candidate.is_volume:
                if float(candidate.volume) < 0.0:
                    candidate.invert()
                return candidate
        except Exception:
            pass

        try:
            face_centers = np.asarray(candidate.triangles_center, dtype=float)
            face_normals = np.asarray(candidate.face_normals, dtype=float)
            if len(face_centers) and len(face_centers) == len(face_normals):
                bounds = candidate.bounds
                mesh_center = np.asarray(
                    [
                        (bounds[0][0] + bounds[1][0]) / 2.0,
                        (bounds[0][1] + bounds[1][1]) / 2.0,
                        (bounds[0][2] + bounds[1][2]) / 2.0,
                    ],
                    dtype=float,
                )
                outward_score = float(np.mean(np.einsum("ij,ij->i", face_normals, face_centers - mesh_center)))
                if outward_score < 0.0:
                    candidate.invert()
        except Exception:
            pass
        return candidate

    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = []
    if not components:
        return orient_single(mesh)
    oriented_components = [orient_single(component) for component in components if len(component.faces)]
    if not oriented_components:
        return orient_single(mesh)
    if len(oriented_components) == 1:
        return oriented_components[0]
    return trimesh.util.concatenate(oriented_components)


def _build_stroked_path_polygon(points_mm: list[tuple[float, float]], width_mm: float) -> list[tuple[float, float]]:
    simplified_points = _simplify_profile_points(points_mm)
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


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array([
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _translation_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 3] = dx
    matrix[1, 3] = dy
    matrix[2, 3] = dz
    return matrix


def _rotation_matrix_xyz(rotation_deg_xyz: tuple[float, float, float]) -> np.ndarray:
    rot_x = math.radians(rotation_deg_xyz[0])
    rot_y = math.radians(rotation_deg_xyz[1])
    rot_z = math.radians(rotation_deg_xyz[2])
    cx, sx = math.cos(rot_x), math.sin(rot_x)
    cy, sy = math.cos(rot_y), math.sin(rot_y)
    cz, sz = math.cos(rot_z), math.sin(rot_z)
    rx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cx, -sx, 0.0], [0.0, sx, cx, 0.0], [0.0, 0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy, 0.0], [0.0, 1.0, 0.0, 0.0], [-sy, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]])
    rz = np.array([[cz, -sz, 0.0, 0.0], [sz, cz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _die_square_size_from_payload(payload: dict) -> float:
    outline_x_mm = max(1.0, _payload_float(payload, "outline_x_mm", 10.0))
    outline_y_mm = max(1.0, _payload_float(payload, "outline_y_mm", 10.0))
    die_size_mm = _payload_float(payload, "die_compartment_square_size_mm", 3.0)
    return max(0.5, min(die_size_mm, min(outline_x_mm, outline_y_mm) * 0.9))


def _silicon_die_dimensions_from_payload(payload: dict) -> tuple[float, float]:
    compartment_size_mm = _die_square_size_from_payload(payload)
    die_width_mm = _payload_float(payload, "silicon_die_width_mm", compartment_size_mm * 0.8)
    die_height_mm = _payload_float(payload, "silicon_die_height_mm", compartment_size_mm * 0.8)
    return (
        max(0.05, die_width_mm),
        max(0.05, die_height_mm),
    )


def _encapsulation_dimensions_from_payload(payload: dict) -> tuple[float, float, float, float]:
    outline_x_mm = max(1.0, _payload_float(payload, "outline_x_mm", 10.0))
    outline_y_mm = max(1.0, _payload_float(payload, "outline_y_mm", 10.0))
    width_mm = max(0.1, _payload_float(payload, "encapsulation_width_mm", outline_x_mm * 0.7))
    length_mm = max(0.1, _payload_float(payload, "encapsulation_length_mm", outline_y_mm * 0.7))
    negative_z_mm = max(0.0, _payload_float(payload, "encapsulation_negative_extrusion_mm", 0.0))
    positive_z_mm = max(0.0, _payload_float(payload, "encapsulation_positive_extrusion_mm", 0.6))
    return (width_mm, length_mm, negative_z_mm, positive_z_mm)


def _viewer_visibility_from_payload(payload: dict) -> dict[str, bool]:
    visibility = payload.get("step5_viewer_visibility", {})
    if not isinstance(visibility, dict):
        visibility = {}
    return {
        "Centered Die Compartment": bool(visibility.get("centered_die_compartment", False)),
        "Silicon Die": bool(visibility.get("silicon_die", True)),
        "Lead Frame Paths": bool(visibility.get("lead_frame_paths", True)),
        "Bond Assemblies": bool(visibility.get("bond_assemblies", True)),
        "Scaled Outer Model": bool(visibility.get("scaled_outer_model", True)),
        "Encapsulation": bool(visibility.get("encapsulation", True)),
    }


def _bond_start_region_centers_from_payload(payload: dict) -> list[dict[str, object]]:
    die_width_mm, die_height_mm = _silicon_die_dimensions_from_payload(payload)
    half_die_width_mm = die_width_mm / 2.0
    half_die_height_mm = die_height_mm / 2.0
    region_size_mm = max(0.05, _payload_float(payload, "bond_start_region_size_mm", 0.2))
    gap_mm = max(0.0, _payload_float(payload, "bond_start_region_gap_mm", 0.15))
    offset_mm = max(0.0, _payload_float(payload, "bond_start_region_offset_mm", 0.08))
    count_payload = payload.get("bond_start_region_counts", {})
    counts = {
        "Top": _payload_int(count_payload if isinstance(count_payload, dict) else {}, "top", 2),
        "Bottom": _payload_int(count_payload if isinstance(count_payload, dict) else {}, "bottom", 2),
        "Left": _payload_int(count_payload if isinstance(count_payload, dict) else {}, "left", 2),
        "Right": _payload_int(count_payload if isinstance(count_payload, dict) else {}, "right", 2),
    }
    regions: list[dict[str, object]] = []

    def positions_for_side(count: int, usable_side_mm: float) -> list[float]:
        if count <= 0:
            return []
        if count == 1:
            return [0.0]
        total_span_mm = (count * region_size_mm) + ((count - 1) * gap_mm)
        usable_span_mm = min(total_span_mm, usable_side_mm)
        actual_gap_mm = gap_mm
        if count > 1 and total_span_mm > usable_side_mm:
            actual_gap_mm = max(0.0, (usable_side_mm - (count * region_size_mm)) / (count - 1))
            usable_span_mm = (count * region_size_mm) + ((count - 1) * actual_gap_mm)
        start_center_mm = -(usable_span_mm / 2.0) + (region_size_mm / 2.0)
        return [start_center_mm + (index * (region_size_mm + actual_gap_mm)) for index in range(count)]

    for index, x_center_mm in enumerate(positions_for_side(max(0, counts["Top"]), die_width_mm), start=1):
        regions.append({"side_name": "Top", "section_index": index, "center_mm": (x_center_mm, half_die_height_mm - offset_mm - (region_size_mm / 2.0))})
    for index, x_center_mm in enumerate(positions_for_side(max(0, counts["Bottom"]), die_width_mm), start=1):
        regions.append({"side_name": "Bottom", "section_index": index, "center_mm": (x_center_mm, -half_die_height_mm + offset_mm + (region_size_mm / 2.0))})
    for index, y_center_mm in enumerate(positions_for_side(max(0, counts["Left"]), die_height_mm), start=1):
        regions.append({"side_name": "Left", "section_index": index, "center_mm": (-half_die_width_mm + offset_mm + (region_size_mm / 2.0), y_center_mm)})
    for index, y_center_mm in enumerate(positions_for_side(max(0, counts["Right"]), die_height_mm), start=1):
        regions.append({"side_name": "Right", "section_index": index, "center_mm": (half_die_width_mm - offset_mm - (region_size_mm / 2.0), y_center_mm)})
    return regions


def _bond_end_region_centers_from_payload(payload: dict) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    offset_mm = max(0.0, _payload_float(payload, "bond_end_region_offset_mm", 0.0))
    for path_index, path_payload in enumerate(payload.get("saved_paths_mm", []), start=1):
        if not isinstance(path_payload, list):
            continue
        path_mm = [tuple(point) for point in path_payload if isinstance(point, (list, tuple)) and len(point) == 2]
        if len(path_mm) < 2:
            continue
        end_x_mm, end_y_mm = path_mm[-1]
        previous_x_mm, previous_y_mm = path_mm[-2]
        direction_xy = _normalize_2d((previous_x_mm - end_x_mm, previous_y_mm - end_y_mm))
        if direction_xy is None:
            center_mm = (end_x_mm, end_y_mm)
        else:
            center_mm = (
                end_x_mm + (direction_xy[0] * offset_mm),
                end_y_mm + (direction_xy[1] * offset_mm),
            )
        dx = end_x_mm - path_mm[0][0]
        dy = end_y_mm - path_mm[0][1]
        if abs(dx) >= abs(dy):
            side_name = "Right" if dx >= 0.0 else "Left"
        else:
            side_name = "Top" if dy >= 0.0 else "Bottom"
        regions.append(
            {
                "path_index": path_index,
                "side_name": side_name,
                "center_mm": center_mm,
                "approach_mm": (previous_x_mm, previous_y_mm),
                "path_end_mm": (end_x_mm, end_y_mm),
            }
        )
    return regions


def _clockwise_indexed_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    side_order = {"Top": 0, "Right": 1, "Bottom": 2, "Left": 3}

    def sort_key(item: dict[str, object]) -> tuple[float, float]:
        side_name = str(item.get("side_name", "")).title()
        x_coord, y_coord = item.get("center_xy", item.get("center_mm", (0.0, 0.0)))
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


def _pair_nearest_leg_regions(leg_markers: list[dict[str, object]], die_regions: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    if not leg_markers or not die_regions:
        return []
    remaining_regions = list(die_regions)
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for leg_data in leg_markers:
        leg_x, leg_y = leg_data.get("center_xy", leg_data.get("center_mm", (0.0, 0.0)))
        nearest_index = min(
            range(len(remaining_regions)),
            key=lambda index: math.dist((leg_x, leg_y), remaining_regions[index].get("center_xy", remaining_regions[index].get("center_mm", (0.0, 0.0)))),
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
        x_coord = ((one_minus_t ** 3) * start[0]) + (3.0 * (one_minus_t ** 2) * t_value * control_1[0]) + (3.0 * one_minus_t * (t_value ** 2) * control_2[0]) + ((t_value ** 3) * end[0])
        y_coord = ((one_minus_t ** 3) * start[1]) + (3.0 * (one_minus_t ** 2) * t_value * control_1[1]) + (3.0 * one_minus_t * (t_value ** 2) * control_2[1]) + ((t_value ** 3) * end[1])
        z_coord = ((one_minus_t ** 3) * start[2]) + (3.0 * (one_minus_t ** 2) * t_value * control_1[2]) + (3.0 * one_minus_t * (t_value ** 2) * control_2[2]) + ((t_value ** 3) * end[2])
        points.append([x_coord, y_coord, z_coord])
    return points


def _build_swept_tube_mesh(
    points: list[tuple[float, float, float]],
    radius_mm: float,
    side_count: int,
    *,
    cap_start: bool = True,
    cap_end: bool = True,
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
            offset = _vector_add(_vector_scale(lateral, math.cos(angle) * radius_mm), _vector_scale(vertical, math.sin(angle) * radius_mm))
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
    body_face_end = len(faces)

    start_cap_face_range: tuple[int, int] | None = None
    if cap_start:
        start_cap_start = len(faces)
        start_center_index = len(vertices)
        vertices.append(list(points[0]))
        for vertex_index in range(section_size):
            next_vertex_index = (vertex_index + 1) % section_size
            faces.append([start_center_index, next_vertex_index, vertex_index])
        start_cap_face_range = (start_cap_start, len(faces))

    end_cap_face_range: tuple[int, int] | None = None
    if cap_end:
        end_cap_start = len(faces)
        end_center_index = len(vertices)
        vertices.append(list(points[-1]))
        end_base_index = (len(sections) - 1) * section_size
        for vertex_index in range(section_size):
            next_vertex_index = (vertex_index + 1) % section_size
            faces.append([end_center_index, end_base_index + vertex_index, end_base_index + next_vertex_index])
        end_cap_face_range = (end_cap_start, len(faces))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return _orient_tube_faces_from_anchor_path(
        mesh,
        points,
        body_face_end=body_face_end,
        start_cap_face_range=start_cap_face_range,
        end_cap_face_range=end_cap_face_range,
    )


def _trim_polyline_from_end(points: list[tuple[float, float, float]], trim_distance_mm: float) -> list[tuple[float, float, float]]:
    if trim_distance_mm <= 1e-9 or len(points) < 2:
        return list(points)
    trimmed = [tuple(point) for point in points]
    remaining_trim = trim_distance_mm
    while remaining_trim > 1e-9 and len(trimmed) > 2:
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
        trimmed[-1] = (
            previous_point[0] + ((last_point[0] - previous_point[0]) * ratio),
            previous_point[1] + ((last_point[1] - previous_point[1]) * ratio),
            previous_point[2] + ((last_point[2] - previous_point[2]) * ratio),
        )
        remaining_trim = 0.0
    return trimmed


def _split_polyline_at_tail_distance(
    points: list[tuple[float, float, float]],
    tail_distance_mm: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    if len(points) < 2:
        return (list(points), list(points))
    if tail_distance_mm <= 1e-9:
        return (list(points), [points[-1]])

    total_length = 0.0
    segment_lengths: list[float] = []
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = math.dist(start_point, end_point)
        segment_lengths.append(segment_length)
        total_length += segment_length

    if tail_distance_mm >= total_length:
        return ([points[0], points[1]], list(points))

    split_distance = total_length - tail_distance_mm
    travelled = 0.0
    for segment_index, segment_length in enumerate(segment_lengths):
        next_travelled = travelled + segment_length
        if split_distance <= next_travelled and segment_length > 1e-9:
            start_point = points[segment_index]
            end_point = points[segment_index + 1]
            t_value = (split_distance - travelled) / segment_length
            split_point = (
                start_point[0] + ((end_point[0] - start_point[0]) * t_value),
                start_point[1] + ((end_point[1] - start_point[1]) * t_value),
                start_point[2] + ((end_point[2] - start_point[2]) * t_value),
            )
            head_points = list(points[: segment_index + 1]) + [split_point]
            tail_points = [split_point] + list(points[segment_index + 1 :])
            return (head_points, tail_points)
        travelled = next_travelled

    return (list(points[:-1]), [points[-2], points[-1]])


def _build_flattened_terminal_mesh(
    points: list[tuple[float, float, float]],
    *,
    wire_radius_mm: float,
    wedge_width_mm: float,
    wedge_thickness_mm: float,
    wedge_length_mm: float,
    wedge_tail_mm: float,
    side_count: int,
    leadframe_top_z: float,
    lateral_axis_xy: tuple[float, float],
    cap_start: bool = False,
) -> trimesh.Trimesh | None:
    if len(points) < 2:
        return None

    tail_direction_xy = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
    tail_direction_xy = _normalize_2d(tail_direction_xy) or (1.0, 0.0)
    geometry_points = list(points)
    if wedge_tail_mm > 1e-9:
        extension_step_mm = max(0.01, wedge_tail_mm / 4.0)
        extension_count = max(1, int(math.ceil(wedge_tail_mm / extension_step_mm)))
        for extension_index in range(1, extension_count + 1):
            extension_distance_mm = (wedge_tail_mm * extension_index) / extension_count
            geometry_points.append(
                (
                    points[-1][0] + (tail_direction_xy[0] * extension_distance_mm),
                    points[-1][1] + (tail_direction_xy[1] * extension_distance_mm),
                    points[-1][2],
                )
            )

    distances = [0.0]
    for start_point, end_point in zip(geometry_points[:-1], geometry_points[1:]):
        distances.append(distances[-1] + math.dist(start_point, end_point))
    original_length = max(1e-9, distances[len(points) - 1])
    total_length = max(1e-9, distances[-1])
    wedge_start_distance = max(0.0, original_length - wedge_length_mm)

    section_size = max(8, side_count)
    width_end_radius = wedge_width_mm / 2.0
    height_end_radius = wedge_thickness_mm / 2.0
    lateral = _vector_normalize((lateral_axis_xy[0], lateral_axis_xy[1], 0.0))
    vertical = (0.0, 0.0, 1.0)

    sections: list[list[tuple[float, float, float]]] = []
    for point_index, center in enumerate(geometry_points):
        if distances[point_index] <= wedge_start_distance:
            flatten_t = 0.0
        elif distances[point_index] >= original_length:
            flatten_t = 1.0
        else:
            flatten_t = (distances[point_index] - wedge_start_distance) / max(1e-9, original_length - wedge_start_distance)
        width_radius = wire_radius_mm + ((width_end_radius - wire_radius_mm) * flatten_t)
        height_radius = wire_radius_mm + ((height_end_radius - wire_radius_mm) * min(1.0, flatten_t * 1.35))
        if flatten_t >= 0.999:
            width_radius = width_end_radius
            height_radius = height_end_radius
        desired_center_z = leadframe_top_z + LEADFRAME_CONTACT_CLEARANCE_MM + height_radius
        blended_center = (
            center[0],
            center[1],
            (center[2] * (1.0 - flatten_t)) + (desired_center_z * flatten_t),
        )
        section: list[tuple[float, float, float]] = []
        for side_index in range(section_size):
            angle = (2.0 * math.pi * side_index) / section_size
            offset = _vector_add(
                _vector_scale(lateral, math.cos(angle) * width_radius),
                _vector_scale(vertical, math.sin(angle) * height_radius),
            )
            section.append(_vector_add(blended_center, offset))
        sections.append(section)

    vertices: list[list[float]] = []
    for section in sections:
        for vertex in section:
            vertices.append([vertex[0], vertex[1], vertex[2]])

    faces: list[list[int]] = []
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
    body_face_end = len(faces)

    start_cap_face_range: tuple[int, int] | None = None
    if cap_start:
        start_cap_start = len(faces)
        start_center_index = len(vertices)
        vertices.append(list(geometry_points[0]))
        for vertex_index in range(section_size):
            next_vertex_index = (vertex_index + 1) % section_size
            faces.append([start_center_index, next_vertex_index, vertex_index])
        start_cap_face_range = (start_cap_start, len(faces))

    end_cap_start = len(faces)
    end_center_index = len(vertices)
    vertices.append(list(geometry_points[-1]))
    end_base_index = (len(sections) - 1) * section_size
    for vertex_index in range(section_size):
        next_vertex_index = (vertex_index + 1) % section_size
        faces.append([end_center_index, end_base_index + vertex_index, end_base_index + next_vertex_index])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return _orient_tube_faces_from_anchor_path(
        mesh,
        geometry_points,
        body_face_end=body_face_end,
        start_cap_face_range=start_cap_face_range,
        end_cap_face_range=(end_cap_start, len(faces)),
    )


def _default_wire_profile(connection_id: str, leg_data: dict[str, object], die_data: dict[str, object]) -> dict[str, object]:
    return {
        "connection_id": connection_id,
        "leg_path_index": int(leg_data.get("path_index", 0)),
        "lead_side_name": str(leg_data.get("side_name", "")),
        "die_side_name": str(die_data.get("side_name", "")),
        "die_section_index": int(die_data.get("section_index", 0)),
        "arc_height_mm": 0.5,
        "arc_xy_noise_mm": 0.0,
        "wire_rise_z_mm": 0.12,
        "wire_diameter_mm": 0.03,
        "wire_arc_point_spacing_mm": 0.08,
        "wire_tube_side_count": 10,
        "ball_bond_diameter_mm": 0.12,
        "ball_bond_length_mm": 0.08,
        "ball_bond_revolution_steps": 24,
        "wedge_bond_length_mm": 0.18,
        "wedge_bond_width_mm": 0.08,
        "wedge_bond_thickness_mm": 0.02,
        "wedge_approach_run_mm": 0.18,
        "wedge_tail_mm": 0.0,
    }


def _wire_connections_from_payload(payload: dict) -> list[dict[str, object]]:
    leg_markers = []
    leadframe_top_z = max(0.05, _payload_float(payload, "thickness_z_mm", 0.2))
    for region in _bond_end_region_centers_from_payload(payload):
        center_x_mm, center_y_mm = region["center_mm"]
        leg_markers.append(
            {
                **region,
                "center_xy": (center_x_mm, center_y_mm),
                "anchor_xyz": (center_x_mm, center_y_mm, leadframe_top_z + LEADFRAME_CONTACT_CLEARANCE_MM),
            }
        )
    die_regions = []
    die_top_z = max(0.05, _payload_float(payload, "thickness_z_mm", 0.2)) + DIE_CLEARANCE_MM + max(0.02, _payload_float(payload, "silicon_die_thickness_mm", 0.15))
    for region in _bond_start_region_centers_from_payload(payload):
        center_x_mm, center_y_mm = region["center_mm"]
        die_regions.append(
            {
                **region,
                "center_xy": (center_x_mm, center_y_mm),
                "anchor_xyz": (center_x_mm, center_y_mm, die_top_z + LEADFRAME_CONTACT_CLEARANCE_MM),
            }
        )

    resolved_leg_markers = _clockwise_indexed_items(leg_markers)
    resolved_die_regions = _clockwise_indexed_items(die_regions)
    pairs = _pair_nearest_leg_regions(resolved_leg_markers, resolved_die_regions)
    existing_profiles_by_id = {}
    wire_profiles_payload = payload.get("wire_profiles", [])
    if isinstance(wire_profiles_payload, list):
        for item in wire_profiles_payload:
            if isinstance(item, dict) and str(item.get("connection_id", "")).strip():
                existing_profiles_by_id[str(item["connection_id"])] = dict(item)

    connections: list[dict[str, object]] = []
    for leg_data, die_data in pairs:
        connection_id = f"P{int(leg_data.get('path_index', 0))}_{str(die_data.get('side_name', '')).title()}_{int(die_data.get('section_index', 0))}"
        wire_profile = existing_profiles_by_id.get(connection_id, _default_wire_profile(connection_id, leg_data, die_data))
        connections.append({"connection_id": connection_id, "leg": leg_data, "die": die_data, "wire_profile": wire_profile})
    return connections


def _wire_assembly_meshes_from_payload(payload: dict, *, solid_for_boolean: bool = False) -> list[trimesh.Trimesh]:
    connection_specs = _wire_connections_from_payload(payload)
    if not connection_specs:
        return []
    bond_assembly_meshes: list[trimesh.Trimesh] = []

    for index, spec in enumerate(connection_specs):
        wire_profile = spec["wire_profile"]
        die_data = spec["die"]
        leg_data = spec["leg"]
        start = die_data.get("anchor_xyz", (0.0, 0.0, 0.0))
        end = leg_data.get("anchor_xyz", (0.0, 0.0, 0.0))

        ball_diameter_mm = max(0.02, _payload_float(wire_profile, "ball_bond_diameter_mm", 0.12))
        ball_length_mm = max(0.01, _payload_float(wire_profile, "ball_bond_length_mm", 0.08))
        revolution_steps = max(6, _payload_int(wire_profile, "ball_bond_revolution_steps", 24))
        radius_mm = ball_diameter_mm / 2.0
        arc_samples = max(8, revolution_steps // 2)
        profile_points: list[list[float]] = [[0.0, 0.0], [ball_length_mm, 0.0]]
        for sample_index in range(1, arc_samples):
            angle = (-math.pi / 2.0) + ((math.pi * sample_index) / arc_samples)
            profile_points.append([ball_length_mm + (radius_mm * math.cos(angle)), radius_mm + (radius_mm * math.sin(angle))])
        profile_points.extend([[ball_length_mm, ball_diameter_mm], [0.0, ball_diameter_mm]])
        ball_mesh = trimesh.creation.revolve(profile_points, angle=(2.0 * math.pi), cap=True, sections=revolution_steps)
        start_x, start_y, start_z = start
        ball_mesh.apply_translation([start_x, start_y, start_z])
        ball_mesh = _orient_ball_faces_from_centroid(ball_mesh)
        ball_top_z = float(ball_mesh.bounds[1][2])
        wire_start = (start_x, start_y, ball_top_z)

        end_x, end_y, end_z = end
        arc_height_mm = max(0.0, _payload_float(wire_profile, "arc_height_mm", 0.5))
        arc_xy_noise_mm = _payload_float(wire_profile, "arc_xy_noise_mm", 0.0)
        point_spacing_mm = max(0.01, _payload_float(wire_profile, "wire_arc_point_spacing_mm", 0.08))
        wire_rise_z_mm = max(0.01, _payload_float(wire_profile, "wire_rise_z_mm", 0.12))
        wedge_approach_run_mm = max(0.02, _payload_float(wire_profile, "wedge_approach_run_mm", 0.18))
        mid_x = (wire_start[0] + end_x) / 2.0
        mid_y = (wire_start[1] + end_y) / 2.0
        mid_z = max(wire_start[2], end_z) + arc_height_mm
        dx = end_x - wire_start[0]
        dy = end_y - wire_start[1]
        planar_length = math.hypot(dx, dy)
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
        rise_point = (wire_start[0], wire_start[1], wire_start[2] + wire_rise_z_mm)
        rise_control = (wire_start[0], wire_start[1], mid_z)
        landing_height = end_z + min(max(wire_rise_z_mm * 0.08, 0.004), max(arc_height_mm * 0.12, 0.004))
        landing_control = (landing_x, landing_y, landing_height)
        approximate_length = math.dist(wire_start, rise_point) + math.dist(rise_point, rise_control) + math.dist(rise_control, landing_control) + math.dist(landing_control, end)
        sample_count = max(12, int(math.ceil(approximate_length / point_spacing_mm)))
        rise_length = math.dist(wire_start, rise_point)
        rise_sample_count = max(2, int(math.ceil(rise_length / point_spacing_mm)))
        curve_sample_count = max(8, sample_count - rise_sample_count + 1)
        rise_points = [(wire_start[0], wire_start[1], wire_start[2] + ((wire_rise_z_mm * sample_index) / rise_sample_count)) for sample_index in range(rise_sample_count + 1)]
        curve_points = _sample_cubic_bezier(rise_point, rise_control, landing_control, end, curve_sample_count)
        path_points = rise_points[:-1] + [tuple(point) for point in curve_points]
        tube_side_count = max(3, _payload_int(wire_profile, "wire_tube_side_count", 10))
        wire_radius_mm = max(0.005, _payload_float(wire_profile, "wire_diameter_mm", 0.03) / 2.0)
        wedge_length_mm = max(0.04, _payload_float(wire_profile, "wedge_bond_length_mm", 0.18))
        wedge_width_mm = max(0.02, _payload_float(wire_profile, "wedge_bond_width_mm", 0.08))
        wedge_thickness_mm = max(0.005, _payload_float(wire_profile, "wedge_bond_thickness_mm", 0.02))
        terminal_length_mm = max(wedge_approach_run_mm, wedge_length_mm * 0.75)
        head_points, terminal_points = _split_polyline_at_tail_distance(path_points, terminal_length_mm)
        wire_mesh = _build_swept_tube_mesh(
            head_points,
            wire_radius_mm,
            max(8, tube_side_count),
            cap_start=solid_for_boolean,
            cap_end=solid_for_boolean,
        )

        terminal_start = terminal_points[0]
        terminal_end = terminal_points[-1]
        approach_dx = terminal_end[0] - terminal_start[0]
        approach_dy = terminal_end[1] - terminal_start[1]
        approach_length = math.hypot(approach_dx, approach_dy)
        if approach_length > 1e-9:
            # Use the incoming XY angle of this specific terminal so the flattened oval
            # is oriented from the actual approach direction instead of a side preset.
            lateral_axis_xy = (-approach_dy / approach_length, approach_dx / approach_length)
        else:
            lateral_axis_xy = (1.0, 0.0)
        terminal_mesh = _build_flattened_terminal_mesh(
            terminal_points,
            wire_radius_mm=wire_radius_mm,
            wedge_width_mm=wedge_width_mm,
            wedge_thickness_mm=wedge_thickness_mm,
            wedge_length_mm=wedge_length_mm,
            wedge_tail_mm=max(0.0, _payload_float(wire_profile, "wedge_tail_mm", 0.0)),
            side_count=max(8, tube_side_count),
            leadframe_top_z=max(0.05, _payload_float(payload, "thickness_z_mm", 0.2)),
            lateral_axis_xy=lateral_axis_xy,
            cap_start=solid_for_boolean,
        )

        component_meshes = [ball_mesh]
        if wire_mesh is not None:
            component_meshes.append(wire_mesh)
        if terminal_mesh is not None:
            component_meshes.append(terminal_mesh)
        if solid_for_boolean:
            bond_assembly_meshes.append(_ensure_outward_normals(trimesh.util.concatenate(component_meshes)))
        else:
            bond_assembly_meshes.append(_make_face_winding_consistent(trimesh.util.concatenate(component_meshes)))

    return bond_assembly_meshes


def _wire_meshes_from_payload(payload: dict) -> list[tuple[str, trimesh.Trimesh, str]]:
    bond_assembly_meshes = _wire_assembly_meshes_from_payload(payload, solid_for_boolean=False)
    if not bond_assembly_meshes:
        return []
    return [("Bond Assemblies", _make_face_winding_consistent(trimesh.util.concatenate(bond_assembly_meshes)), "#2563eb")]


def _scaled_mesh_about_center(mesh: trimesh.Trimesh, scale_percent: float) -> trimesh.Trimesh | None:
    mesh = _ensure_outward_normals(mesh)
    outward_distance_mm = max(0.0, float(mesh.scale) * (max(0.0, float(scale_percent)) / 100.0))
    if outward_distance_mm <= 1e-12:
        return None
    expanded_mesh = mesh.copy()
    try:
        vertex_normals = np.asarray(expanded_mesh.vertex_normals, dtype=float)
    except Exception:
        vertex_normals = np.zeros_like(expanded_mesh.vertices, dtype=float)
    if len(vertex_normals) != len(expanded_mesh.vertices):
        return None
    expanded_mesh.vertices = np.asarray(expanded_mesh.vertices, dtype=float) + (vertex_normals * outward_distance_mm)
    return _ensure_outward_normals(expanded_mesh)


def _subtract_mesh_from_encapsulation(
    encapsulation_mesh: trimesh.Trimesh,
    subtract_meshes: list[trimesh.Trimesh],
) -> trimesh.Trimesh:
    valid_meshes = [mesh for mesh in subtract_meshes if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    if not valid_meshes:
        return encapsulation_mesh
    try:
        result = trimesh.boolean.difference([encapsulation_mesh, *valid_meshes], engine="manifold")
        if isinstance(result, trimesh.Trimesh) and len(result.faces):
            return _ensure_outward_normals(result)
    except Exception:
        pass
    return encapsulation_mesh


def build_next_step_meshes_from_payload(payload: dict) -> list[tuple[str, trimesh.Trimesh, str]]:
    thickness_mm = max(0.05, _payload_float(payload, "thickness_z_mm", 0.2))
    width_mm = max(0.05, _payload_float(payload, "leadframe_path_width_mm", 0.8))
    meshes: list[tuple[str, trimesh.Trimesh, str]] = []
    subtract_reference_meshes: list[trimesh.Trimesh] = []

    die_size_mm = _die_square_size_from_payload(payload)
    die_mesh = _ensure_outward_normals(trimesh.creation.box(extents=(die_size_mm, die_size_mm, thickness_mm)))
    die_mesh.apply_transform(_translation_matrix(0.0, 0.0, thickness_mm / 2.0))
    meshes.append(("Centered Die Compartment", die_mesh, "#b91c1c"))

    silicon_die_width_mm, silicon_die_height_mm = _silicon_die_dimensions_from_payload(payload)
    silicon_die_thickness_mm = max(0.02, _payload_float(payload, "silicon_die_thickness_mm", 0.15))
    silicon_die_mesh = _ensure_outward_normals(trimesh.creation.box(extents=(silicon_die_width_mm, silicon_die_height_mm, silicon_die_thickness_mm)))
    silicon_die_mesh.apply_transform(_translation_matrix(0.0, 0.0, thickness_mm + DIE_CLEARANCE_MM + (silicon_die_thickness_mm / 2.0)))
    meshes.append(("Silicon Die", silicon_die_mesh, "#232323"))

    path_meshes: list[trimesh.Trimesh] = []
    for path_payload in payload.get("saved_paths_mm", []):
        if not isinstance(path_payload, list):
            continue
        path_mm = [tuple(point) for point in path_payload if isinstance(point, (list, tuple)) and len(point) == 2]
        if len(path_mm) < 2:
            continue
        try:
            stroked_polygon_mm = _build_stroked_path_polygon(path_mm, width_mm)
            path_mesh = _extrude_closed_polygon(stroked_polygon_mm, thickness_mm)
        except Exception:
            continue
        path_meshes.append(path_mesh)
    if path_meshes:
        leadframe_paths_mesh = _ensure_outward_normals(trimesh.util.concatenate(path_meshes))
        meshes.append(("Lead Frame Paths", leadframe_paths_mesh, "#14532d"))
        subtract_reference_meshes.append(leadframe_paths_mesh.copy())

    if int(payload.get("current_step_index", 0)) >= 2:
        meshes.extend(_wire_meshes_from_payload(payload))
        subtract_reference_meshes.extend(_wire_assembly_meshes_from_payload(payload, solid_for_boolean=True))
    if int(payload.get("current_step_index", 0)) >= 3:
        scale_percent = max(0.0, _payload_float(payload, "outer_model_scale_percent", 0.1))
        subtract_reference_meshes.extend([die_mesh.copy(), silicon_die_mesh.copy()])
        scaled_reference_meshes: list[trimesh.Trimesh] = []
        for base_mesh in subtract_reference_meshes:
            scaled_mesh = _scaled_mesh_about_center(base_mesh.copy(), scale_percent)
            if scaled_mesh is not None:
                scaled_reference_meshes.append(scaled_mesh)
            else:
                scaled_reference_meshes.append(base_mesh.copy())
        if scaled_reference_meshes:
            meshes.append(("Scaled Outer Model", _ensure_outward_normals(trimesh.util.concatenate(scaled_reference_meshes)), "#d97706"))
        subtract_reference_meshes = scaled_reference_meshes
    if int(payload.get("current_step_index", 0)) >= 4:
        encapsulation_width_mm, encapsulation_length_mm, negative_z_mm, positive_z_mm = _encapsulation_dimensions_from_payload(payload)
        encapsulation_height_mm = negative_z_mm + positive_z_mm
        if encapsulation_height_mm > 1e-9:
            encapsulation_mesh = _ensure_outward_normals(
                trimesh.creation.box(extents=(encapsulation_width_mm, encapsulation_length_mm, encapsulation_height_mm))
            )
            encapsulation_mesh.apply_transform(_translation_matrix(0.0, 0.0, (positive_z_mm - negative_z_mm) / 2.0))
            encapsulation_mesh = _subtract_mesh_from_encapsulation(encapsulation_mesh, subtract_reference_meshes)
            meshes.append(("Encapsulation", encapsulation_mesh, "#c08457"))
    return meshes


def _mesh_face_normal_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        face_normals = np.asarray(mesh.face_normals, dtype=float)
    except Exception:
        return None
    if len(face_normals) == 0:
        return None

    if bool(getattr(mesh, "is_watertight", False)) and bool(getattr(mesh, "is_volume", False)):
        corrected_mesh = _ensure_outward_normals(mesh)
    else:
        corrected_mesh = _make_face_winding_consistent(mesh)
    corrected_face_normals = np.asarray(corrected_mesh.face_normals, dtype=float)
    if len(face_normals) != len(corrected_face_normals):
        return None

    face_colors = np.zeros((len(face_normals), 4), dtype=np.uint8)
    for face_index, (normal_vector, corrected_normal_vector) in enumerate(zip(face_normals, corrected_face_normals)):
        if np.linalg.norm(normal_vector) <= 1e-12 or np.linalg.norm(corrected_normal_vector) <= 1e-12:
            face_colors[face_index] = [128, 128, 128, 255]
        elif float(np.dot(normal_vector, corrected_normal_vector)) >= 0.0:
            face_colors[face_index] = [37, 99, 235, 255]
        else:
            face_colors[face_index] = [220, 38, 38, 255]
    return face_colors


class IcChipGeneratorNewViewer:
    def __init__(self, payload_path: Path) -> None:
        self.payload_path = payload_path
        self.plotter = Plotter(
            title="IC Chip Generator New - Step 2 Preview",
            bg="#f7f0e4",
            bg2="#efe4cf",
            axes=1,
            size=(1200, 820),
        )
        self.info = Text2D("", pos="top-left", s=0.8, c="#3a3028", bg=None, font="Courier")
        self.actors: list = []
        self.last_payload: dict = {}
        self.show_normals = False
        self.normal_button = None
        self.last_signature: tuple | None = None

    def _payload_signature(self, payload: dict) -> tuple:
        return (
            payload.get("current_step_index", 0),
            payload.get("outer_model_scale_percent", 0.0),
            payload.get("encapsulation_width_mm", 0.0),
            payload.get("encapsulation_length_mm", 0.0),
            payload.get("encapsulation_negative_extrusion_mm", 0.0),
            payload.get("encapsulation_positive_extrusion_mm", 0.0),
            tuple(sorted(_viewer_visibility_from_payload(payload).items())),
            payload.get("outline_x_mm", 0.0),
            payload.get("outline_y_mm", 0.0),
            payload.get("thickness_z_mm", 0.0),
            payload.get("leadframe_path_width_mm", 0.0),
            payload.get("die_compartment_square_size_mm", 0.0),
            payload.get("silicon_die_width_mm", 0.0),
            payload.get("silicon_die_height_mm", 0.0),
            payload.get("silicon_die_thickness_mm", 0.0),
            payload.get("bond_end_region_size_mm", 0.0),
            payload.get("bond_end_region_offset_mm", 0.0),
            payload.get("bond_start_region_size_mm", 0.0),
            payload.get("bond_start_region_gap_mm", 0.0),
            payload.get("bond_start_region_offset_mm", 0.0),
            tuple(sorted((payload.get("bond_start_region_counts") or {}).items())) if isinstance(payload.get("bond_start_region_counts"), dict) else (),
            tuple(tuple(tuple(point) for point in path) for path in payload.get("saved_paths_mm", []) if isinstance(path, list)),
            tuple(
                (
                    item.get("connection_id", ""),
                    item.get("arc_height_mm", 0.0),
                    item.get("arc_xy_noise_mm", 0.0),
                    item.get("wire_rise_z_mm", 0.0),
                    item.get("wire_diameter_mm", 0.0),
                    item.get("wire_arc_point_spacing_mm", 0.0),
                    item.get("wire_tube_side_count", 0),
                    item.get("ball_bond_diameter_mm", 0.0),
                    item.get("ball_bond_length_mm", 0.0),
                    item.get("ball_bond_revolution_steps", 0),
                    item.get("wedge_bond_length_mm", 0.0),
                    item.get("wedge_bond_width_mm", 0.0),
                    item.get("wedge_bond_thickness_mm", 0.0),
                    item.get("wedge_approach_run_mm", 0.0),
                    item.get("wedge_tail_mm", 0.0),
                )
                for item in payload.get("wire_profiles", [])
                if isinstance(item, dict)
            ),
            payload.get("status_message", ""),
        )

    def _build_scene(self, payload: dict) -> None:
        self.last_payload = dict(payload)
        for actor in self.actors:
            self.plotter.remove(actor)
        self.actors.clear()

        summary = "Create and save at least one leadframe path before opening Step 2."
        try:
            meshes = build_next_step_meshes_from_payload(payload)
            visibility_by_label = _viewer_visibility_from_payload(payload)
            if len(meshes) > 2:
                for label, mesh, color in meshes:
                    if not visibility_by_label.get(label, True):
                        continue
                    alpha = 0.28 if label == "Scaled Outer Model" else 1.0
                    actor = Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(color).alpha(alpha)
                    if self.show_normals and label != "Scaled Outer Model":
                        face_colors = _mesh_face_normal_colors(mesh)
                        if face_colors is not None and len(face_colors) == mesh.faces.shape[0]:
                            actor.cellcolors = face_colors
                    actor.info = label
                    self.actors.append(actor)
                    self.plotter += actor
                summary = (
                    f"Lead paths: {len(payload.get('saved_paths_mm', []))}  "
                    f"Die: {payload.get('silicon_die_width_mm', 0.0):.2f} x {payload.get('silicon_die_height_mm', 0.0):.2f} x {payload.get('silicon_die_thickness_mm', 0.0):.3f} mm  "
                    f"Bond end offset: {payload.get('bond_end_region_offset_mm', 0.0):.2f} mm  "
                    f"Outer scale: {payload.get('outer_model_scale_percent', 0.0):.3f}%  "
                    f"Encap: {payload.get('encapsulation_width_mm', 0.0):.2f} x {payload.get('encapsulation_length_mm', 0.0):.2f} mm"
                )
        except Exception as exc:
            summary = f"Preview blocked: {exc}"

        self.info.text(
            "IC Chip Generator New Preview\n"
            "Live preview: leadframe, die, wire bonding, scaled outer model, and encapsulation\n"
            f"Normals: {'ON' if self.show_normals else 'OFF'}  Blue=outward  Red=inward\n"
            f"{summary}\n"
            f"{str(payload.get('status_message', '')).strip() or 'Adjust parameters in the main UI to update this preview live.'}"
        )
        self.plotter.render()

    def _toggle_normals(self, *_args) -> None:
        self.show_normals = not self.show_normals
        if self.normal_button is not None:
            self.normal_button.switch()
        if self.last_payload:
            self._build_scene(self.last_payload)
            self.last_signature = self._payload_signature(self.last_payload)

    def _on_timer(self, _event) -> None:
        payload = read_viewer_payload(self.payload_path)
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
            payload = read_viewer_payload(self.payload_path)
            if payload:
                return payload
            time.sleep(poll_interval_seconds)
        return payload

    def run(self) -> None:
        payload = self._wait_for_initial_payload()
        if not payload:
            raise RuntimeError(f"Viewer payload missing: {self.payload_path}")
        self.normal_button = self.plotter.add_button(
            self._toggle_normals,
            pos=(0.78, 0.06),
            states=("Show Normals", "Hide Normals"),
            c=("white", "white"),
            bc=("#1f4b99", "#7f1d1d"),
            font="Courier",
            size=16,
            bold=True,
        )
        self.plotter.show(self.info, zoom="tight", interactive=False)
        self._build_scene(payload)
        self.last_signature = self._payload_signature(payload)
        self.plotter.add_callback("Timer", self._on_timer)
        self.plotter.timer_callback("create", dt=150)
        self.plotter.interactive()


@dataclass(slots=True)
class GuideSlot:
    side_name: str
    index: int
    center_mm: tuple[float, float]
    start_mm: tuple[float, float]
    end_mm: tuple[float, float]


class IcChipGeneratorNewApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("IC Chip Generator New")
        self.root.geometry("1420x860")
        self.root.minsize(1180, 760)

        self.scale_px_per_mm = DEFAULT_SCALE_PX_PER_MM
        self.view_offset_px = (0.0, 0.0)
        self.is_panning = False
        self.last_pan_px: tuple[float, float] | None = None

        self.outline_x_var = tk.DoubleVar(value=10.0)
        self.outline_y_var = tk.DoubleVar(value=10.0)
        self.thickness_z_var = tk.DoubleVar(value=0.2)
        self.path_width_var = tk.DoubleVar(value=0.8)
        self.top_count_var = tk.IntVar(value=3)
        self.bottom_count_var = tk.IntVar(value=3)
        self.left_count_var = tk.IntVar(value=0)
        self.right_count_var = tk.IntVar(value=0)
        self.top_spacing_var = tk.DoubleVar(value=2.0)
        self.bottom_spacing_var = tk.DoubleVar(value=2.0)
        self.left_spacing_var = tk.DoubleVar(value=2.0)
        self.right_spacing_var = tk.DoubleVar(value=2.0)
        self.die_size_var = tk.DoubleVar(value=3.0)
        self.end_region_offset_x_var = tk.DoubleVar(value=0.0)
        self.end_region_offset_y_var = tk.DoubleVar(value=-2.0)
        self.end_region_size_var = tk.DoubleVar(value=0.8)
        self.silicon_die_width_var = tk.DoubleVar(value=2.4)
        self.silicon_die_height_var = tk.DoubleVar(value=2.4)
        self.silicon_die_thickness_var = tk.DoubleVar(value=0.15)
        self.bond_end_region_size_var = tk.DoubleVar(value=0.2)
        self.bond_end_region_offset_var = tk.DoubleVar(value=0.0)
        self.bond_start_region_size_var = tk.DoubleVar(value=0.2)
        self.bond_start_region_gap_var = tk.DoubleVar(value=0.15)
        self.bond_start_region_offset_var = tk.DoubleVar(value=0.08)
        self.bond_start_top_count_var = tk.IntVar(value=2)
        self.bond_start_bottom_count_var = tk.IntVar(value=2)
        self.bond_start_left_count_var = tk.IntVar(value=2)
        self.bond_start_right_count_var = tk.IntVar(value=2)
        self.project_name_var = tk.StringVar(value="untitled_chip")
        self.status_var = tk.StringVar(value="Set the leadframe size and side counts, then click a guide midpoint to start drawing.")
        self.step_titles = ["1. Lead Frame Design", "2. Silicon Die And Bond Regions", "3. Wire Bonding", "4. Scaled Outer Model", "5. Encapsulation"]
        self.step_index = 0
        self.current_project_dir = self._project_dir_for_name(self.project_name_var.get())
        self.viewer_process: subprocess.Popen | None = None
        self.saved_paths_mm: list[list[tuple[float, float]]] = []
        self.current_path_mm: list[tuple[float, float]] = []
        self.preview_point_mm: tuple[float, float] | None = None
        self.dragging_vertex: tuple[int, int] | None = None
        self.wire_profiles: list[dict[str, object]] = []
        self.loading_wire_profile = False
        self.syncing_wire_profiles = False
        self.selected_wire_profile_id_var = tk.StringVar(value="")
        self.wire_arc_height_var = tk.DoubleVar(value=0.5)
        self.wire_arc_xy_noise_var = tk.DoubleVar(value=0.0)
        self.wire_rise_z_var = tk.DoubleVar(value=0.12)
        self.wire_diameter_var = tk.DoubleVar(value=0.03)
        self.wire_point_spacing_var = tk.DoubleVar(value=0.08)
        self.wire_tube_side_count_var = tk.IntVar(value=10)
        self.ball_bond_diameter_var = tk.DoubleVar(value=0.12)
        self.ball_bond_length_var = tk.DoubleVar(value=0.08)
        self.ball_bond_revolution_steps_var = tk.IntVar(value=24)
        self.wedge_bond_length_var = tk.DoubleVar(value=0.18)
        self.wedge_bond_width_var = tk.DoubleVar(value=0.08)
        self.wedge_bond_thickness_var = tk.DoubleVar(value=0.02)
        self.wedge_approach_run_var = tk.DoubleVar(value=0.18)
        self.wedge_tail_var = tk.DoubleVar(value=0.0)
        self.outer_model_scale_percent_var = tk.DoubleVar(value=0.1)
        self.encapsulation_width_var = tk.DoubleVar(value=7.0)
        self.encapsulation_length_var = tk.DoubleVar(value=7.0)
        self.encapsulation_negative_extrusion_var = tk.DoubleVar(value=0.0)
        self.encapsulation_positive_extrusion_var = tk.DoubleVar(value=0.6)
        self.show_leadframe_paths_var = tk.BooleanVar(value=True)
        self.show_centered_die_compartment_var = tk.BooleanVar(value=False)
        self.show_silicon_die_var = tk.BooleanVar(value=True)
        self.show_bond_assemblies_var = tk.BooleanVar(value=True)
        self.show_scaled_outer_model_var = tk.BooleanVar(value=True)
        self.show_encapsulation_var = tk.BooleanVar(value=True)
        self.rand_ball_radius_mean_var = tk.DoubleVar(value=0.06)
        self.rand_ball_radius_std_var = tk.DoubleVar(value=0.01)
        self.rand_ball_height_mean_var = tk.DoubleVar(value=0.08)
        self.rand_ball_height_std_var = tk.DoubleVar(value=0.015)
        self.rand_arc_height_mean_var = tk.DoubleVar(value=0.5)
        self.rand_arc_height_std_var = tk.DoubleVar(value=0.08)
        self.rand_arc_xy_noise_mean_var = tk.DoubleVar(value=0.0)
        self.rand_arc_xy_noise_std_var = tk.DoubleVar(value=0.05)
        self.rand_wire_rise_mean_var = tk.DoubleVar(value=0.12)
        self.rand_wire_rise_std_var = tk.DoubleVar(value=0.02)
        self.rand_wire_diameter_mean_var = tk.DoubleVar(value=0.03)
        self.rand_wire_diameter_std_var = tk.DoubleVar(value=0.004)
        self.rand_wedge_length_mean_var = tk.DoubleVar(value=0.18)
        self.rand_wedge_length_std_var = tk.DoubleVar(value=0.02)
        self.rand_wedge_width_mean_var = tk.DoubleVar(value=0.08)
        self.rand_wedge_width_std_var = tk.DoubleVar(value=0.01)
        self.rand_wedge_thickness_mean_var = tk.DoubleVar(value=0.02)
        self.rand_wedge_thickness_std_var = tk.DoubleVar(value=0.003)
        self.rand_wedge_approach_mean_var = tk.DoubleVar(value=0.18)
        self.rand_wedge_approach_std_var = tk.DoubleVar(value=0.02)
        self.rand_wedge_tail_mean_var = tk.DoubleVar(value=0.0)
        self.rand_wedge_tail_std_var = tk.DoubleVar(value=0.02)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._bind_live_refresh()
        self._ensure_current_project_dir()
        self._show_step()
        self._redraw_canvas()
        self._autosave_project_files()

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        left_host = ttk.Frame(root_frame, width=380)
        left_host.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        left_host.columnconfigure(0, weight=1)
        left_host.rowconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_host, width=360, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_host, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        left = ttk.Frame(left_canvas, padding=(0, 0, 8, 0))
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        right = ttk.Frame(root_frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        def on_left_frame_configure(_event) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def on_left_canvas_configure(event) -> None:
            left_canvas.itemconfigure(left_window, width=event.width)

        def on_left_mousewheel(event) -> str:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta != 0:
                left_canvas.yview_scroll(delta, "units")
            return "break"

        left.bind("<Configure>", on_left_frame_configure)
        left_canvas.bind("<Configure>", on_left_canvas_configure)
        left_canvas.bind("<MouseWheel>", on_left_mousewheel)
        left_canvas.bind("<Button-4>", on_left_mousewheel)
        left_canvas.bind("<Button-5>", on_left_mousewheel)

        ttk.Label(left, text="IC Chip Generator", font=("Georgia", 18, "bold")).pack(anchor="w")
        ttk.Label(left, text="New multi-step IC package generator", font=("Georgia", 12, "bold")).pack(anchor="w", pady=(2, 10))
        ttk.Label(
            left,
            text="Build the package step by step: leadframe, die placement, wire bonding, and a scaled outer model preview.",
            wraplength=320,
        ).pack(anchor="w")
        self.step_label = ttk.Label(left, text=self.step_titles[self.step_index], font=("Georgia", 13, "bold"))
        self.step_label.pack(anchor="w", pady=(10, 8))
        nav_row = ttk.Frame(left)
        nav_row.pack(fill="x", pady=(0, 10))
        ttk.Button(nav_row, text="Previous", command=self._previous_step).pack(side="left", fill="x", expand=True)
        ttk.Button(nav_row, text="Next", command=self._next_step).pack(side="left", fill="x", expand=True, padx=(8, 0))

        project_box = ttk.LabelFrame(left, text="Project", padding=10)
        project_box.pack(fill="x", pady=(12, 0))
        ttk.Label(project_box, text="Project Name").pack(anchor="w")
        ttk.Entry(project_box, textvariable=self.project_name_var).pack(fill="x", pady=(2, 8))
        ttk.Button(project_box, text="Create / Switch Project Folder", command=self._create_or_switch_project_folder).pack(fill="x")
        ttk.Button(project_box, text="Load Project Folder", command=self._prompt_load_project_folder).pack(fill="x", pady=(8, 0))

        self.step_sections: list[ttk.Frame] = []

        step1_section = ttk.Frame(left)
        self.step_sections.append(step1_section)
        frame_box = ttk.LabelFrame(step1_section, text="Leadframe Parameters", padding=10)
        frame_box.pack(fill="x", pady=(12, 0))
        ttk.Label(frame_box, text="Outline X / Y (mm)").pack(anchor="w")
        xy_row = ttk.Frame(frame_box)
        xy_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(xy_row, textvariable=self.outline_x_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(xy_row, textvariable=self.outline_y_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(frame_box, text="Thickness Z (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_box, textvariable=self.thickness_z_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame_box, text="Leadframe Path Width (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_box, textvariable=self.path_width_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame_box, text="Centered Die Compartment Size (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_box, textvariable=self.die_size_var).pack(fill="x", pady=(2, 0))
        ttk.Label(frame_box, text="Ending Region Offset X / Y (mm)").pack(anchor="w", pady=(10, 0))
        end_offset_row = ttk.Frame(frame_box)
        end_offset_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(end_offset_row, textvariable=self.end_region_offset_x_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(end_offset_row, textvariable=self.end_region_offset_y_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(frame_box, text="Ending Region Size (mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame_box, textvariable=self.end_region_size_var).pack(fill="x", pady=(2, 0))

        pin_box = ttk.LabelFrame(step1_section, text="Pins Per Side", padding=10)
        pin_box.pack(fill="x", pady=(12, 0))
        grid = ttk.Frame(pin_box)
        grid.pack(fill="x")
        ttk.Label(grid, text="Side").grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Count").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(grid, text="Gap (mm)").grid(row=0, column=2, sticky="w", padx=(8, 0))
        rows = [
            ("TOP", self.top_count_var, self.top_spacing_var),
            ("BOTTOM", self.bottom_count_var, self.bottom_spacing_var),
            ("LEFT", self.left_count_var, self.left_spacing_var),
            ("RIGHT", self.right_count_var, self.right_spacing_var),
        ]
        for row_index, (label, count_var, spacing_var) in enumerate(rows, start=1):
            ttk.Label(grid, text=label).grid(row=row_index, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(grid, textvariable=count_var, width=10).grid(row=row_index, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
            ttk.Entry(grid, textvariable=spacing_var, width=10).grid(row=row_index, column=2, sticky="ew", padx=(8, 0), pady=(6, 0))

        help_box = ttk.LabelFrame(step1_section, text="Canvas Controls", padding=10)
        help_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            help_box,
            text="Click a yellow guide midpoint to start a path. After the first point, clicks snap to 45-degree angles from the previous point. If the cursor gets close to the ending region, the path snaps there and finishes. Drag saved vertices to adjust. Middle or right mouse pans. Mouse wheel zooms.",
            wraplength=310,
        ).pack(anchor="w")
        ttk.Button(help_box, text="Finish Current Path", command=self._finish_current_path).pack(fill="x", pady=(10, 0))
        ttk.Button(help_box, text="Undo Point", command=self._undo_point).pack(fill="x", pady=(8, 0))
        ttk.Button(help_box, text="Mirror Horizontal", command=self._mirror_paths_horizontal).pack(fill="x", pady=(8, 0))
        ttk.Button(help_box, text="Mirror Vertical", command=self._mirror_paths_vertical).pack(fill="x", pady=(8, 0))
        ttk.Button(help_box, text="Clear Current Path", command=self._clear_current_path).pack(fill="x", pady=(8, 0))
        ttk.Button(help_box, text="Clear Saved Paths", command=self._clear_saved_paths).pack(fill="x", pady=(8, 0))

        save_box = ttk.LabelFrame(step1_section, text="Save", padding=10)
        save_box.pack(fill="x", pady=(12, 0))
        ttk.Button(save_box, text="Export Step 1 JSON", command=self._save_json).pack(fill="x")

        step2_section = ttk.Frame(left)
        self.step_sections.append(step2_section)
        step2_box = ttk.LabelFrame(step2_section, text="Step 2: Silicon Die And Bond Regions", padding=10)
        step2_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            step2_box,
            text="These parameters stay in the same UI. Move to Step 2 when you want to preview the silicon die, bond end regions, and die-side bond start regions.",
            wraplength=310,
        ).pack(anchor="w")
        die_box = ttk.LabelFrame(step2_box, text="Silicon Die", padding=8)
        die_box.pack(fill="x", pady=(10, 0))
        ttk.Label(die_box, text="The die is centered over the compartment and held 0.001 mm above the leadframe.").pack(anchor="w")
        ttk.Label(die_box, text="Die Width / Height (mm)").pack(anchor="w", pady=(8, 0))
        die_xy_row = ttk.Frame(die_box)
        die_xy_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(die_xy_row, textvariable=self.silicon_die_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(die_xy_row, textvariable=self.silicon_die_height_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(die_box, text="Die Thickness (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(die_box, textvariable=self.silicon_die_thickness_var).pack(fill="x", pady=(2, 0))

        bond_end_box = ttk.LabelFrame(step2_box, text="Bond End Regions", padding=8)
        bond_end_box.pack(fill="x", pady=(10, 0))
        ttk.Label(bond_end_box, text="One square region is placed back along each saved copper path near the die side.").pack(anchor="w")
        ttk.Label(bond_end_box, text="Square Size (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(bond_end_box, textvariable=self.bond_end_region_size_var).pack(fill="x", pady=(2, 0))
        ttk.Label(bond_end_box, text="Offset Away From Die (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(bond_end_box, textvariable=self.bond_end_region_offset_var).pack(fill="x", pady=(2, 0))

        bond_start_box = ttk.LabelFrame(step2_box, text="Bond Start Regions", padding=8)
        bond_start_box.pack(fill="x", pady=(10, 0))
        ttk.Label(bond_start_box, text="Each bond start region is a square area arranged along the die edges.").pack(anchor="w")
        ttk.Label(bond_start_box, text="Square Size (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(bond_start_box, textvariable=self.bond_start_region_size_var).pack(fill="x", pady=(2, 0))
        ttk.Label(bond_start_box, text="Distance Between Regions (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(bond_start_box, textvariable=self.bond_start_region_gap_var).pack(fill="x", pady=(2, 0))
        ttk.Label(bond_start_box, text="Offset From Die Edge (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(bond_start_box, textvariable=self.bond_start_region_offset_var).pack(fill="x", pady=(2, 0))
        count_box = ttk.LabelFrame(bond_start_box, text="Regions Per Side", padding=8)
        count_box.pack(fill="x", pady=(10, 0))
        rows = [
            ("Top", self.bond_start_top_count_var),
            ("Bottom", self.bond_start_bottom_count_var),
            ("Left", self.bond_start_left_count_var),
            ("Right", self.bond_start_right_count_var),
        ]
        for row_index, (label, variable) in enumerate(rows):
            ttk.Label(count_box, text=label).grid(row=row_index, column=0, sticky="w", pady=(6 if row_index else 0, 0))
            ttk.Entry(count_box, textvariable=variable, width=10).grid(row=row_index, column=1, sticky="ew", padx=(8, 0), pady=(6 if row_index else 0, 0))

        step3_section = ttk.Frame(left)
        self.step_sections.append(step3_section)
        step3_box = ttk.LabelFrame(step3_section, text="Step 3: Wire Bonding", padding=10)
        step3_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            step3_box,
            text="Each wire starts with a ball bond on the die-side region and ends with a wedge bond on the leadframe-side region. Every wire keeps its own geometry settings.",
            wraplength=310,
        ).pack(anchor="w")
        ttk.Button(step3_box, text="Refresh Wire Pairing", command=self._refresh_wire_profiles).pack(fill="x", pady=(10, 0))
        ttk.Label(step3_box, text="Selected Wire").pack(anchor="w", pady=(10, 0))
        self.wire_selector = ttk.Combobox(step3_box, textvariable=self.selected_wire_profile_id_var, state="readonly")
        self.wire_selector.pack(fill="x", pady=(2, 0))
        self.wire_selector.bind("<<ComboboxSelected>>", self._on_wire_profile_selected)

        wire_shape_box = ttk.LabelFrame(step3_box, text="Wire Curve", padding=8)
        wire_shape_box.pack(fill="x", pady=(10, 0))
        ttk.Label(wire_shape_box, text="Arc Height (mm)").pack(anchor="w")
        ttk.Entry(wire_shape_box, textvariable=self.wire_arc_height_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wire_shape_box, text="Side Curvature / XY Noise (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wire_shape_box, textvariable=self.wire_arc_xy_noise_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wire_shape_box, text="Initial Rise Z (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wire_shape_box, textvariable=self.wire_rise_z_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wire_shape_box, text="Point Spacing (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wire_shape_box, textvariable=self.wire_point_spacing_var).pack(fill="x", pady=(2, 0))

        wire_body_box = ttk.LabelFrame(step3_box, text="Wire Body", padding=8)
        wire_body_box.pack(fill="x", pady=(10, 0))
        ttk.Label(wire_body_box, text="Wire Diameter (mm)").pack(anchor="w")
        ttk.Entry(wire_body_box, textvariable=self.wire_diameter_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wire_body_box, text="Tube Side Count").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wire_body_box, textvariable=self.wire_tube_side_count_var).pack(fill="x", pady=(2, 0))

        ball_box = ttk.LabelFrame(step3_box, text="Ball Bond At Die Start", padding=8)
        ball_box.pack(fill="x", pady=(10, 0))
        ttk.Label(ball_box, text="Ball Diameter (mm)").pack(anchor="w")
        ttk.Entry(ball_box, textvariable=self.ball_bond_diameter_var).pack(fill="x", pady=(2, 0))
        ttk.Label(ball_box, text="Ball Length (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(ball_box, textvariable=self.ball_bond_length_var).pack(fill="x", pady=(2, 0))
        ttk.Label(ball_box, text="Revolution Steps").pack(anchor="w", pady=(8, 0))
        ttk.Entry(ball_box, textvariable=self.ball_bond_revolution_steps_var).pack(fill="x", pady=(2, 0))

        wedge_box = ttk.LabelFrame(step3_box, text="Wedge Bond At Lead End", padding=8)
        wedge_box.pack(fill="x", pady=(10, 0))
        ttk.Label(wedge_box, text="Wedge Length (mm)").pack(anchor="w")
        ttk.Entry(wedge_box, textvariable=self.wedge_bond_length_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wedge_box, text="Wedge Width (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wedge_box, textvariable=self.wedge_bond_width_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wedge_box, text="Wedge Thickness (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wedge_box, textvariable=self.wedge_bond_thickness_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wedge_box, text="Approach Run (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wedge_box, textvariable=self.wedge_approach_run_var).pack(fill="x", pady=(2, 0))
        ttk.Label(wedge_box, text="Wedge Tail (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(wedge_box, textvariable=self.wedge_tail_var).pack(fill="x", pady=(2, 0))

        random_box = ttk.LabelFrame(step3_box, text="Randomize Wires", padding=8)
        random_box.pack(fill="x", pady=(10, 0))

        def add_mean_std_row(parent: ttk.Frame, label: str, mean_var: tk.Variable, std_var: tk.Variable) -> None:
            ttk.Label(parent, text=label).pack(anchor="w")
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=(2, 6))
            ttk.Entry(row, textvariable=mean_var, width=8).pack(side="left", fill="x", expand=True)
            ttk.Entry(row, textvariable=std_var, width=8).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Label(random_box, text="Mean / Std Dev per parameter, then randomize all wires.").pack(anchor="w")
        add_mean_std_row(random_box, "Ball Radius (mm)", self.rand_ball_radius_mean_var, self.rand_ball_radius_std_var)
        add_mean_std_row(random_box, "Ball Height (mm)", self.rand_ball_height_mean_var, self.rand_ball_height_std_var)
        add_mean_std_row(random_box, "Arc Height (mm)", self.rand_arc_height_mean_var, self.rand_arc_height_std_var)
        add_mean_std_row(random_box, "Side Curvature (mm)", self.rand_arc_xy_noise_mean_var, self.rand_arc_xy_noise_std_var)
        add_mean_std_row(random_box, "Initial Rise Z (mm)", self.rand_wire_rise_mean_var, self.rand_wire_rise_std_var)
        add_mean_std_row(random_box, "Wire Diameter (mm)", self.rand_wire_diameter_mean_var, self.rand_wire_diameter_std_var)
        add_mean_std_row(random_box, "Wedge Length (mm)", self.rand_wedge_length_mean_var, self.rand_wedge_length_std_var)
        add_mean_std_row(random_box, "Wedge Width (mm)", self.rand_wedge_width_mean_var, self.rand_wedge_width_std_var)
        add_mean_std_row(random_box, "Wedge Thickness (mm)", self.rand_wedge_thickness_mean_var, self.rand_wedge_thickness_std_var)
        add_mean_std_row(random_box, "Approach Run (mm)", self.rand_wedge_approach_mean_var, self.rand_wedge_approach_std_var)
        add_mean_std_row(random_box, "Wedge Tail (mm)", self.rand_wedge_tail_mean_var, self.rand_wedge_tail_std_var)
        random_button_row = ttk.Frame(random_box)
        random_button_row.pack(fill="x", pady=(8, 0))
        ttk.Button(random_button_row, text="Randomize", command=self._randomize_wire_profiles).pack(side="left", fill="x", expand=True)
        ttk.Button(random_button_row, text="Show Average", command=lambda: self._apply_wire_profile_distribution_view("mean")).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(random_button_row, text="Show Lowest (5%)", command=lambda: self._apply_wire_profile_distribution_view("p05")).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(random_button_row, text="Show Highest (95%)", command=lambda: self._apply_wire_profile_distribution_view("p95")).pack(side="left", fill="x", expand=True, padx=(6, 0))

        step4_section = ttk.Frame(left)
        self.step_sections.append(step4_section)
        step4_box = ttk.LabelFrame(step4_section, text="Step 4: Scaled Outer Model", padding=10)
        step4_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            step4_box,
            text="Create an enlarged outer copy of the full assembly by pushing its surfaces outward. The original geometry stays visible inside the expanded model in the live 3D preview.",
            wraplength=310,
        ).pack(anchor="w")
        ttk.Label(step4_box, text="Outer Model Expansion (%)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(step4_box, textvariable=self.outer_model_scale_percent_var).pack(fill="x", pady=(2, 0))
        ttk.Label(
            step4_box,
            text="Example: 0.1 means the outer model grows outward by a small amount derived from 0.1% of the full assembly size.",
            wraplength=310,
        ).pack(anchor="w", pady=(8, 0))

        step5_section = ttk.Frame(left)
        self.step_sections.append(step5_section)
        step5_box = ttk.LabelFrame(step5_section, text="Step 5: Encapsulation", padding=10)
        step5_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            step5_box,
            text="The encapsulation is centered on the origin. The canvas shows the encapsulation footprint, and the 3D preview extrudes it upward and downward from Z=0.",
            wraplength=310,
        ).pack(anchor="w")
        ttk.Label(step5_box, text="Width X / Length Y (mm)").pack(anchor="w", pady=(10, 0))
        encaps_xy_row = ttk.Frame(step5_box)
        encaps_xy_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(encaps_xy_row, textvariable=self.encapsulation_width_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(encaps_xy_row, textvariable=self.encapsulation_length_var, width=8).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(step5_box, text="Negative Extrusion (-Z mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(step5_box, textvariable=self.encapsulation_negative_extrusion_var).pack(fill="x", pady=(2, 0))
        ttk.Label(step5_box, text="Positive Extrusion (+Z mm)").pack(anchor="w", pady=(10, 0))
        ttk.Entry(step5_box, textvariable=self.encapsulation_positive_extrusion_var).pack(fill="x", pady=(2, 0))
        visibility_box = ttk.LabelFrame(step5_box, text="Viewer Visibility", padding=8)
        visibility_box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(visibility_box, text="Lead Frame Paths", variable=self.show_leadframe_paths_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Centered Die Compartment", variable=self.show_centered_die_compartment_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Silicon Die", variable=self.show_silicon_die_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Bond Assemblies", variable=self.show_bond_assemblies_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Scaled Outer Model", variable=self.show_scaled_outer_model_var).pack(anchor="w")
        ttk.Checkbutton(visibility_box, text="Encapsulation", variable=self.show_encapsulation_var).pack(anchor="w")

        ttk.Label(left, textvariable=self.status_var, wraplength=320).pack(anchor="w", pady=(12, 0))

        canvas_header = ttk.Frame(right)
        canvas_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        canvas_header.columnconfigure(0, weight=1)
        ttk.Label(canvas_header, text="Leadframe Canvas", font=("Georgia", 15, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(canvas_header, text="Next Step", command=self._next_step).grid(row=0, column=1, sticky="e")
        self.canvas = tk.Canvas(
            right,
            width=DEFAULT_CANVAS_WIDTH,
            height=DEFAULT_CANVAS_HEIGHT,
            bg=CANVAS_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#bca88d",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)

    def _bind_live_refresh(self) -> None:
        for variable in (
            self.outline_x_var,
            self.outline_y_var,
            self.thickness_z_var,
            self.path_width_var,
            self.top_count_var,
            self.bottom_count_var,
            self.left_count_var,
            self.right_count_var,
            self.top_spacing_var,
            self.bottom_spacing_var,
            self.left_spacing_var,
            self.right_spacing_var,
            self.die_size_var,
            self.end_region_offset_x_var,
            self.end_region_offset_y_var,
            self.end_region_size_var,
            self.silicon_die_width_var,
            self.silicon_die_height_var,
            self.silicon_die_thickness_var,
            self.bond_end_region_size_var,
            self.bond_end_region_offset_var,
            self.bond_start_region_size_var,
            self.bond_start_region_gap_var,
            self.bond_start_region_offset_var,
            self.bond_start_top_count_var,
            self.bond_start_bottom_count_var,
            self.bond_start_left_count_var,
            self.bond_start_right_count_var,
            self.wire_arc_height_var,
            self.wire_arc_xy_noise_var,
            self.wire_rise_z_var,
            self.wire_diameter_var,
            self.wire_point_spacing_var,
            self.wire_tube_side_count_var,
            self.ball_bond_diameter_var,
            self.ball_bond_length_var,
            self.ball_bond_revolution_steps_var,
            self.wedge_bond_length_var,
            self.wedge_bond_width_var,
            self.wedge_bond_thickness_var,
            self.wedge_approach_run_var,
            self.wedge_tail_var,
            self.outer_model_scale_percent_var,
            self.encapsulation_width_var,
            self.encapsulation_length_var,
            self.encapsulation_negative_extrusion_var,
            self.encapsulation_positive_extrusion_var,
            self.show_leadframe_paths_var,
            self.show_centered_die_compartment_var,
            self.show_silicon_die_var,
            self.show_bond_assemblies_var,
            self.show_scaled_outer_model_var,
            self.show_encapsulation_var,
        ):
            variable.trace_add("write", self._on_live_value_change)

    def _on_live_value_change(self, *_args) -> None:
        self._apply_wire_profile_editor()
        if (self.step_index >= 2 or self.wire_profiles) and not self.syncing_wire_profiles:
            self._refresh_wire_profiles()
        self._redraw_canvas()
        self._autosave_project_files()
        self._push_viewer_payload()

    def _safe_float(self, variable: tk.Variable, fallback: float) -> float:
        try:
            value = variable.get()
        except tk.TclError:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _safe_int(self, variable: tk.Variable, fallback: int) -> int:
        try:
            value = variable.get()
        except tk.TclError:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _safe_float_from_value(self, value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _safe_int_from_value(self, value, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _project_dir_for_name(self, project_name: str) -> Path:
        return PROJECTS_ROOT_DIR / _safe_stage_slug(project_name)

    def _project_snapshot_path(self, project_dir: Path | None = None) -> Path:
        target_dir = project_dir if project_dir is not None else self.current_project_dir
        return target_dir / "ic_chip_generator_New_project.json"

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
        self._ensure_current_project_dir()
        self._autosave_project_files()
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

        self.project_name_var.set(str(payload.get("project_name", project_dir.name)).strip() or project_dir.name)
        self.current_project_dir = project_dir
        self.outline_x_var.set(self._safe_float_from_value(payload.get("outline_x_mm"), 10.0))
        self.outline_y_var.set(self._safe_float_from_value(payload.get("outline_y_mm"), 10.0))
        self.thickness_z_var.set(self._safe_float_from_value(payload.get("thickness_z_mm"), 0.2))
        self.path_width_var.set(self._safe_float_from_value(payload.get("leadframe_path_width_mm"), 0.8))
        self.die_size_var.set(self._safe_float_from_value(payload.get("die_compartment_square_size_mm"), 3.0))
        self.end_region_offset_x_var.set(self._safe_float_from_value(payload.get("end_region_offset_x_mm"), 0.0))
        self.end_region_offset_y_var.set(self._safe_float_from_value(payload.get("end_region_offset_y_mm"), -2.0))
        self.end_region_size_var.set(self._safe_float_from_value(payload.get("end_region_size_mm"), 0.8))
        self.outer_model_scale_percent_var.set(self._safe_float_from_value(payload.get("outer_model_scale_percent"), 0.1))
        self.encapsulation_width_var.set(self._safe_float_from_value(payload.get("encapsulation_width_mm"), self._outline_dims()[0] * 0.7))
        self.encapsulation_length_var.set(self._safe_float_from_value(payload.get("encapsulation_length_mm"), self._outline_dims()[1] * 0.7))
        self.encapsulation_negative_extrusion_var.set(self._safe_float_from_value(payload.get("encapsulation_negative_extrusion_mm"), 0.0))
        self.encapsulation_positive_extrusion_var.set(self._safe_float_from_value(payload.get("encapsulation_positive_extrusion_mm"), 0.6))
        viewer_visibility = payload.get("step5_viewer_visibility", {})
        if isinstance(viewer_visibility, dict):
            self.show_leadframe_paths_var.set(bool(viewer_visibility.get("lead_frame_paths", True)))
            self.show_centered_die_compartment_var.set(bool(viewer_visibility.get("centered_die_compartment", False)))
            self.show_silicon_die_var.set(bool(viewer_visibility.get("silicon_die", True)))
            self.show_bond_assemblies_var.set(bool(viewer_visibility.get("bond_assemblies", True)))
            self.show_scaled_outer_model_var.set(bool(viewer_visibility.get("scaled_outer_model", True)))
            self.show_encapsulation_var.set(bool(viewer_visibility.get("encapsulation", True)))
        self.silicon_die_thickness_var.set(self._safe_float_from_value(payload.get("silicon_die_thickness_mm"), 0.15))
        self.silicon_die_width_var.set(self._safe_float_from_value(payload.get("silicon_die_width_mm"), self._die_square_size_mm() * 0.8))
        self.silicon_die_height_var.set(self._safe_float_from_value(payload.get("silicon_die_height_mm"), self._die_square_size_mm() * 0.8))
        self.bond_end_region_size_var.set(self._safe_float_from_value(payload.get("bond_end_region_size_mm"), 0.2))
        self.bond_end_region_offset_var.set(self._safe_float_from_value(payload.get("bond_end_region_offset_mm"), 0.0))
        self.bond_start_region_size_var.set(self._safe_float_from_value(payload.get("bond_start_region_size_mm"), 0.2))
        self.bond_start_region_gap_var.set(self._safe_float_from_value(payload.get("bond_start_region_gap_mm"), 0.15))
        self.bond_start_region_offset_var.set(self._safe_float_from_value(payload.get("bond_start_region_offset_mm"), 0.08))
        bond_start_region_counts = payload.get("bond_start_region_counts", {})
        if isinstance(bond_start_region_counts, dict):
            self.bond_start_top_count_var.set(self._safe_int_from_value(bond_start_region_counts.get("top"), 2))
            self.bond_start_bottom_count_var.set(self._safe_int_from_value(bond_start_region_counts.get("bottom"), 2))
            self.bond_start_left_count_var.set(self._safe_int_from_value(bond_start_region_counts.get("left"), 2))
            self.bond_start_right_count_var.set(self._safe_int_from_value(bond_start_region_counts.get("right"), 2))

        pins_per_side = payload.get("pins_per_side", {})
        if isinstance(pins_per_side, dict):
            self.top_count_var.set(self._safe_int_from_value(pins_per_side.get("top"), 3))
            self.bottom_count_var.set(self._safe_int_from_value(pins_per_side.get("bottom"), 3))
            self.left_count_var.set(self._safe_int_from_value(pins_per_side.get("left"), 0))
            self.right_count_var.set(self._safe_int_from_value(pins_per_side.get("right"), 0))

        leg_spacing = payload.get("leg_spacing_mm", {})
        if isinstance(leg_spacing, dict):
            self.top_spacing_var.set(self._safe_float_from_value(leg_spacing.get("top"), 2.0))
            self.bottom_spacing_var.set(self._safe_float_from_value(leg_spacing.get("bottom"), 2.0))
            self.left_spacing_var.set(self._safe_float_from_value(leg_spacing.get("left"), 2.0))
            self.right_spacing_var.set(self._safe_float_from_value(leg_spacing.get("right"), 2.0))

        self.saved_paths_mm = [
            [tuple(point) for point in path if isinstance(point, (list, tuple)) and len(point) == 2]
            for path in payload.get("saved_paths_mm", [])
            if isinstance(path, list)
        ]
        self.saved_paths_mm = [path for path in self.saved_paths_mm if path]
        self.current_path_mm = [
            tuple(point)
            for point in payload.get("current_path_mm", [])
            if isinstance(point, (list, tuple)) and len(point) == 2
        ]
        randomization = payload.get("wire_randomization", {})
        if isinstance(randomization, dict):
            self.rand_ball_radius_mean_var.set(self._safe_float_from_value(randomization.get("ball_radius_mean_mm"), 0.06))
            self.rand_ball_radius_std_var.set(self._safe_float_from_value(randomization.get("ball_radius_std_mm"), 0.01))
            self.rand_ball_height_mean_var.set(self._safe_float_from_value(randomization.get("ball_height_mean_mm"), 0.08))
            self.rand_ball_height_std_var.set(self._safe_float_from_value(randomization.get("ball_height_std_mm"), 0.015))
            self.rand_arc_height_mean_var.set(self._safe_float_from_value(randomization.get("arc_height_mean_mm"), 0.5))
            self.rand_arc_height_std_var.set(self._safe_float_from_value(randomization.get("arc_height_std_mm"), 0.08))
            self.rand_arc_xy_noise_mean_var.set(self._safe_float_from_value(randomization.get("arc_xy_noise_mean_mm"), 0.0))
            self.rand_arc_xy_noise_std_var.set(self._safe_float_from_value(randomization.get("arc_xy_noise_std_mm"), 0.05))
            self.rand_wire_rise_mean_var.set(self._safe_float_from_value(randomization.get("wire_rise_mean_mm"), 0.12))
            self.rand_wire_rise_std_var.set(self._safe_float_from_value(randomization.get("wire_rise_std_mm"), 0.02))
            self.rand_wire_diameter_mean_var.set(self._safe_float_from_value(randomization.get("wire_diameter_mean_mm"), 0.03))
            self.rand_wire_diameter_std_var.set(self._safe_float_from_value(randomization.get("wire_diameter_std_mm"), 0.004))
            self.rand_wedge_length_mean_var.set(self._safe_float_from_value(randomization.get("wedge_length_mean_mm"), 0.18))
            self.rand_wedge_length_std_var.set(self._safe_float_from_value(randomization.get("wedge_length_std_mm"), 0.02))
            self.rand_wedge_width_mean_var.set(self._safe_float_from_value(randomization.get("wedge_width_mean_mm"), 0.08))
            self.rand_wedge_width_std_var.set(self._safe_float_from_value(randomization.get("wedge_width_std_mm"), 0.01))
            self.rand_wedge_thickness_mean_var.set(self._safe_float_from_value(randomization.get("wedge_thickness_mean_mm"), 0.02))
            self.rand_wedge_thickness_std_var.set(self._safe_float_from_value(randomization.get("wedge_thickness_std_mm"), 0.003))
            self.rand_wedge_approach_mean_var.set(self._safe_float_from_value(randomization.get("wedge_approach_mean_mm"), 0.18))
            self.rand_wedge_approach_std_var.set(self._safe_float_from_value(randomization.get("wedge_approach_std_mm"), 0.02))
            self.rand_wedge_tail_mean_var.set(self._safe_float_from_value(randomization.get("wedge_tail_mean_mm"), 0.0))
            self.rand_wedge_tail_std_var.set(self._safe_float_from_value(randomization.get("wedge_tail_std_mm"), 0.02))
            self.wire_profiles = [dict(item) for item in payload.get("wire_profiles", []) if isinstance(item, dict)]
        preview_point = payload.get("preview_point_mm")
        self.preview_point_mm = tuple(preview_point) if isinstance(preview_point, (list, tuple)) and len(preview_point) == 2 else None
        self.step_index = max(0, min(len(self.step_titles) - 1, self._safe_int_from_value(payload.get("current_step_index"), 0)))
        self.status_var.set(str(payload.get("status_message", f"Loaded project {project_dir.name}.")))
        self._ensure_current_project_dir()
        self._refresh_wire_profiles()
        self._show_step()
        self._redraw_canvas()

    def _refresh_wire_profiles(self) -> None:
        if self.syncing_wire_profiles:
            return
        self.syncing_wire_profiles = True
        try:
            current_payload = self._project_payload_without_wire_profiles()
            current_payload["wire_profiles"] = [dict(item) for item in self.wire_profiles]
            connection_specs = _wire_connections_from_payload(current_payload)
            self.wire_profiles = [dict(spec["wire_profile"]) for spec in connection_specs]
            labels = []
            for profile in self.wire_profiles:
                label = f"{profile.get('connection_id', '')}  lead P{int(profile.get('leg_path_index', 0))} -> {profile.get('die_side_name', '')} {int(profile.get('die_section_index', 0))}"
                labels.append(label)
            self.wire_selector.configure(values=labels)
            if not self.wire_profiles:
                self.selected_wire_profile_id_var.set("")
                return
            selected_id = self.selected_wire_profile_id_var.get().split("  ", 1)[0].strip()
            selected_profile = next((item for item in self.wire_profiles if str(item.get("connection_id", "")) == selected_id), None)
            if selected_profile is None:
                selected_profile = self.wire_profiles[0]
            self.selected_wire_profile_id_var.set(
                f"{selected_profile.get('connection_id', '')}  lead P{int(selected_profile.get('leg_path_index', 0))} -> {selected_profile.get('die_side_name', '')} {int(selected_profile.get('die_section_index', 0))}"
            )
            self._load_selected_wire_profile_into_editor(selected_profile)
        finally:
            self.syncing_wire_profiles = False

    def _on_wire_profile_selected(self, _event=None) -> None:
        selected_id = self.selected_wire_profile_id_var.get().split("  ", 1)[0].strip()
        selected_profile = next((item for item in self.wire_profiles if str(item.get("connection_id", "")) == selected_id), None)
        if selected_profile is not None:
            self._load_selected_wire_profile_into_editor(selected_profile)

    def _load_selected_wire_profile_into_editor(self, profile: dict[str, object]) -> None:
        self.loading_wire_profile = True
        try:
            self.wire_arc_height_var.set(self._safe_float_from_value(profile.get("arc_height_mm"), 0.5))
            self.wire_arc_xy_noise_var.set(self._safe_float_from_value(profile.get("arc_xy_noise_mm"), 0.0))
            self.wire_rise_z_var.set(self._safe_float_from_value(profile.get("wire_rise_z_mm"), 0.12))
            self.wire_diameter_var.set(self._safe_float_from_value(profile.get("wire_diameter_mm"), 0.03))
            self.wire_point_spacing_var.set(self._safe_float_from_value(profile.get("wire_arc_point_spacing_mm"), 0.08))
            self.wire_tube_side_count_var.set(self._safe_int_from_value(profile.get("wire_tube_side_count"), 10))
            self.ball_bond_diameter_var.set(self._safe_float_from_value(profile.get("ball_bond_diameter_mm"), 0.12))
            self.ball_bond_length_var.set(self._safe_float_from_value(profile.get("ball_bond_length_mm"), 0.08))
            self.ball_bond_revolution_steps_var.set(self._safe_int_from_value(profile.get("ball_bond_revolution_steps"), 24))
            self.wedge_bond_length_var.set(self._safe_float_from_value(profile.get("wedge_bond_length_mm"), 0.18))
            self.wedge_bond_width_var.set(self._safe_float_from_value(profile.get("wedge_bond_width_mm"), 0.08))
            self.wedge_bond_thickness_var.set(self._safe_float_from_value(profile.get("wedge_bond_thickness_mm"), 0.02))
            self.wedge_approach_run_var.set(self._safe_float_from_value(profile.get("wedge_approach_run_mm"), 0.18))
            self.wedge_tail_var.set(self._safe_float_from_value(profile.get("wedge_tail_mm"), 0.0))
        finally:
            self.loading_wire_profile = False

    def _apply_wire_profile_editor(self) -> None:
        if self.loading_wire_profile:
            return
        selected_id = self.selected_wire_profile_id_var.get().split("  ", 1)[0].strip()
        if not selected_id:
            return
        for profile in self.wire_profiles:
            if str(profile.get("connection_id", "")) != selected_id:
                continue
            profile["arc_height_mm"] = max(0.0, self._safe_float(self.wire_arc_height_var, 0.5))
            profile["arc_xy_noise_mm"] = self._safe_float(self.wire_arc_xy_noise_var, 0.0)
            profile["wire_rise_z_mm"] = max(0.01, self._safe_float(self.wire_rise_z_var, 0.12))
            profile["wire_diameter_mm"] = max(0.005, self._safe_float(self.wire_diameter_var, 0.03))
            profile["wire_arc_point_spacing_mm"] = max(0.01, self._safe_float(self.wire_point_spacing_var, 0.08))
            profile["wire_tube_side_count"] = max(3, self._safe_int(self.wire_tube_side_count_var, 10))
            profile["ball_bond_diameter_mm"] = max(0.02, self._safe_float(self.ball_bond_diameter_var, 0.12))
            profile["ball_bond_length_mm"] = max(0.01, self._safe_float(self.ball_bond_length_var, 0.08))
            profile["ball_bond_revolution_steps"] = max(6, self._safe_int(self.ball_bond_revolution_steps_var, 24))
            profile["wedge_bond_length_mm"] = max(0.04, self._safe_float(self.wedge_bond_length_var, 0.18))
            profile["wedge_bond_width_mm"] = max(0.02, self._safe_float(self.wedge_bond_width_var, 0.08))
            profile["wedge_bond_thickness_mm"] = max(0.005, self._safe_float(self.wedge_bond_thickness_var, 0.02))
            profile["wedge_approach_run_mm"] = max(0.02, self._safe_float(self.wedge_approach_run_var, 0.18))
            profile["wedge_tail_mm"] = max(0.0, self._safe_float(self.wedge_tail_var, 0.0))
            break

    def _sample_gaussian(self, mean_var: tk.Variable, std_var: tk.Variable, minimum: float, allow_negative: bool = False) -> float:
        mean_value = self._safe_float(mean_var, minimum)
        std_value = max(0.0, self._safe_float(std_var, 0.0))
        sampled_value = random.gauss(mean_value, std_value)
        if allow_negative:
            return sampled_value
        return max(minimum, sampled_value)

    def _distribution_value(self, mean_var: tk.Variable, std_var: tk.Variable, minimum: float, mode: str, allow_negative: bool = False) -> float:
        mean_value = self._safe_float(mean_var, minimum)
        std_value = max(0.0, self._safe_float(std_var, 0.0))
        z_score = 0.0
        if mode == "p05":
            z_score = -1.6448536269514722
        elif mode == "p95":
            z_score = 1.6448536269514722
        value = mean_value + (std_value * z_score)
        if allow_negative:
            return value
        return max(minimum, value)

    def _randomize_wire_profiles(self) -> None:
        if not self.wire_profiles:
            self.status_var.set("No wire profiles are available to randomize yet.")
            return
        for profile in self.wire_profiles:
            ball_radius_mm = self._sample_gaussian(self.rand_ball_radius_mean_var, self.rand_ball_radius_std_var, 0.01)
            profile["ball_bond_diameter_mm"] = max(0.02, ball_radius_mm * 2.0)
            profile["ball_bond_length_mm"] = self._sample_gaussian(self.rand_ball_height_mean_var, self.rand_ball_height_std_var, 0.01)
            profile["arc_height_mm"] = self._sample_gaussian(self.rand_arc_height_mean_var, self.rand_arc_height_std_var, 0.0)
            profile["arc_xy_noise_mm"] = self._sample_gaussian(self.rand_arc_xy_noise_mean_var, self.rand_arc_xy_noise_std_var, 0.0, allow_negative=True)
            profile["wire_rise_z_mm"] = self._sample_gaussian(self.rand_wire_rise_mean_var, self.rand_wire_rise_std_var, 0.01)
            profile["wire_diameter_mm"] = self._sample_gaussian(self.rand_wire_diameter_mean_var, self.rand_wire_diameter_std_var, 0.005)
            profile["wedge_bond_length_mm"] = self._sample_gaussian(self.rand_wedge_length_mean_var, self.rand_wedge_length_std_var, 0.04)
            profile["wedge_bond_width_mm"] = self._sample_gaussian(self.rand_wedge_width_mean_var, self.rand_wedge_width_std_var, 0.02)
            profile["wedge_bond_thickness_mm"] = self._sample_gaussian(self.rand_wedge_thickness_mean_var, self.rand_wedge_thickness_std_var, 0.005)
            profile["wedge_approach_run_mm"] = self._sample_gaussian(self.rand_wedge_approach_mean_var, self.rand_wedge_approach_std_var, 0.02)
            profile["wedge_tail_mm"] = self._sample_gaussian(self.rand_wedge_tail_mean_var, self.rand_wedge_tail_std_var, 0.0)
        self._refresh_wire_profiles()
        self._redraw_canvas()
        self._autosave_project_files()
        self._push_viewer_payload()
        self.status_var.set(f"Randomized {len(self.wire_profiles)} wire profile(s) from mean/std settings.")

    def _apply_wire_profile_distribution_view(self, mode: str) -> None:
        if not self.wire_profiles:
            self.status_var.set("No wire profiles are available to update yet.")
            return
        for profile in self.wire_profiles:
            ball_radius_mm = self._distribution_value(self.rand_ball_radius_mean_var, self.rand_ball_radius_std_var, 0.01, mode)
            profile["ball_bond_diameter_mm"] = max(0.02, ball_radius_mm * 2.0)
            profile["ball_bond_length_mm"] = self._distribution_value(self.rand_ball_height_mean_var, self.rand_ball_height_std_var, 0.01, mode)
            profile["arc_height_mm"] = self._distribution_value(self.rand_arc_height_mean_var, self.rand_arc_height_std_var, 0.0, mode)
            profile["arc_xy_noise_mm"] = self._distribution_value(self.rand_arc_xy_noise_mean_var, self.rand_arc_xy_noise_std_var, 0.0, mode, allow_negative=True)
            profile["wire_rise_z_mm"] = self._distribution_value(self.rand_wire_rise_mean_var, self.rand_wire_rise_std_var, 0.01, mode)
            profile["wire_diameter_mm"] = self._distribution_value(self.rand_wire_diameter_mean_var, self.rand_wire_diameter_std_var, 0.005, mode)
            profile["wedge_bond_length_mm"] = self._distribution_value(self.rand_wedge_length_mean_var, self.rand_wedge_length_std_var, 0.04, mode)
            profile["wedge_bond_width_mm"] = self._distribution_value(self.rand_wedge_width_mean_var, self.rand_wedge_width_std_var, 0.02, mode)
            profile["wedge_bond_thickness_mm"] = self._distribution_value(self.rand_wedge_thickness_mean_var, self.rand_wedge_thickness_std_var, 0.005, mode)
            profile["wedge_approach_run_mm"] = self._distribution_value(self.rand_wedge_approach_mean_var, self.rand_wedge_approach_std_var, 0.02, mode)
            profile["wedge_tail_mm"] = self._distribution_value(self.rand_wedge_tail_mean_var, self.rand_wedge_tail_std_var, 0.0, mode)
        self._refresh_wire_profiles()
        self._redraw_canvas()
        self._autosave_project_files()
        self._push_viewer_payload()
        mode_label = "average" if mode == "mean" else "lowest 5%" if mode == "p05" else "highest 95%"
        self.status_var.set(f"Applied {mode_label} distribution values to {len(self.wire_profiles)} wire profile(s).")

    def _project_payload_without_wire_profiles(self) -> dict:
        payload = self._project_payload()
        payload["wire_profiles"] = []
        return payload

    def _show_step(self) -> None:
        self.step_label.configure(text=self.step_titles[self.step_index])
        for index, section in enumerate(self.step_sections):
            if index == self.step_index:
                section.pack(fill="x", pady=(0, 12))
            else:
                section.pack_forget()

    def _previous_step(self) -> None:
        self.step_index = max(0, self.step_index - 1)
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.")
        self._autosave_project_files()

    def _next_step(self) -> None:
        if self.step_index >= len(self.step_titles) - 1:
            self.status_var.set(f"{self.step_titles[self.step_index]} active.")
            self._open_next_step_viewer()
            return
        self.step_index = min(len(self.step_titles) - 1, self.step_index + 1)
        if self.step_index >= 2:
            self._refresh_wire_profiles()
        self._show_step()
        self.status_var.set(f"{self.step_titles[self.step_index]} active.")
        self._autosave_project_files()
        if self.step_index >= 1:
            self._open_next_step_viewer()

    def _world_to_canvas(self, point_mm: tuple[float, float]) -> tuple[float, float]:
        x_mm, y_mm = point_mm
        return (
            (DEFAULT_CANVAS_WIDTH / 2.0) + self.view_offset_px[0] + (x_mm * self.scale_px_per_mm),
            (DEFAULT_CANVAS_HEIGHT / 2.0) + self.view_offset_px[1] - (y_mm * self.scale_px_per_mm),
        )

    def _canvas_to_world(self, point_px: tuple[float, float]) -> tuple[float, float]:
        x_px, y_px = point_px
        return (
            (x_px - (DEFAULT_CANVAS_WIDTH / 2.0) - self.view_offset_px[0]) / self.scale_px_per_mm,
            ((DEFAULT_CANVAS_HEIGHT / 2.0) + self.view_offset_px[1] - y_px) / self.scale_px_per_mm,
        )

    def _draw_grid(self) -> None:
        step_px = max(8, int(self.scale_px_per_mm))
        for x_coord in range(0, DEFAULT_CANVAS_WIDTH, step_px):
            self.canvas.create_line(x_coord, 0, x_coord, DEFAULT_CANVAS_HEIGHT, fill=CANVAS_GRID)
        for y_coord in range(0, DEFAULT_CANVAS_HEIGHT, step_px):
            self.canvas.create_line(0, y_coord, DEFAULT_CANVAS_WIDTH, y_coord, fill=CANVAS_GRID)
        self.canvas.create_line(DEFAULT_CANVAS_WIDTH / 2.0, 0, DEFAULT_CANVAS_WIDTH / 2.0, DEFAULT_CANVAS_HEIGHT, fill=CANVAS_AXIS, width=2)
        self.canvas.create_line(0, DEFAULT_CANVAS_HEIGHT / 2.0, DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT / 2.0, fill=CANVAS_AXIS, width=2)

    def _outline_dims(self) -> tuple[float, float]:
        return (
            max(1.0, self._safe_float(self.outline_x_var, 10.0)),
            max(1.0, self._safe_float(self.outline_y_var, 10.0)),
        )

    def _die_square_size_mm(self) -> float:
        outline_x_mm, outline_y_mm = self._outline_dims()
        return max(0.5, min(self._safe_float(self.die_size_var, 3.0), min(outline_x_mm, outline_y_mm) * 0.9))

    def _slot_positions(self, span_mm: float, count: int, spacing_mm: float) -> list[float]:
        count = max(0, int(count))
        if count <= 0:
            return []
        if count == 1:
            return [0.0]
        spread_mm = max(0.0, float(spacing_mm)) * (count - 1)
        spread_mm = min(spread_mm, span_mm)
        actual_spacing_mm = spread_mm / (count - 1)
        return [(index * actual_spacing_mm) - (spread_mm / 2.0) for index in range(count)]

    def _slot_edge_margin_mm(self, span_mm: float, count: int, spacing_mm: float) -> float:
        count = max(0, int(count))
        if count <= 1:
            return span_mm / 2.0
        spread_mm = max(0.0, float(spacing_mm)) * (count - 1)
        spread_mm = min(spread_mm, span_mm)
        return max(0.0, (span_mm - spread_mm) / 2.0)

    def _guide_slots(self) -> list[GuideSlot]:
        outline_x_mm, outline_y_mm = self._outline_dims()
        half_width_mm = outline_x_mm / 2.0
        half_height_mm = outline_y_mm / 2.0
        guide_half_length_mm = max(0.1, self._safe_float(self.path_width_var, 0.8) / 2.0)
        slots: list[GuideSlot] = []

        for index, x_pos in enumerate(self._slot_positions(outline_x_mm, self._safe_int(self.top_count_var, 3), self._safe_float(self.top_spacing_var, 2.0)), start=1):
            slots.append(GuideSlot("TOP", index, (x_pos, half_height_mm), (x_pos - guide_half_length_mm, half_height_mm), (x_pos + guide_half_length_mm, half_height_mm)))
        for index, x_pos in enumerate(self._slot_positions(outline_x_mm, self._safe_int(self.bottom_count_var, 3), self._safe_float(self.bottom_spacing_var, 2.0)), start=1):
            slots.append(GuideSlot("BOTTOM", index, (x_pos, -half_height_mm), (x_pos - guide_half_length_mm, -half_height_mm), (x_pos + guide_half_length_mm, -half_height_mm)))
        for index, y_pos in enumerate(self._slot_positions(outline_y_mm, self._safe_int(self.left_count_var, 0), self._safe_float(self.left_spacing_var, 2.0)), start=1):
            slots.append(GuideSlot("LEFT", index, (-half_width_mm, y_pos), (-half_width_mm, y_pos - guide_half_length_mm), (-half_width_mm, y_pos + guide_half_length_mm)))
        for index, y_pos in enumerate(self._slot_positions(outline_y_mm, self._safe_int(self.right_count_var, 0), self._safe_float(self.right_spacing_var, 2.0)), start=1):
            slots.append(GuideSlot("RIGHT", index, (half_width_mm, y_pos), (half_width_mm, y_pos - guide_half_length_mm), (half_width_mm, y_pos + guide_half_length_mm)))
        return slots

    def _die_square_corners_mm(self) -> list[tuple[float, float]]:
        half_size_mm = self._die_square_size_mm() / 2.0
        return [
            (-half_size_mm, -half_size_mm),
            (-half_size_mm, half_size_mm),
            (half_size_mm, half_size_mm),
            (half_size_mm, -half_size_mm),
        ]

    def _end_region_center_mm(self) -> tuple[float, float]:
        return (
            self._safe_float(self.end_region_offset_x_var, 0.0),
            self._safe_float(self.end_region_offset_y_var, -2.0),
        )

    def _end_region_size_mm(self) -> float:
        return max(0.2, self._safe_float(self.end_region_size_var, 0.8))

    def _end_region_corners_mm(self) -> list[tuple[float, float]]:
        center_x_mm, center_y_mm = self._end_region_center_mm()
        half_size_mm = self._end_region_size_mm() / 2.0
        return [
            (center_x_mm - half_size_mm, center_y_mm - half_size_mm),
            (center_x_mm - half_size_mm, center_y_mm + half_size_mm),
            (center_x_mm + half_size_mm, center_y_mm + half_size_mm),
            (center_x_mm + half_size_mm, center_y_mm - half_size_mm),
        ]

    def _encapsulation_width_mm(self) -> float:
        outline_x_mm, _outline_y_mm = self._outline_dims()
        return max(0.1, self._safe_float(self.encapsulation_width_var, outline_x_mm * 0.7))

    def _encapsulation_length_mm(self) -> float:
        _outline_x_mm, outline_y_mm = self._outline_dims()
        return max(0.1, self._safe_float(self.encapsulation_length_var, outline_y_mm * 0.7))

    def _encapsulation_negative_extrusion_mm(self) -> float:
        return max(0.0, self._safe_float(self.encapsulation_negative_extrusion_var, 0.0))

    def _encapsulation_positive_extrusion_mm(self) -> float:
        return max(0.0, self._safe_float(self.encapsulation_positive_extrusion_var, 0.6))

    def _encapsulation_corners_mm(self) -> list[tuple[float, float]]:
        half_width_mm = self._encapsulation_width_mm() / 2.0
        half_length_mm = self._encapsulation_length_mm() / 2.0
        return [
            (-half_width_mm, -half_length_mm),
            (-half_width_mm, half_length_mm),
            (half_width_mm, half_length_mm),
            (half_width_mm, -half_length_mm),
        ]

    def _bond_start_region_counts(self) -> dict[str, int]:
        return {
            "Top": max(0, self._safe_int(self.bond_start_top_count_var, 2)),
            "Bottom": max(0, self._safe_int(self.bond_start_bottom_count_var, 2)),
            "Left": max(0, self._safe_int(self.bond_start_left_count_var, 2)),
            "Right": max(0, self._safe_int(self.bond_start_right_count_var, 2)),
        }

    def _bond_start_region_size_mm(self) -> float:
        return max(0.05, self._safe_float(self.bond_start_region_size_var, 0.2))

    def _bond_start_region_gap_mm(self) -> float:
        return max(0.0, self._safe_float(self.bond_start_region_gap_var, 0.15))

    def _bond_start_region_offset_mm(self) -> float:
        return max(0.0, self._safe_float(self.bond_start_region_offset_var, 0.08))

    def _bond_end_region_size_mm(self) -> float:
        return max(0.05, self._safe_float(self.bond_end_region_size_var, 0.2))

    def _bond_end_region_offset_mm(self) -> float:
        return max(0.0, self._safe_float(self.bond_end_region_offset_var, 0.0))

    def _silicon_die_thickness_mm(self) -> float:
        return max(0.02, self._safe_float(self.silicon_die_thickness_var, 0.15))

    def _silicon_die_width_mm(self) -> float:
        return max(0.05, self._safe_float(self.silicon_die_width_var, self._die_square_size_mm() * 0.8))

    def _silicon_die_height_mm(self) -> float:
        return max(0.05, self._safe_float(self.silicon_die_height_var, self._die_square_size_mm() * 0.8))

    def _bond_start_region_centers_mm(self) -> list[dict[str, object]]:
        die_width_mm = self._silicon_die_width_mm()
        die_height_mm = self._silicon_die_height_mm()
        half_die_width_mm = die_width_mm / 2.0
        half_die_height_mm = die_height_mm / 2.0
        region_size_mm = self._bond_start_region_size_mm()
        gap_mm = self._bond_start_region_gap_mm()
        offset_mm = self._bond_start_region_offset_mm()
        counts = self._bond_start_region_counts()
        regions: list[dict[str, object]] = []

        def positions_for_side(count: int, usable_side_mm: float) -> list[float]:
            if count <= 0:
                return []
            if count == 1:
                return [0.0]
            total_span_mm = (count * region_size_mm) + ((count - 1) * gap_mm)
            usable_span_mm = min(total_span_mm, usable_side_mm)
            actual_gap_mm = gap_mm
            if count > 1 and total_span_mm > usable_side_mm:
                actual_gap_mm = max(0.0, (usable_side_mm - (count * region_size_mm)) / (count - 1))
                usable_span_mm = (count * region_size_mm) + ((count - 1) * actual_gap_mm)
            start_center_mm = -(usable_span_mm / 2.0) + (region_size_mm / 2.0)
            return [start_center_mm + (index * (region_size_mm + actual_gap_mm)) for index in range(count)]

        for index, x_center_mm in enumerate(positions_for_side(counts["Top"], die_width_mm), start=1):
            regions.append({"side_name": "Top", "section_index": index, "center_mm": (x_center_mm, half_die_height_mm - offset_mm - (region_size_mm / 2.0))})
        for index, x_center_mm in enumerate(positions_for_side(counts["Bottom"], die_width_mm), start=1):
            regions.append({"side_name": "Bottom", "section_index": index, "center_mm": (x_center_mm, -half_die_height_mm + offset_mm + (region_size_mm / 2.0))})
        for index, y_center_mm in enumerate(positions_for_side(counts["Left"], die_height_mm), start=1):
            regions.append({"side_name": "Left", "section_index": index, "center_mm": (-half_die_width_mm + offset_mm + (region_size_mm / 2.0), y_center_mm)})
        for index, y_center_mm in enumerate(positions_for_side(counts["Right"], die_height_mm), start=1):
            regions.append({"side_name": "Right", "section_index": index, "center_mm": (half_die_width_mm - offset_mm - (region_size_mm / 2.0), y_center_mm)})
        return regions

    def _bond_end_region_centers_mm(self) -> list[dict[str, object]]:
        regions: list[dict[str, object]] = []
        offset_mm = self._bond_end_region_offset_mm()
        for path_index, path_mm in enumerate(self.saved_paths_mm, start=1):
            if len(path_mm) < 2:
                continue
            end_x_mm, end_y_mm = path_mm[-1]
            previous_x_mm, previous_y_mm = path_mm[-2]
            direction_xy = _normalize_2d((previous_x_mm - end_x_mm, previous_y_mm - end_y_mm))
            if direction_xy is None:
                center_mm = (end_x_mm, end_y_mm)
            else:
                center_mm = (
                    end_x_mm + (direction_xy[0] * offset_mm),
                    end_y_mm + (direction_xy[1] * offset_mm),
                )
            dx = end_x_mm - path_mm[0][0]
            dy = end_y_mm - path_mm[0][1]
            if abs(dx) >= abs(dy):
                side_name = "Right" if dx >= 0.0 else "Left"
            else:
                side_name = "Top" if dy >= 0.0 else "Bottom"
            regions.append(
                {
                    "path_index": path_index,
                    "side_name": side_name,
                    "center_mm": center_mm,
                    "approach_mm": (previous_x_mm, previous_y_mm),
                    "path_end_mm": (end_x_mm, end_y_mm),
                }
            )
        return regions

    def _draw_outline(self) -> None:
        outline_x_mm, outline_y_mm = self._outline_dims()
        left_top = self._world_to_canvas((-outline_x_mm / 2.0, outline_y_mm / 2.0))
        right_bottom = self._world_to_canvas((outline_x_mm / 2.0, -outline_y_mm / 2.0))
        self.canvas.create_rectangle(
            left_top[0],
            left_top[1],
            right_bottom[0],
            right_bottom[1],
            outline="#6f5132",
            width=2,
            dash=(6, 4),
        )
        self.canvas.create_text(
            (left_top[0] + right_bottom[0]) / 2.0,
            left_top[1] - 18,
            text=f"Leadframe Outline {outline_x_mm:.2f} x {outline_y_mm:.2f} mm   Z {self._safe_float(self.thickness_z_var, 0.2):.2f} mm",
            fill="#4a3727",
            font=("Segoe UI", 10, "bold"),
        )

    def _draw_die_square(self) -> None:
        flat_points: list[float] = []
        for point_mm in self._die_square_corners_mm():
            x_px, y_px = self._world_to_canvas(point_mm)
            flat_points.extend([x_px, y_px])
        self.canvas.create_polygon(*flat_points, outline="#b91c1c", width=2, dash=(4, 3), fill="")
        center_px = self._world_to_canvas((0.0, 0.0))
        self.canvas.create_text(
            center_px[0],
            center_px[1] - (self._die_square_size_mm() * self.scale_px_per_mm / 2.0) - 14,
            text=f"Centered Die Compartment {self._die_square_size_mm():.2f} mm square",
            fill="#7f1d1d",
            font=("Segoe UI", 9, "bold"),
        )
        end_region_flat_points: list[float] = []
        for point_mm in self._end_region_corners_mm():
            x_px, y_px = self._world_to_canvas(point_mm)
            end_region_flat_points.extend([x_px, y_px])
        self.canvas.create_polygon(*end_region_flat_points, outline="#0f766e", width=2, dash=(3, 3), fill="")
        end_center_px = self._world_to_canvas(self._end_region_center_mm())
        self.canvas.create_oval(end_center_px[0] - 4, end_center_px[1] - 4, end_center_px[0] + 4, end_center_px[1] + 4, fill="#14b8a6", outline="")
        self.canvas.create_text(
            end_center_px[0],
            end_center_px[1] - (self._end_region_size_mm() * self.scale_px_per_mm / 2.0) - 12,
            text=f"Ending Region {self._end_region_size_mm():.2f} mm",
            fill="#115e59",
            font=("Segoe UI", 8, "bold"),
        )

    def _draw_encapsulation_outline(self) -> None:
        flat_points: list[float] = []
        for point_mm in self._encapsulation_corners_mm():
            x_px, y_px = self._world_to_canvas(point_mm)
            flat_points.extend([x_px, y_px])
        self.canvas.create_polygon(*flat_points, outline="#7c2d12", width=2, dash=(8, 4), fill="")
        center_px = self._world_to_canvas((0.0, 0.0))
        self.canvas.create_text(
            center_px[0],
            center_px[1] + (self._encapsulation_length_mm() * self.scale_px_per_mm / 2.0) + 16,
            text=(
                f"Encapsulation {self._encapsulation_width_mm():.2f} x {self._encapsulation_length_mm():.2f} mm  "
                f"-Z {self._encapsulation_negative_extrusion_mm():.2f}  +Z {self._encapsulation_positive_extrusion_mm():.2f}"
            ),
            fill="#7c2d12",
            font=("Segoe UI", 8, "bold"),
        )

    def _draw_guide_slots(self) -> None:
        for slot in self._guide_slots():
            start_px = self._world_to_canvas(slot.start_mm)
            end_px = self._world_to_canvas(slot.end_mm)
            center_px = self._world_to_canvas(slot.center_mm)
            self.canvas.create_line(start_px[0], start_px[1], end_px[0], end_px[1], fill="#f59e0b", width=6, capstyle="round")
            self.canvas.create_oval(center_px[0] - 5, center_px[1] - 5, center_px[0] + 5, center_px[1] + 5, fill="#fde68a", outline="#f97316", width=2)
            self.canvas.create_text(center_px[0] + 16, center_px[1] - 12, text=f"{slot.side_name[0]}{slot.index}", fill="#8a4b08", font=("Segoe UI", 8, "bold"))
            self.canvas.create_text(center_px[0] + 18, center_px[1] + 12, text="Click", fill="#8a4b08", font=("Segoe UI", 8, "bold"))
        outline_x_mm, outline_y_mm = self._outline_dims()
        margin_rows = [
            ("TOP", outline_x_mm, self._safe_int(self.top_count_var, 3), self._safe_float(self.top_spacing_var, 2.0), (0.0, (outline_y_mm / 2.0) + 0.55)),
            ("BOTTOM", outline_x_mm, self._safe_int(self.bottom_count_var, 3), self._safe_float(self.bottom_spacing_var, 2.0), (0.0, (-outline_y_mm / 2.0) - 0.85)),
            ("LEFT", outline_y_mm, self._safe_int(self.left_count_var, 0), self._safe_float(self.left_spacing_var, 2.0), ((-outline_x_mm / 2.0) - 1.1, 0.0)),
            ("RIGHT", outline_y_mm, self._safe_int(self.right_count_var, 0), self._safe_float(self.right_spacing_var, 2.0), ((outline_x_mm / 2.0) + 1.1, 0.0)),
        ]
        for side_name, span_mm, count, spacing_mm, label_point_mm in margin_rows:
            if count <= 0:
                continue
            edge_margin_mm = self._slot_edge_margin_mm(span_mm, count, spacing_mm)
            label_px = self._world_to_canvas(label_point_mm)
            self.canvas.create_text(
                label_px[0],
                label_px[1],
                text=f"{side_name} gap {spacing_mm:.2f} mm  edge {edge_margin_mm:.2f} mm",
                fill="#6b4f1d",
                font=("Segoe UI", 8, "bold"),
            )

    def _draw_paths(self) -> None:
        for path_index, path_mm in enumerate(self.saved_paths_mm):
            self._draw_path(path_mm, "#14532d", "#22c55e", f"P{path_index + 1}")
        if self.current_path_mm:
            preview_path = list(self.current_path_mm)
            if self.preview_point_mm is not None:
                preview_path.append(self.preview_point_mm)
            self._draw_path(preview_path, "#1d4ed8", "#60a5fa", "DRAFT", dashed=len(preview_path) > len(self.current_path_mm))

    def _offset_path_mm(self, path_mm: list[tuple[float, float]], offset_mm: float) -> list[tuple[float, float]]:
        if len(path_mm) < 2:
            return list(path_mm)
        segment_directions: list[tuple[float, float]] = []
        segment_normals: list[tuple[float, float]] = []
        for start_mm, end_mm in zip(path_mm[:-1], path_mm[1:]):
            direction_xy = _normalize_2d((end_mm[0] - start_mm[0], end_mm[1] - start_mm[1]))
            if direction_xy is None:
                continue
            segment_directions.append(direction_xy)
            segment_normals.append(_left_normal_2d(direction_xy))
        if not segment_directions:
            return list(path_mm)

        offset_points: list[tuple[float, float]] = [(
            path_mm[0][0] + (segment_normals[0][0] * offset_mm),
            path_mm[0][1] + (segment_normals[0][1] * offset_mm),
        )]
        for point_index in range(1, len(path_mm) - 1):
            point_mm = path_mm[point_index]
            prev_direction = segment_directions[point_index - 1]
            next_direction = segment_directions[point_index]
            prev_normal = segment_normals[point_index - 1]
            next_normal = segment_normals[point_index]
            intersection = _line_intersection_2d(
                (point_mm[0] + (prev_normal[0] * offset_mm), point_mm[1] + (prev_normal[1] * offset_mm)),
                prev_direction,
                (point_mm[0] + (next_normal[0] * offset_mm), point_mm[1] + (next_normal[1] * offset_mm)),
                next_direction,
            )
            if intersection is None:
                average_normal = _normalize_2d((prev_normal[0] + next_normal[0], prev_normal[1] + next_normal[1]))
                if average_normal is None:
                    average_normal = prev_normal
                intersection = (
                    point_mm[0] + (average_normal[0] * offset_mm),
                    point_mm[1] + (average_normal[1] * offset_mm),
                )
            offset_points.append(intersection)
        offset_points.append((
            path_mm[-1][0] + (segment_normals[-1][0] * offset_mm),
            path_mm[-1][1] + (segment_normals[-1][1] * offset_mm),
        ))
        return offset_points

    def _draw_polyline_mm(self, path_mm: list[tuple[float, float]], color: str, width_px: float, dashed: bool = False) -> None:
        if len(path_mm) < 2:
            return
        flat_points: list[float] = []
        for point_mm in path_mm:
            x_px, y_px = self._world_to_canvas(point_mm)
            flat_points.extend([x_px, y_px])
        self.canvas.create_line(
            *flat_points,
            fill=color,
            width=width_px,
            dash=(5, 3) if dashed else (),
            capstyle="round",
            joinstyle="round",
        )

    def _draw_path(self, path_mm: list[tuple[float, float]], line_color: str, vertex_color: str, label: str, dashed: bool = False) -> None:
        half_width_mm = max(0.05, self._safe_float(self.path_width_var, 0.8) / 2.0)
        rail_width_px = 2.0
        if len(path_mm) >= 2:
            left_edge_mm = self._offset_path_mm(path_mm, half_width_mm)
            right_edge_mm = self._offset_path_mm(path_mm, -half_width_mm)
            self._draw_polyline_mm(left_edge_mm, line_color, rail_width_px, dashed=dashed)
            self._draw_polyline_mm(right_edge_mm, line_color, rail_width_px, dashed=dashed)
            self._draw_polyline_mm(path_mm, vertex_color, 1.0, dashed=dashed)
        for point_index, point_mm in enumerate(path_mm):
            x_px, y_px = self._world_to_canvas(point_mm)
            self.canvas.create_oval(x_px - 2.5, y_px - 2.5, x_px + 2.5, y_px + 2.5, fill=vertex_color, outline="")
            if point_index == 0:
                self.canvas.create_text(x_px + 14, y_px - 10, text=f"{label}  {self._safe_float(self.path_width_var, 0.8):.2f} mm", fill=line_color, font=("Segoe UI", 8, "bold"))

    def _closest_point_on_segment_mm(
        self,
        point_mm: tuple[float, float],
        start_mm: tuple[float, float],
        end_mm: tuple[float, float],
    ) -> tuple[float, float]:
        dx = end_mm[0] - start_mm[0]
        dy = end_mm[1] - start_mm[1]
        length_sq = (dx * dx) + (dy * dy)
        if length_sq <= 1e-12:
            return start_mm
        t_value = (((point_mm[0] - start_mm[0]) * dx) + ((point_mm[1] - start_mm[1]) * dy)) / length_sq
        t_value = max(0.0, min(1.0, t_value))
        return (
            start_mm[0] + (dx * t_value),
            start_mm[1] + (dy * t_value),
        )

    def _find_end_region_snap_point(self, event_x: float, event_y: float) -> tuple[float, float] | None:
        cursor_mm = self._canvas_to_world((event_x, event_y))
        corners_mm = self._end_region_corners_mm()
        edges = [
            (corners_mm[0], corners_mm[1]),
            (corners_mm[1], corners_mm[2]),
            (corners_mm[2], corners_mm[3]),
            (corners_mm[3], corners_mm[0]),
        ]
        best_point_mm: tuple[float, float] | None = None
        best_distance_px = float("inf")
        threshold_px = max(10.0, (self._end_region_size_mm() * self.scale_px_per_mm * 0.35))
        for start_mm, end_mm in edges:
            candidate_mm = self._closest_point_on_segment_mm(cursor_mm, start_mm, end_mm)
            candidate_px = self._world_to_canvas(candidate_mm)
            distance_px = math.dist(candidate_px, (event_x, event_y))
            if distance_px <= threshold_px and distance_px < best_distance_px:
                best_distance_px = distance_px
                best_point_mm = candidate_mm
        if best_point_mm is not None:
            return best_point_mm
        return None

    def _mirror_paths(self, axis_name: str) -> None:
        if not self.saved_paths_mm:
            self.status_var.set("No saved paths are available to mirror.")
            return
        center_x_mm, center_y_mm = (0.0, 0.0)
        mirrored_paths_mm: list[list[tuple[float, float]]] = []
        for source_path_mm in self.saved_paths_mm:
            mirrored_path_mm: list[tuple[float, float]] = []
            for point_x_mm, point_y_mm in source_path_mm:
                if axis_name == "horizontal":
                    mirrored_path_mm.append((point_x_mm, (2.0 * center_y_mm) - point_y_mm))
                else:
                    mirrored_path_mm.append((((2.0 * center_x_mm) - point_x_mm), point_y_mm))
            mirrored_paths_mm.append(mirrored_path_mm)
        self.saved_paths_mm.extend(mirrored_paths_mm)
        self.status_var.set(f"Mirrored {len(mirrored_paths_mm)} path(s) across the {axis_name} axis.")
        self._redraw_canvas()
        self._autosave_project_files()

    def _mirror_paths_horizontal(self) -> None:
        self._mirror_paths("horizontal")

    def _mirror_paths_vertical(self) -> None:
        self._mirror_paths("vertical")

    def _find_slot_hit(self, event_x: float, event_y: float) -> GuideSlot | None:
        closest_slot: GuideSlot | None = None
        closest_distance_px = float("inf")
        for slot in self._guide_slots():
            center_px = self._world_to_canvas(slot.center_mm)
            distance_px = math.dist(center_px, (event_x, event_y))
            if distance_px <= 10.0 and distance_px < closest_distance_px:
                closest_slot = slot
                closest_distance_px = distance_px
        return closest_slot

    def _find_vertex_hit(self, event_x: float, event_y: float) -> tuple[int, int] | None:
        for path_index, path_mm in enumerate(self.saved_paths_mm):
            for point_index, point_mm in enumerate(path_mm):
                x_px, y_px = self._world_to_canvas(point_mm)
            if abs(x_px - event_x) <= 7 and abs(y_px - event_y) <= 7:
                return (path_index, point_index)
        return None

    def _build_next_step_meshes(self) -> list[tuple[str, trimesh.Trimesh, str]]:
        thickness_mm = max(0.05, self._safe_float(self.thickness_z_var, 0.2))
        width_mm = max(0.05, self._safe_float(self.path_width_var, 0.8))
        meshes: list[tuple[str, trimesh.Trimesh, str]] = []

        die_size_mm = self._die_square_size_mm()
        die_mesh = _ensure_outward_normals(trimesh.creation.box(extents=(die_size_mm, die_size_mm, thickness_mm)))
        die_mesh.apply_transform(_translation_matrix(0.0, 0.0, thickness_mm / 2.0))
        meshes.append(("Centered Die Compartment", die_mesh, "#b91c1c"))

        silicon_die_width_mm = self._silicon_die_width_mm()
        silicon_die_height_mm = self._silicon_die_height_mm()
        silicon_die_thickness_mm = self._silicon_die_thickness_mm()
        silicon_die_mesh = _ensure_outward_normals(trimesh.creation.box(extents=(silicon_die_width_mm, silicon_die_height_mm, silicon_die_thickness_mm)))
        silicon_die_mesh.apply_transform(_translation_matrix(0.0, 0.0, thickness_mm + DIE_CLEARANCE_MM + (silicon_die_thickness_mm / 2.0)))
        meshes.append(("Silicon Die", silicon_die_mesh, "#232323"))

        path_meshes: list[trimesh.Trimesh] = []
        for path_mm in self.saved_paths_mm:
            if len(path_mm) < 2:
                continue
            try:
                stroked_polygon_mm = _build_stroked_path_polygon(path_mm, width_mm)
                path_mesh = _extrude_closed_polygon(stroked_polygon_mm, thickness_mm)
            except Exception:
                continue
            path_meshes.append(path_mesh)
        if path_meshes:
            meshes.append(("Lead Frame Paths", _ensure_outward_normals(trimesh.util.concatenate(path_meshes)), "#14532d"))

        bond_end_region_size_mm = self._bond_end_region_size_mm()
        bond_end_meshes: list[trimesh.Trimesh] = []
        for region in self._bond_end_region_centers_mm():
            center_x_mm, center_y_mm = region["center_mm"]
            mesh = _ensure_outward_normals(trimesh.creation.box(extents=(bond_end_region_size_mm, bond_end_region_size_mm, 0.03)))
            mesh.apply_transform(_translation_matrix(center_x_mm, center_y_mm, thickness_mm + 0.015))
            bond_end_meshes.append(mesh)
        if bond_end_meshes:
            meshes.append(("Bond End Regions", _ensure_outward_normals(trimesh.util.concatenate(bond_end_meshes)), "#2563eb"))

        bond_start_region_size_mm = self._bond_start_region_size_mm()
        bond_start_meshes: list[trimesh.Trimesh] = []
        for region in self._bond_start_region_centers_mm():
            center_x_mm, center_y_mm = region["center_mm"]
            mesh = _ensure_outward_normals(trimesh.creation.box(extents=(bond_start_region_size_mm, bond_start_region_size_mm, 0.03)))
            mesh.apply_transform(_translation_matrix(center_x_mm, center_y_mm, thickness_mm + DIE_CLEARANCE_MM + silicon_die_thickness_mm + 0.015))
            bond_start_meshes.append(mesh)
        if bond_start_meshes:
            meshes.append(("Bond Start Regions", _ensure_outward_normals(trimesh.util.concatenate(bond_start_meshes)), "#7c3aed"))
        return meshes

    def _write_viewer_payload(self) -> None:
        payload_path = self.current_project_dir / VIEWER_PAYLOAD_NAME
        payload_path.write_text(json.dumps(self._project_payload(), indent=2), encoding="utf-8")

    def _push_viewer_payload(self) -> None:
        if self.step_index < 1:
            return
        if self.viewer_process is None or self.viewer_process.poll() is not None:
            return
        try:
            self._write_viewer_payload()
        except Exception as exc:
            self.status_var.set(f"Viewer payload update failed: {exc}")

    def _open_next_step_viewer(self) -> None:
        payload = self._project_payload()
        saved_paths = payload.get("saved_paths_mm", [])
        if not isinstance(saved_paths, list) or not any(isinstance(path, list) and len(path) >= 2 for path in saved_paths):
            messagebox.showerror("Next Step Blocked", "Create and save at least one leadframe path before opening the next-step 3D viewer.", parent=self.root)
            return
        try:
            payload_path = self.current_project_dir / VIEWER_PAYLOAD_NAME
            self._write_viewer_payload()
            if self.viewer_process is not None and self.viewer_process.poll() is None:
                self._write_viewer_payload()
                return
            self.viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--viewer",
                    str(payload_path),
                ],
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            messagebox.showerror("Viewer Failed", str(exc), parent=self.root)
            self.status_var.set(f"Next-step viewer failed: {exc}")

    def _snap_to_45_degrees(self, anchor_mm: tuple[float, float], candidate_mm: tuple[float, float]) -> tuple[float, float]:
        dx = candidate_mm[0] - anchor_mm[0]
        dy = candidate_mm[1] - anchor_mm[1]
        distance_mm = math.hypot(dx, dy)
        if distance_mm <= 1e-9:
            return candidate_mm
        angle_deg = math.degrees(math.atan2(dy, dx))
        snapped_angle_deg = round(angle_deg / 45.0) * 45.0
        snapped_angle_rad = math.radians(snapped_angle_deg)
        return (
            anchor_mm[0] + (math.cos(snapped_angle_rad) * distance_mm),
            anchor_mm[1] + (math.sin(snapped_angle_rad) * distance_mm),
        )

    def _on_left_click(self, event) -> None:
        vertex_hit = self._find_vertex_hit(event.x, event.y)
        if vertex_hit is not None:
            self.dragging_vertex = vertex_hit
            return

        slot_hit = self._find_slot_hit(event.x, event.y)
        if not self.current_path_mm:
            if slot_hit is None:
                self.status_var.set("Start by clicking a guide midpoint on one of the enabled sides.")
                return
            self.current_path_mm.append(slot_hit.center_mm)
            self.preview_point_mm = None
            self.status_var.set(f"Started path from {slot_hit.side_name} pin {slot_hit.index}.")
            self._redraw_canvas()
            self._autosave_project_files()
            return

        end_snap_mm = self._find_end_region_snap_point(event.x, event.y)
        if end_snap_mm is not None:
            snapped_mm = self._snap_to_45_degrees(self.current_path_mm[-1], end_snap_mm)
            self.current_path_mm.append(snapped_mm)
            self.preview_point_mm = None
            self.status_var.set("Snapped to ending region and finished the path.")
            self._finish_current_path()
            return

        candidate_mm = self._canvas_to_world((event.x, event.y))
        snapped_mm = self._snap_to_45_degrees(self.current_path_mm[-1], candidate_mm)
        self.current_path_mm.append(snapped_mm)
        self.preview_point_mm = None
        self.status_var.set(f"Added point {len(self.current_path_mm)} with 45-degree snap.")
        self._redraw_canvas()
        self._autosave_project_files()

    def _on_left_drag(self, event) -> None:
        if self.dragging_vertex is None:
            return
        path_index, point_index = self.dragging_vertex
        if 0 <= path_index < len(self.saved_paths_mm) and 0 <= point_index < len(self.saved_paths_mm[path_index]):
            self.saved_paths_mm[path_index][point_index] = self._canvas_to_world((event.x, event.y))
            self.status_var.set(f"Adjusted saved path {path_index + 1}, point {point_index + 1}.")
            self._redraw_canvas()

    def _on_left_release(self, _event) -> None:
        self.dragging_vertex = None
        self._autosave_project_files()

    def _on_motion(self, event) -> None:
        if self.is_panning or self.dragging_vertex is not None:
            return
        if not self.current_path_mm:
            self.preview_point_mm = None
            self._redraw_canvas()
            return
        end_snap_mm = self._find_end_region_snap_point(event.x, event.y)
        if end_snap_mm is not None:
            self.preview_point_mm = self._snap_to_45_degrees(self.current_path_mm[-1], end_snap_mm)
            self._redraw_canvas()
            return
        candidate_mm = self._canvas_to_world((event.x, event.y))
        self.preview_point_mm = self._snap_to_45_degrees(self.current_path_mm[-1], candidate_mm)
        self._redraw_canvas()

    def _on_pan_start(self, event) -> None:
        self.is_panning = True
        self.last_pan_px = (event.x, event.y)

    def _on_pan_move(self, event) -> None:
        if not self.is_panning or self.last_pan_px is None:
            return
        dx = event.x - self.last_pan_px[0]
        dy = event.y - self.last_pan_px[1]
        self.view_offset_px = (self.view_offset_px[0] + dx, self.view_offset_px[1] + dy)
        self.last_pan_px = (event.x, event.y)
        self._redraw_canvas()

    def _on_pan_end(self, _event) -> None:
        self.is_panning = False
        self.last_pan_px = None

    def _on_zoom(self, event) -> None:
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            delta = 1
        elif getattr(event, "num", None) == 5:
            delta = -1
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else (1.0 / 1.12)
        self.scale_px_per_mm = min(200.0, max(4.0, self.scale_px_per_mm * factor))
        self._redraw_canvas()

    def _finish_current_path(self) -> None:
        if len(self.current_path_mm) < 2:
            messagebox.showerror("Path Incomplete", "Add at least 2 points before finishing the path.", parent=self.root)
            return
        self.saved_paths_mm.append(list(self.current_path_mm))
        self.current_path_mm.clear()
        self.preview_point_mm = None
        self.status_var.set(f"Saved path {len(self.saved_paths_mm)}.")
        self._redraw_canvas()
        self._autosave_project_files()

    def _undo_point(self) -> None:
        if self.current_path_mm:
            self.current_path_mm.pop()
            self.preview_point_mm = None
            self.status_var.set(f"Draft points remaining: {len(self.current_path_mm)}.")
            self._redraw_canvas()
            self._autosave_project_files()
            return
        if self.saved_paths_mm:
            self.saved_paths_mm.pop()
            self.status_var.set("Removed last saved path.")
            self._redraw_canvas()
            self._autosave_project_files()

    def _clear_current_path(self) -> None:
        self.current_path_mm.clear()
        self.preview_point_mm = None
        self.status_var.set("Cleared current draft path.")
        self._redraw_canvas()
        self._autosave_project_files()

    def _clear_saved_paths(self) -> None:
        self.saved_paths_mm.clear()
        self.current_path_mm.clear()
        self.preview_point_mm = None
        self.status_var.set("Cleared all saved paths.")
        self._redraw_canvas()
        self._autosave_project_files()

    def _save_json(self) -> None:
        initial_dir = Path.cwd()
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Step 1 JSON",
            initialdir=str(initial_dir),
            initialfile="ic_chip_generator_New_step_1.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not target:
            return
        payload = self._project_payload()
        Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved step 1 JSON to {target}.")

    def _project_payload(self) -> dict:
        outline_x_mm, outline_y_mm = self._outline_dims()
        return {
            "project_name": self.project_name_var.get().strip() or self.current_project_dir.name,
            "project_dir": str(self.current_project_dir),
            "current_step_index": self.step_index,
            "current_step_title": self.step_titles[self.step_index],
            "status_message": self.status_var.get(),
            "outline_x_mm": outline_x_mm,
            "outline_y_mm": outline_y_mm,
            "thickness_z_mm": self._safe_float(self.thickness_z_var, 0.2),
            "leadframe_path_width_mm": self._safe_float(self.path_width_var, 0.8),
            "pins_per_side": {
                "top": self._safe_int(self.top_count_var, 3),
                "bottom": self._safe_int(self.bottom_count_var, 3),
                "left": self._safe_int(self.left_count_var, 0),
                "right": self._safe_int(self.right_count_var, 0),
            },
            "leg_spacing_mm": {
                "top": self._safe_float(self.top_spacing_var, 2.0),
                "bottom": self._safe_float(self.bottom_spacing_var, 2.0),
                "left": self._safe_float(self.left_spacing_var, 2.0),
                "right": self._safe_float(self.right_spacing_var, 2.0),
            },
            "edge_margin_mm": {
                "top": self._slot_edge_margin_mm(outline_x_mm, self._safe_int(self.top_count_var, 3), self._safe_float(self.top_spacing_var, 2.0)),
                "bottom": self._slot_edge_margin_mm(outline_x_mm, self._safe_int(self.bottom_count_var, 3), self._safe_float(self.bottom_spacing_var, 2.0)),
                "left": self._slot_edge_margin_mm(outline_y_mm, self._safe_int(self.left_count_var, 0), self._safe_float(self.left_spacing_var, 2.0)),
                "right": self._slot_edge_margin_mm(outline_y_mm, self._safe_int(self.right_count_var, 0), self._safe_float(self.right_spacing_var, 2.0)),
            },
            "die_compartment_square_size_mm": self._die_square_size_mm(),
            "die_compartment_center_mm": [0.0, 0.0],
            "end_region_offset_x_mm": self._safe_float(self.end_region_offset_x_var, 0.0),
            "end_region_offset_y_mm": self._safe_float(self.end_region_offset_y_var, -2.0),
            "end_region_size_mm": self._end_region_size_mm(),
            "end_region_center_mm": list(self._end_region_center_mm()),
            "outer_model_scale_percent": max(0.0, self._safe_float(self.outer_model_scale_percent_var, 0.1)),
            "encapsulation_width_mm": self._encapsulation_width_mm(),
            "encapsulation_length_mm": self._encapsulation_length_mm(),
            "encapsulation_negative_extrusion_mm": self._encapsulation_negative_extrusion_mm(),
            "encapsulation_positive_extrusion_mm": self._encapsulation_positive_extrusion_mm(),
            "step5_viewer_visibility": {
                "lead_frame_paths": bool(self.show_leadframe_paths_var.get()),
                "centered_die_compartment": bool(self.show_centered_die_compartment_var.get()),
                "silicon_die": bool(self.show_silicon_die_var.get()),
                "bond_assemblies": bool(self.show_bond_assemblies_var.get()),
                "scaled_outer_model": bool(self.show_scaled_outer_model_var.get()),
                "encapsulation": bool(self.show_encapsulation_var.get()),
            },
            "silicon_die_width_mm": self._silicon_die_width_mm(),
            "silicon_die_height_mm": self._silicon_die_height_mm(),
            "silicon_die_thickness_mm": self._silicon_die_thickness_mm(),
            "bond_end_region_size_mm": self._bond_end_region_size_mm(),
            "bond_end_region_offset_mm": self._bond_end_region_offset_mm(),
            "bond_start_region_size_mm": self._bond_start_region_size_mm(),
            "bond_start_region_gap_mm": self._bond_start_region_gap_mm(),
            "bond_start_region_offset_mm": self._bond_start_region_offset_mm(),
            "bond_start_region_counts": {
                "top": self._safe_int(self.bond_start_top_count_var, 2),
                "bottom": self._safe_int(self.bond_start_bottom_count_var, 2),
                "left": self._safe_int(self.bond_start_left_count_var, 2),
                "right": self._safe_int(self.bond_start_right_count_var, 2),
            },
            "guide_slots": [
                {
                    "side": slot.side_name,
                    "index": slot.index,
                    "center_mm": list(slot.center_mm),
                    "start_mm": list(slot.start_mm),
                    "end_mm": list(slot.end_mm),
                }
                for slot in self._guide_slots()
            ],
            "saved_paths_mm": [[list(point) for point in path] for path in self.saved_paths_mm],
            "current_path_mm": [list(point) for point in self.current_path_mm],
            "preview_point_mm": list(self.preview_point_mm) if self.preview_point_mm is not None else None,
            "wire_profiles": [dict(profile) for profile in self.wire_profiles],
            "wire_randomization": {
                "ball_radius_mean_mm": self._safe_float(self.rand_ball_radius_mean_var, 0.06),
                "ball_radius_std_mm": self._safe_float(self.rand_ball_radius_std_var, 0.01),
                "ball_height_mean_mm": self._safe_float(self.rand_ball_height_mean_var, 0.08),
                "ball_height_std_mm": self._safe_float(self.rand_ball_height_std_var, 0.015),
                "arc_height_mean_mm": self._safe_float(self.rand_arc_height_mean_var, 0.5),
                "arc_height_std_mm": self._safe_float(self.rand_arc_height_std_var, 0.08),
                "arc_xy_noise_mean_mm": self._safe_float(self.rand_arc_xy_noise_mean_var, 0.0),
                "arc_xy_noise_std_mm": self._safe_float(self.rand_arc_xy_noise_std_var, 0.05),
                "wire_rise_mean_mm": self._safe_float(self.rand_wire_rise_mean_var, 0.12),
                "wire_rise_std_mm": self._safe_float(self.rand_wire_rise_std_var, 0.02),
                "wire_diameter_mean_mm": self._safe_float(self.rand_wire_diameter_mean_var, 0.03),
                "wire_diameter_std_mm": self._safe_float(self.rand_wire_diameter_std_var, 0.004),
                "wedge_length_mean_mm": self._safe_float(self.rand_wedge_length_mean_var, 0.18),
                "wedge_length_std_mm": self._safe_float(self.rand_wedge_length_std_var, 0.02),
                "wedge_width_mean_mm": self._safe_float(self.rand_wedge_width_mean_var, 0.08),
                "wedge_width_std_mm": self._safe_float(self.rand_wedge_width_std_var, 0.01),
                "wedge_thickness_mean_mm": self._safe_float(self.rand_wedge_thickness_mean_var, 0.02),
                "wedge_thickness_std_mm": self._safe_float(self.rand_wedge_thickness_std_var, 0.003),
                "wedge_approach_mean_mm": self._safe_float(self.rand_wedge_approach_mean_var, 0.18),
                "wedge_approach_std_mm": self._safe_float(self.rand_wedge_approach_std_var, 0.02),
                "wedge_tail_mean_mm": self._safe_float(self.rand_wedge_tail_mean_var, 0.0),
                "wedge_tail_std_mm": self._safe_float(self.rand_wedge_tail_std_var, 0.02),
            },
        }

    def _stage_payload(self, stage_index: int) -> dict:
        payload = {
            "saved_stage_index": stage_index,
            "saved_stage_title": self.step_titles[stage_index],
            "current_step_index": self.step_index,
            "current_step_title": self.step_titles[self.step_index],
            "status_message": self.status_var.get(),
        }
        if stage_index >= 0:
            payload["lead_frame_design"] = self._project_payload()
        if stage_index >= 1:
            payload["silicon_die_and_bond_regions"] = {
                "silicon_die_width_mm": self._silicon_die_width_mm(),
                "silicon_die_height_mm": self._silicon_die_height_mm(),
                "silicon_die_thickness_mm": self._silicon_die_thickness_mm(),
                "bond_end_region_size_mm": self._bond_end_region_size_mm(),
                "bond_end_region_offset_mm": self._bond_end_region_offset_mm(),
                "bond_start_region_size_mm": self._bond_start_region_size_mm(),
                "bond_start_region_gap_mm": self._bond_start_region_gap_mm(),
                "bond_start_region_offset_mm": self._bond_start_region_offset_mm(),
                "bond_start_region_counts": {
                    "top": self._safe_int(self.bond_start_top_count_var, 2),
                    "bottom": self._safe_int(self.bond_start_bottom_count_var, 2),
                    "left": self._safe_int(self.bond_start_left_count_var, 2),
                    "right": self._safe_int(self.bond_start_right_count_var, 2),
                },
            }
        if stage_index >= 2:
            payload["wire_bonding"] = {
                "wire_profiles": [dict(profile) for profile in self.wire_profiles],
                "wire_randomization": {
                    "ball_radius_mean_mm": self._safe_float(self.rand_ball_radius_mean_var, 0.06),
                    "ball_radius_std_mm": self._safe_float(self.rand_ball_radius_std_var, 0.01),
                    "ball_height_mean_mm": self._safe_float(self.rand_ball_height_mean_var, 0.08),
                    "ball_height_std_mm": self._safe_float(self.rand_ball_height_std_var, 0.015),
                    "arc_height_mean_mm": self._safe_float(self.rand_arc_height_mean_var, 0.5),
                    "arc_height_std_mm": self._safe_float(self.rand_arc_height_std_var, 0.08),
                    "arc_xy_noise_mean_mm": self._safe_float(self.rand_arc_xy_noise_mean_var, 0.0),
                    "arc_xy_noise_std_mm": self._safe_float(self.rand_arc_xy_noise_std_var, 0.05),
                    "wire_rise_mean_mm": self._safe_float(self.rand_wire_rise_mean_var, 0.12),
                    "wire_rise_std_mm": self._safe_float(self.rand_wire_rise_std_var, 0.02),
                    "wire_diameter_mean_mm": self._safe_float(self.rand_wire_diameter_mean_var, 0.03),
                    "wire_diameter_std_mm": self._safe_float(self.rand_wire_diameter_std_var, 0.004),
                    "wedge_length_mean_mm": self._safe_float(self.rand_wedge_length_mean_var, 0.18),
                    "wedge_length_std_mm": self._safe_float(self.rand_wedge_length_std_var, 0.02),
                    "wedge_width_mean_mm": self._safe_float(self.rand_wedge_width_mean_var, 0.08),
                    "wedge_width_std_mm": self._safe_float(self.rand_wedge_width_std_var, 0.01),
                    "wedge_thickness_mean_mm": self._safe_float(self.rand_wedge_thickness_mean_var, 0.02),
                    "wedge_thickness_std_mm": self._safe_float(self.rand_wedge_thickness_std_var, 0.003),
                    "wedge_approach_mean_mm": self._safe_float(self.rand_wedge_approach_mean_var, 0.18),
                    "wedge_approach_std_mm": self._safe_float(self.rand_wedge_approach_std_var, 0.02),
                    "wedge_tail_mean_mm": self._safe_float(self.rand_wedge_tail_mean_var, 0.0),
                    "wedge_tail_std_mm": self._safe_float(self.rand_wedge_tail_std_var, 0.02),
                },
            }
        if stage_index >= 3:
            payload["scaled_outer_model"] = {
                "outer_model_scale_percent": max(0.0, self._safe_float(self.outer_model_scale_percent_var, 0.1)),
            }
        if stage_index >= 4:
            payload["encapsulation"] = {
                "encapsulation_width_mm": self._encapsulation_width_mm(),
                "encapsulation_length_mm": self._encapsulation_length_mm(),
                "encapsulation_negative_extrusion_mm": self._encapsulation_negative_extrusion_mm(),
                "encapsulation_positive_extrusion_mm": self._encapsulation_positive_extrusion_mm(),
                "step5_viewer_visibility": {
                    "lead_frame_paths": bool(self.show_leadframe_paths_var.get()),
                    "centered_die_compartment": bool(self.show_centered_die_compartment_var.get()),
                    "silicon_die": bool(self.show_silicon_die_var.get()),
                    "bond_assemblies": bool(self.show_bond_assemblies_var.get()),
                    "scaled_outer_model": bool(self.show_scaled_outer_model_var.get()),
                    "encapsulation": bool(self.show_encapsulation_var.get()),
                },
            }
        return payload

    def _autosave_project_files(self) -> None:
        self._ensure_current_project_dir()
        self._project_snapshot_path().write_text(json.dumps(self._project_payload(), indent=2), encoding="utf-8")
        for stage_index, stage_title in enumerate(self.step_titles):
            filename = f"stage_{stage_index + 1}_{_safe_stage_slug(stage_title)}.json"
            stage_path = self._project_stage_dir() / filename
            stage_path.write_text(json.dumps(self._stage_payload(stage_index), indent=2), encoding="utf-8")

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_outline()
        if self.step_index >= 4:
            self._draw_encapsulation_outline()
        self._draw_die_square()
        self._draw_guide_slots()
        self._draw_paths()

    def _on_close(self) -> None:
        if self.viewer_process is not None and self.viewer_process.poll() is None:
            self.viewer_process.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IC Chip Generator New")
    parser.add_argument("--viewer", type=str, default="", help="Launch the Step 2 vedo viewer with the given payload JSON path.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.viewer:
        IcChipGeneratorNewViewer(Path(args.viewer)).run()
    else:
        IcChipGeneratorNewApp().run()
