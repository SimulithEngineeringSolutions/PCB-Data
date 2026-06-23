from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tkinter as tk
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import trimesh
from vedo import Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED_PART_PATH = REPO_ROOT / "output" / "component_maker" / "ConnectorPin" / "Plastic Covering.stl"
DEFAULT_MOVING_PART_PATH = REPO_ROOT / "output" / "component_maker" / "ConnectorPin" / "connector Pin.stl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "female_pin_array_builder"
DEFAULT_PROJECT_PATH = DEFAULT_OUTPUT_DIR / "center_mate_project.json"
BRIDGE_DIR = DEFAULT_OUTPUT_DIR / "viewer_bridge"
PARALLEL_DOT_TOLERANCE = 0.995


@dataclass(slots=True)
class TrianglePick:
    face_index: int
    centroid_mm: list[float]
    normal: list[float]


@dataclass(slots=True)
class ParallelTrianglePair:
    triangle_a: TrianglePick
    triangle_b: TrianglePick
    center_point_mm: list[float]
    plane_normal: list[float]


@dataclass(slots=True)
class ArraySpec:
    count_x: int = 5
    count_y: int = 5
    pitch_x_mm: float = 2.54
    pitch_y_mm: float = 2.54


@dataclass(slots=True)
class CenterMateProject:
    moving_part_path: str
    fixed_part_path: str
    moving_pair: ParallelTrianglePair
    fixed_pair: ParallelTrianglePair
    center_mate_transform: list[list[float]]
    contact_moving_pair: ParallelTrianglePair | None
    contact_fixed_pair: ParallelTrianglePair | None
    full_transform: list[list[float]]
    array: ArraySpec
    instance_transforms: list[dict]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick parallel triangles to define a center mate between two STL parts.")
    parser.add_argument("--fixed-part", type=Path, default=DEFAULT_FIXED_PART_PATH)
    parser.add_argument("--moving-part", type=Path, default=DEFAULT_MOVING_PART_PATH)
    parser.add_argument("--viewer-bridge", type=Path, default=None, help="Internal: viewer bridge JSON file.")
    return parser.parse_args()


def load_mesh_mm(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected STL mesh at {path}, got {type(mesh)!r}.")
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    if float(max(mesh.extents)) < 1.0:
        mesh.apply_scale(1000.0)
    return mesh


def center_mesh_to_centroid(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    centered = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    centered.apply_translation(-centered.centroid)
    return centered


def normalize_vector(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector.")
    return [component / length for component in vector]


def dot(a_vec: list[float], b_vec: list[float]) -> float:
    return sum(a_val * b_val for a_val, b_val in zip(a_vec, b_vec))


def cross(a_vec: list[float], b_vec: list[float]) -> list[float]:
    return [
        (a_vec[1] * b_vec[2]) - (a_vec[2] * b_vec[1]),
        (a_vec[2] * b_vec[0]) - (a_vec[0] * b_vec[2]),
        (a_vec[0] * b_vec[1]) - (a_vec[1] * b_vec[0]),
    ]


def identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_multiply(a_mat: list[list[float]], b_mat: list[list[float]]) -> list[list[float]]:
    result = [[0.0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            result[row][col] = sum(a_mat[row][idx] * b_mat[idx][col] for idx in range(4))
    return result


def translation_transform(dx: float, dy: float, dz: float) -> list[list[float]]:
    matrix = identity_transform()
    matrix[0][3] = dx
    matrix[1][3] = dy
    matrix[2][3] = dz
    return matrix


def validate_array_spec(array_spec: ArraySpec) -> None:
    if array_spec.count_x <= 0 or array_spec.count_y <= 0:
        raise ValueError("Array counts must be positive integers.")
    if array_spec.pitch_x_mm <= 0.0 or array_spec.pitch_y_mm <= 0.0:
        raise ValueError("Array pitches must be positive.")


def compute_touching_pitch_mm(fixed_part_path: Path) -> tuple[float, float]:
    fixed_mesh = center_mesh_to_centroid(load_mesh_mm(fixed_part_path))
    extent_x, extent_y = fixed_mesh.extents[0], fixed_mesh.extents[1]
    if extent_x <= 0.0 or extent_y <= 0.0:
        raise ValueError("Plastic covering extents must be positive to build a touching array.")
    return float(extent_x), float(extent_y)


def rotation_about_axis(axis: list[float], angle_rad: float) -> list[list[float]]:
    ax, ay, az = normalize_vector(axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus_cosine = 1.0 - cosine
    return [
        [
            cosine + (ax * ax * one_minus_cosine),
            (ax * ay * one_minus_cosine) - (az * sine),
            (ax * az * one_minus_cosine) + (ay * sine),
            0.0,
        ],
        [
            (ay * ax * one_minus_cosine) + (az * sine),
            cosine + (ay * ay * one_minus_cosine),
            (ay * az * one_minus_cosine) - (ax * sine),
            0.0,
        ],
        [
            (az * ax * one_minus_cosine) - (ay * sine),
            (az * ay * one_minus_cosine) + (ax * sine),
            cosine + (az * az * one_minus_cosine),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_align_vectors(source: list[float], target: list[float]) -> list[list[float]]:
    source_n = normalize_vector(source)
    target_n = normalize_vector(target)
    cosine = max(-1.0, min(1.0, dot(source_n, target_n)))

    if cosine > 1.0 - 1e-9:
        return identity_transform()

    if cosine < -1.0 + 1e-9:
        basis = [1.0, 0.0, 0.0] if abs(source_n[0]) < 0.9 else [0.0, 1.0, 0.0]
        axis = normalize_vector(cross(source_n, basis))
        return rotation_about_axis(axis, math.pi)

    axis = normalize_vector(cross(source_n, target_n))
    angle = math.acos(cosine)
    return rotation_about_axis(axis, angle)


def transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    x_coord, y_coord, z_coord = point
    return [
        (matrix[0][0] * x_coord) + (matrix[0][1] * y_coord) + (matrix[0][2] * z_coord) + matrix[0][3],
        (matrix[1][0] * x_coord) + (matrix[1][1] * y_coord) + (matrix[1][2] * z_coord) + matrix[1][3],
        (matrix[2][0] * x_coord) + (matrix[2][1] * y_coord) + (matrix[2][2] * z_coord) + matrix[2][3],
    ]


def apply_transform_to_mesh(mesh: trimesh.Trimesh, transform: list[list[float]]) -> trimesh.Trimesh:
    transformed = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    transformed.apply_transform(transform)
    return transformed


def triangle_pick_from_face(mesh: trimesh.Trimesh, face_index: int) -> TrianglePick:
    centroid = mesh.triangles_center[face_index].tolist()
    normal = normalize_vector(mesh.face_normals[face_index].tolist())
    return TrianglePick(face_index=face_index, centroid_mm=[float(value) for value in centroid], normal=normal)


def build_parallel_triangle_pair(mesh: trimesh.Trimesh, face_index_a: int, face_index_b: int) -> ParallelTrianglePair:
    triangle_a = triangle_pick_from_face(mesh, face_index_a)
    triangle_b = triangle_pick_from_face(mesh, face_index_b)

    parallel_dot = abs(dot(triangle_a.normal, triangle_b.normal))
    if parallel_dot < PARALLEL_DOT_TOLERANCE:
        raise ValueError(
            "Selected triangles are not parallel enough for a center mate.\n"
            f"Triangle {face_index_a} normal: {format_vector(triangle_a.normal)}\n"
            f"Triangle {face_index_b} normal: {format_vector(triangle_b.normal)}\n"
            f"|dot| = {parallel_dot:.4f}"
        )

    aligned_normal_b = triangle_b.normal
    if dot(triangle_a.normal, triangle_b.normal) < 0.0:
        aligned_normal_b = [-value for value in triangle_b.normal]

    plane_normal = normalize_vector([
        triangle_a.normal[idx] + aligned_normal_b[idx]
        for idx in range(3)
    ])
    center_point_mm = [
        (triangle_a.centroid_mm[idx] + triangle_b.centroid_mm[idx]) / 2.0
        for idx in range(3)
    ]
    return ParallelTrianglePair(
        triangle_a=triangle_a,
        triangle_b=triangle_b,
        center_point_mm=[float(value) for value in center_point_mm],
        plane_normal=plane_normal,
    )


def nearest_face_index(mesh: trimesh.Trimesh, picked_point_mm: list[float]) -> int:
    best_index = min(
        range(len(mesh.faces)),
        key=lambda idx: sum(
            (mesh.triangles_center[idx][axis] - picked_point_mm[axis]) ** 2
            for axis in range(3)
        ),
    )
    return int(best_index)


def triangle_overlay(mesh: trimesh.Trimesh, face_index: int, color: str) -> Mesh:
    vertices = mesh.vertices[mesh.faces[face_index]].tolist()
    overlay = Mesh([vertices, [[0, 1, 2]]]).c(color).alpha(1.0)
    overlay.linewidth(4)
    overlay.pickable(False)
    return overlay


def format_vector(vector: list[float]) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in vector) + ")"


def pick_parallel_triangles(mesh: trimesh.Trimesh, title: str) -> ParallelTrianglePair:
    plotter = Plotter(title=title, bg="#efe7d2", bg2="#f6f0e2", axes=1, size=(1100, 760))
    mesh_actor = Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c("#d4b08a").alpha(1.0)
    info = Text2D(
        "Click triangle 1, then triangle 2.\nBoth triangles must be parallel.\nPress Esc to cancel.",
        pos="top-left",
        s=0.75,
        c="#2d241f",
        bg=None,
        font="Courier",
    )
    picked_faces: list[int] = []
    overlays: list = []
    hover_actor = None
    hover_face_index: int | None = None
    result: dict[str, ParallelTrianglePair | Exception | None] = {"pair": None, "error": None}

    def update_info(message: str) -> None:
        info.text(message)
        plotter.render()

    def clear_hover() -> None:
        nonlocal hover_actor, hover_face_index
        if hover_actor is not None:
            plotter.remove(hover_actor)
            hover_actor = None
        hover_face_index = None

    def handle_hover(event) -> None:
        nonlocal hover_actor, hover_face_index
        if event.object is None or event.object is not mesh_actor:
            if hover_actor is not None:
                clear_hover()
                plotter.render()
            return

        picked_point = list(event.picked3d)
        if len(picked_point) != 3:
            return
        face_index = nearest_face_index(mesh, picked_point)
        if face_index in picked_faces:
            if hover_actor is not None:
                clear_hover()
                plotter.render()
            return
        if hover_face_index == face_index:
            return

        clear_hover()
        hover_face_index = face_index
        hover_actor = triangle_overlay(mesh, face_index, "#f2c94c")
        plotter.add(hover_actor)
        hover_pick = triangle_pick_from_face(mesh, face_index)
        if picked_faces:
            first_pick = triangle_pick_from_face(mesh, picked_faces[0])
            info.text(
                f"Hover face: {face_index}\n"
                f"Hover normal: {format_vector(hover_pick.normal)}\n"
                f"Selected face 1: {picked_faces[0]}\n"
                f"Selected normal 1: {format_vector(first_pick.normal)}\n"
                "Click triangle 2."
            )
        else:
            info.text(
                f"Hover face: {face_index}\n"
                f"Hover normal: {format_vector(hover_pick.normal)}\n"
                "Click triangle 1."
            )
        plotter.render()

    def handle_pick(event) -> None:
        if event.object is None or event.object is not mesh_actor:
            return
        picked_point = list(event.picked3d)
        if len(picked_point) != 3:
            return
        face_index = nearest_face_index(mesh, picked_point)
        if face_index in picked_faces:
            update_info("Triangle already selected. Pick a different triangle.")
            return

        picked_faces.append(face_index)
        picked_triangle = triangle_pick_from_face(mesh, face_index)
        color = "#0e0e0e" if len(picked_faces) == 1 else "#cc5a2a"
        overlay = triangle_overlay(mesh, face_index, color)
        overlays.append(overlay)
        clear_hover()
        plotter.add(overlay)
        plotter.render()

        if len(picked_faces) == 1:
            update_info(
                f"Triangle 1 selected: face {face_index}\n"
                f"Normal 1: {format_vector(picked_triangle.normal)}\n"
                "Click triangle 2."
            )
            return

        try:
            pair = build_parallel_triangle_pair(mesh, picked_faces[0], picked_faces[1])
            result["pair"] = pair
            update_info(
                "Parallel pair accepted.\n"
                f"Triangle 1 normal: {format_vector(pair.triangle_a.normal)}\n"
                f"Triangle 2 normal: {format_vector(pair.triangle_b.normal)}\n"
                f"Pair plane normal: {format_vector(pair.plane_normal)}\n"
                "Closing picker..."
            )
            plotter.close()
        except Exception as exc:
            result["error"] = exc
            update_info(f"{exc}\nSelection reset. Pick triangle 1 again.")
            picked_faces.clear()
            clear_hover()
            for actor in overlays:
                plotter.remove(actor)
            overlays.clear()
            plotter.render()

    def handle_key(event) -> None:
        if event.keypress == "Escape":
            result["error"] = RuntimeError("Triangle picking cancelled.")
            plotter.close()

    plotter.show(mesh_actor, info, zoom="tight", interactive=False)
    plotter.add_callback("MouseMove", handle_hover)
    plotter.add_callback("LeftButtonPress", handle_pick)
    plotter.add_callback("KeyPress", handle_key, enable_picking=False)
    plotter.interactive()

    if isinstance(result["error"], Exception):
        raise result["error"]
    if result["pair"] is None:
        raise RuntimeError("Triangle picking did not produce a valid parallel pair.")
    return result["pair"]


def solve_center_mate_transform(
    fixed_pair: ParallelTrianglePair,
    moving_pair: ParallelTrianglePair,
) -> list[list[float]]:
    rotation = rotation_align_vectors(moving_pair.plane_normal, fixed_pair.plane_normal)
    moved_center = transform_point(rotation, moving_pair.center_point_mm)
    offset_vector = [
        fixed_pair.center_point_mm[idx] - moved_center[idx]
        for idx in range(3)
    ]
    normal_distance = dot(offset_vector, fixed_pair.plane_normal)
    transform = matrix_multiply(
        translation_transform(
            fixed_pair.plane_normal[0] * normal_distance,
            fixed_pair.plane_normal[1] * normal_distance,
            fixed_pair.plane_normal[2] * normal_distance,
        ),
        rotation,
    )
    return transform


def solve_contact_mate_transform(
    current_transform: list[list[float]],
    fixed_pair: ParallelTrianglePair,
    moving_pair: ParallelTrianglePair,
) -> list[list[float]]:
    moved_center = transform_point(current_transform, moving_pair.center_point_mm)
    moved_normal = normalize_vector([
        sum(current_transform[row][col] * moving_pair.plane_normal[col] for col in range(3))
        for row in range(3)
    ])
    abs_dot = abs(dot(fixed_pair.plane_normal, moved_normal))
    if abs_dot < PARALLEL_DOT_TOLERANCE:
        raise ValueError(
            "Contact mate planes are not parallel after the center mate.\n"
            f"Fixed contact normal: {format_vector(fixed_pair.plane_normal)}\n"
            f"Moved contact normal: {format_vector(moved_normal)}\n"
            f"|dot| = {abs_dot:.4f}"
        )

    offset_vector = [
        fixed_pair.center_point_mm[idx] - moved_center[idx]
        for idx in range(3)
    ]
    normal_distance = dot(offset_vector, fixed_pair.plane_normal)
    return matrix_multiply(
        translation_transform(
            fixed_pair.plane_normal[0] * normal_distance,
            fixed_pair.plane_normal[1] * normal_distance,
            fixed_pair.plane_normal[2] * normal_distance,
        ),
        current_transform,
    )


def build_instance_transforms(full_transform: list[list[float]], array_spec: ArraySpec) -> list[dict]:
    validate_array_spec(array_spec)
    instances: list[dict] = []
    for index_x in range(array_spec.count_x):
        for index_y in range(array_spec.count_y):
            offset_x = index_x * array_spec.pitch_x_mm
            offset_y = index_y * array_spec.pitch_y_mm
            instance_transform = matrix_multiply(
                translation_transform(offset_x, offset_y, 0.0),
                full_transform,
            )
            instances.append(
                {
                    "index_x": index_x,
                    "index_y": index_y,
                    "translation_mm": [offset_x, offset_y, 0.0],
                    "transform": instance_transform,
                }
            )
    return instances


def build_center_mate_project(
    moving_part_path: Path,
    fixed_part_path: Path,
    moving_pair: ParallelTrianglePair,
    fixed_pair: ParallelTrianglePair,
    contact_moving_pair: ParallelTrianglePair | None = None,
    contact_fixed_pair: ParallelTrianglePair | None = None,
    array: ArraySpec | None = None,
) -> CenterMateProject:
    array_spec = array if array is not None else ArraySpec()
    center_transform = solve_center_mate_transform(fixed_pair=fixed_pair, moving_pair=moving_pair)
    full_transform = center_transform
    if contact_moving_pair is not None and contact_fixed_pair is not None:
        full_transform = solve_contact_mate_transform(
            current_transform=center_transform,
            fixed_pair=contact_fixed_pair,
            moving_pair=contact_moving_pair,
        )
    instances = build_instance_transforms(full_transform, array_spec)
    return CenterMateProject(
        moving_part_path=str(moving_part_path.expanduser().resolve()),
        fixed_part_path=str(fixed_part_path.expanduser().resolve()),
        moving_pair=moving_pair,
        fixed_pair=fixed_pair,
        center_mate_transform=center_transform,
        contact_moving_pair=contact_moving_pair,
        contact_fixed_pair=contact_fixed_pair,
        full_transform=full_transform,
        array=array_spec,
        instance_transforms=instances,
        notes=(
            "Both meshes are centered to their own centroids first. "
            "Center mate constrains the moving part along the chosen center-plane normal direction. "
            "Optional contact mate then adds a second along-normal contact constraint. "
            "Array instances replicate the fully mated unit in X and Y."
        ),
    )


def write_bridge_payload(bridge_path: Path, payload: dict) -> None:
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_bridge_payload(bridge_path: Path) -> dict:
    if not bridge_path.exists():
        return {}
    try:
        payload = json.loads(bridge_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_viewer_payload(
    moving_part_path: Path,
    fixed_part_path: Path,
    moving_pair: ParallelTrianglePair | None,
    fixed_pair: ParallelTrianglePair | None,
    contact_moving_pair: ParallelTrianglePair | None,
    contact_fixed_pair: ParallelTrianglePair | None,
    array: ArraySpec,
    status_message: str,
    summary_message: str,
) -> dict:
    return {
        "moving_part_path": str(moving_part_path),
        "fixed_part_path": str(fixed_part_path),
        "moving_pair": asdict(moving_pair) if moving_pair is not None else None,
        "fixed_pair": asdict(fixed_pair) if fixed_pair is not None else None,
        "contact_moving_pair": asdict(contact_moving_pair) if contact_moving_pair is not None else None,
        "contact_fixed_pair": asdict(contact_fixed_pair) if contact_fixed_pair is not None else None,
        "array": asdict(array),
        "status_message": status_message,
        "summary_message": summary_message,
    }


def load_center_mate_project(project_path: Path) -> CenterMateProject:
    payload = json.loads(project_path.expanduser().resolve().read_text(encoding="utf-8"))

    def load_pair(pair_payload: dict | None) -> ParallelTrianglePair | None:
        if pair_payload is None:
            return None
        return ParallelTrianglePair(
            triangle_a=TrianglePick(**pair_payload["triangle_a"]),
            triangle_b=TrianglePick(**pair_payload["triangle_b"]),
            center_point_mm=pair_payload["center_point_mm"],
            plane_normal=pair_payload["plane_normal"],
        )

    return CenterMateProject(
        moving_part_path=str(payload["moving_part_path"]),
        fixed_part_path=str(payload["fixed_part_path"]),
        moving_pair=load_pair(payload["moving_pair"]),
        fixed_pair=load_pair(payload["fixed_pair"]),
        center_mate_transform=payload["center_mate_transform"],
        contact_moving_pair=load_pair(payload.get("contact_moving_pair")),
        contact_fixed_pair=load_pair(payload.get("contact_fixed_pair")),
        full_transform=payload["full_transform"],
        array=ArraySpec(**payload.get("array", {})),
        instance_transforms=payload.get("instance_transforms", []),
        notes=str(payload.get("notes", "")),
    )


def pair_from_payload(pair_payload: dict | None) -> ParallelTrianglePair | None:
    if pair_payload is None:
        return None
    return ParallelTrianglePair(
        triangle_a=TrianglePick(**pair_payload["triangle_a"]),
        triangle_b=TrianglePick(**pair_payload["triangle_b"]),
        center_point_mm=pair_payload["center_point_mm"],
        plane_normal=pair_payload["plane_normal"],
    )


class CenterMateViewer:
    def __init__(self, bridge_path: Path) -> None:
        self.bridge_path = bridge_path
        self.plotter = Plotter(
            title="Female Pin Array Preview",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1200, 820),
        )
        self.info = Text2D("", pos="top-left", s=0.75, c="#2d241f", bg=None, font="Courier")
        self.actors: list = []
        self.last_signature: tuple | None = None

    def _payload_signature(self, payload: dict) -> tuple:
        return (
            payload.get("moving_part_path", ""),
            payload.get("fixed_part_path", ""),
            json.dumps(payload.get("moving_pair", {}), sort_keys=True),
            json.dumps(payload.get("fixed_pair", {}), sort_keys=True),
            json.dumps(payload.get("contact_moving_pair", {}), sort_keys=True),
            json.dumps(payload.get("contact_fixed_pair", {}), sort_keys=True),
            json.dumps(payload.get("array", {}), sort_keys=True),
            payload.get("status_message", ""),
            payload.get("summary_message", ""),
        )

    def _mesh_actor(self, mesh: trimesh.Trimesh, color: str, alpha: float) -> Mesh:
        return Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(color).alpha(alpha)

    def _build_scene(self, payload: dict) -> None:
        for actor in self.actors:
            self.plotter.remove(actor)
        self.actors.clear()

        status_message = str(payload.get("status_message", "")).strip()
        summary_message = str(payload.get("summary_message", "")).strip()

        try:
            fixed_mesh = center_mesh_to_centroid(load_mesh_mm(Path(str(payload["fixed_part_path"]))))
            moving_mesh = center_mesh_to_centroid(load_mesh_mm(Path(str(payload["moving_part_path"]))))

            moving_pair_payload = payload.get("moving_pair")
            fixed_pair_payload = payload.get("fixed_pair")
            if moving_pair_payload and fixed_pair_payload:
                moving_pair = pair_from_payload(moving_pair_payload)
                fixed_pair = pair_from_payload(fixed_pair_payload)
                transform = solve_center_mate_transform(fixed_pair=fixed_pair, moving_pair=moving_pair)
                contact_moving_payload = payload.get("contact_moving_pair")
                contact_fixed_payload = payload.get("contact_fixed_pair")
                if contact_moving_payload and contact_fixed_payload:
                    transform = solve_contact_mate_transform(
                        current_transform=transform,
                        fixed_pair=pair_from_payload(contact_fixed_payload),
                        moving_pair=pair_from_payload(contact_moving_payload),
                    )
                array_spec = ArraySpec(**payload.get("array", {}))
                instances = build_instance_transforms(transform, array_spec)
                preview_line = "Preview: center mate"
                if contact_moving_payload and contact_fixed_payload:
                    preview_line += " + contact mate"
                preview_line += f" with array {array_spec.count_x} x {array_spec.count_y}"
                for instance in instances:
                    fixed_actor = self._mesh_actor(fixed_mesh, "#202020", 0.45)
                    fixed_actor.pos(instance["translation_mm"])
                    moved_mesh = apply_transform_to_mesh(moving_mesh, instance["transform"])
                    moving_actor = self._mesh_actor(moved_mesh, "#d08a35", 1.0)
                    self.actors.extend([fixed_actor, moving_actor])
                    self.plotter += fixed_actor
                    self.plotter += moving_actor
            else:
                fixed_actor = self._mesh_actor(fixed_mesh, "#202020", 0.45)
                self.actors.append(fixed_actor)
                self.plotter += fixed_actor
                moving_actor = self._mesh_actor(moving_mesh, "#d08a35", 0.65)
                self.actors.append(moving_actor)
                self.plotter += moving_actor
                preview_line = "Preview: pick center-mate triangles to constrain the moving part"
        except Exception as exc:
            preview_line = f"Preview blocked: {exc}"

        self.info.text(
            "Female Pin Array Builder\n"
            f"{preview_line}\n"
            f"{status_message or 'Click Center Mate to pick triangles.'}\n"
            f"{summary_message or ''}"
        )
        self.plotter.render()

    def _on_timer(self, _event) -> None:
        payload = read_bridge_payload(self.bridge_path)
        if not payload:
            return
        signature = self._payload_signature(payload)
        if signature != self.last_signature:
            self._build_scene(payload)
            self.last_signature = signature

    def run(self) -> None:
        payload = read_bridge_payload(self.bridge_path)
        if not payload:
            raise RuntimeError(f"Viewer bridge payload missing: {self.bridge_path}")
        self.plotter.show(self.info, zoom="tight", interactive=False)
        self._build_scene(payload)
        self.last_signature = self._payload_signature(payload)
        self.plotter.add_callback("Timer", self._on_timer)
        self.plotter.timer_callback("create", dt=150)
        self.plotter.interactive()


class CenterMateControlPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fixed_part_path = args.fixed_part.expanduser().resolve()
        self.moving_part_path = args.moving_part.expanduser().resolve()
        self.moving_pair: ParallelTrianglePair | None = None
        self.fixed_pair: ParallelTrianglePair | None = None
        self.contact_moving_pair: ParallelTrianglePair | None = None
        self.contact_fixed_pair: ParallelTrianglePair | None = None

        self.root = tk.Tk()
        self.root.title("Female Pin Array Builder")
        self.root.geometry("760x620")
        self.root.minsize(720, 580)

        self.fixed_part_var = tk.StringVar(value=str(self.fixed_part_path))
        self.moving_part_var = tk.StringVar(value=str(self.moving_part_path))
        self.array_count_x_var = tk.IntVar(value=5)
        self.array_count_y_var = tk.IntVar(value=5)
        self.array_pitch_x_var = tk.DoubleVar(value=2.54)
        self.array_pitch_y_var = tk.DoubleVar(value=2.54)
        self.status_var = tk.StringVar(value="Click Center Mate to pick two parallel triangles on each part.")
        self.summary_var = tk.StringVar(value="")
        self.bridge_path = BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
        self.viewer_process: subprocess.Popen | None = None

        self._build_ui()
        self._bind_live_updates()
        self._start_viewer()
        self._refresh_summary()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Female Pin Array Builder", font=("Georgia", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                "Workflow:\n"
                "1. Click Center Mate.\n"
                "2. Pick 2 parallel triangles on the connector pin.\n"
                "3. Pick 2 parallel triangles on the plastic covering.\n"
                "4. The moving part becomes constrained along the resulting center-plane normal.\n"
                "5. Optionally click Contact Mate and repeat the same picking flow for a second contact constraint."
            ),
            justify="left",
            wraplength=680,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        parts_box = ttk.LabelFrame(outer, text="Parts", padding=10)
        parts_box.grid(row=2, column=0, sticky="ew")
        parts_box.columnconfigure(1, weight=1)

        ttk.Label(parts_box, text="Connector Pin").grid(row=0, column=0, sticky="w")
        ttk.Entry(parts_box, textvariable=self.moving_part_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(parts_box, text="Browse", command=self._browse_moving_part).grid(row=0, column=2, sticky="ew")

        ttk.Label(parts_box, text="Plastic Covering").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(parts_box, textvariable=self.fixed_part_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(parts_box, text="Browse", command=self._browse_fixed_part).grid(row=1, column=2, sticky="ew", pady=(8, 0))

        summary_box = ttk.LabelFrame(outer, text="Female Pin Mate State", padding=10)
        summary_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(summary_box, textvariable=self.summary_var, wraplength=680, justify="left").pack(anchor="w")

        array_box = ttk.LabelFrame(outer, text="Array", padding=10)
        array_box.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        array_box.columnconfigure(1, weight=1)
        array_box.columnconfigure(3, weight=1)
        ttk.Label(array_box, text="Count X").grid(row=0, column=0, sticky="w")
        ttk.Entry(array_box, textvariable=self.array_count_x_var).grid(row=0, column=1, sticky="ew", padx=(8, 16))
        ttk.Label(array_box, text="Count Y").grid(row=0, column=2, sticky="w")
        ttk.Entry(array_box, textvariable=self.array_count_y_var).grid(row=0, column=3, sticky="ew", padx=(8, 0))
        ttk.Label(array_box, text="Pitch X (auto mm)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(array_box, textvariable=self.array_pitch_x_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=(8, 16), pady=(8, 0)
        )
        ttk.Label(array_box, text="Pitch Y (auto mm)").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(array_box, textvariable=self.array_pitch_y_var, state="readonly").grid(
            row=1, column=3, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)
        actions.columnconfigure(4, weight=1)
        ttk.Button(actions, text="Center Mate", command=self._run_center_mate_pick_workflow).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Contact Mate", command=self._run_contact_mate_pick_workflow).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(actions, text="Load Mate JSON", command=self._load_project_json).grid(row=0, column=2, sticky="ew", padx=(0, 8))
        ttk.Button(actions, text="Save Mate JSON", command=self._save_project_json).grid(row=0, column=3, sticky="ew", padx=(0, 8))
        ttk.Button(actions, text="Reset", command=self._reset_state).grid(row=0, column=4, sticky="ew")

        ttk.Label(outer, textvariable=self.status_var, wraplength=680, foreground="#7a2f20").grid(
            row=6, column=0, sticky="w", pady=(12, 0)
        )

    def _bind_live_updates(self) -> None:
        for variable in (
            self.array_count_x_var,
            self.array_count_y_var,
        ):
            variable.trace_add("write", self._on_array_inputs_changed)

    def _on_array_inputs_changed(self, *_args) -> None:
        self._refresh_summary()

    def _sync_touching_pitch_vars(self) -> tuple[float, float]:
        pitch_x_mm, pitch_y_mm = compute_touching_pitch_mm(self.fixed_part_path)
        self.array_pitch_x_var.set(round(pitch_x_mm, 6))
        self.array_pitch_y_var.set(round(pitch_y_mm, 6))
        return pitch_x_mm, pitch_y_mm

    def _browse_fixed_part(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Select plastic covering STL",
            initialdir=str(self.fixed_part_path.parent),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if chosen:
            self.fixed_part_path = Path(chosen).expanduser().resolve()
            self.fixed_part_var.set(str(self.fixed_part_path))
            self.fixed_pair = None
            self.contact_fixed_pair = None
            self._refresh_summary()

    def _browse_moving_part(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Select connector pin STL",
            initialdir=str(self.moving_part_path.parent),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if chosen:
            self.moving_part_path = Path(chosen).expanduser().resolve()
            self.moving_part_var.set(str(self.moving_part_path))
            self.moving_pair = None
            self.contact_moving_pair = None
            self._refresh_summary()

    def _current_array_spec(self) -> ArraySpec:
        pitch_x_mm, pitch_y_mm = self._sync_touching_pitch_vars()
        return ArraySpec(
            count_x=int(self.array_count_x_var.get()),
            count_y=int(self.array_count_y_var.get()),
            pitch_x_mm=pitch_x_mm,
            pitch_y_mm=pitch_y_mm,
        )

    def _run_center_mate_pick_workflow(self) -> None:
        try:
            self.moving_part_path = Path(self.moving_part_var.get()).expanduser().resolve()
            self.fixed_part_path = Path(self.fixed_part_var.get()).expanduser().resolve()

            moving_mesh = center_mesh_to_centroid(load_mesh_mm(self.moving_part_path))
            self.status_var.set("Pick 2 parallel triangles on the connector pin.")
            self.root.update_idletasks()
            moving_pair = pick_parallel_triangles(moving_mesh, "Center Mate Pick: Connector Pin")

            fixed_mesh = center_mesh_to_centroid(load_mesh_mm(self.fixed_part_path))
            self.status_var.set("Pick 2 parallel triangles on the plastic covering.")
            self.root.update_idletasks()
            fixed_pair = pick_parallel_triangles(fixed_mesh, "Center Mate Pick: Plastic Covering")

            self.moving_pair = moving_pair
            self.fixed_pair = fixed_pair
            self.contact_moving_pair = None
            self.contact_fixed_pair = None
            self.status_var.set("Center mate pair selection completed.")
            self._refresh_summary()
        except Exception as exc:
            self.status_var.set(f"Center mate picking failed: {exc}")
            self._push_preview_payload()

    def _run_contact_mate_pick_workflow(self) -> None:
        if self.moving_pair is None or self.fixed_pair is None:
            self.status_var.set("Pick the center mate first before defining the contact mate.")
            return
        try:
            self.moving_part_path = Path(self.moving_part_var.get()).expanduser().resolve()
            self.fixed_part_path = Path(self.fixed_part_var.get()).expanduser().resolve()

            moving_mesh = center_mesh_to_centroid(load_mesh_mm(self.moving_part_path))
            self.status_var.set("Pick 2 parallel triangles on the connector pin for the contact mate.")
            self.root.update_idletasks()
            moving_pair = pick_parallel_triangles(moving_mesh, "Contact Mate Pick: Connector Pin")

            fixed_mesh = center_mesh_to_centroid(load_mesh_mm(self.fixed_part_path))
            self.status_var.set("Pick 2 parallel triangles on the plastic covering for the contact mate.")
            self.root.update_idletasks()
            fixed_pair = pick_parallel_triangles(fixed_mesh, "Contact Mate Pick: Plastic Covering")

            self.contact_moving_pair = moving_pair
            self.contact_fixed_pair = fixed_pair
            self.status_var.set("Contact mate pair selection completed.")
            self._refresh_summary()
        except Exception as exc:
            self.status_var.set(f"Contact mate picking failed: {exc}")
            self._push_preview_payload()

    def _refresh_summary(self) -> None:
        if self.moving_pair is None or self.fixed_pair is None:
            self.summary_var.set("No full center mate has been picked yet.")
            self._push_preview_payload()
            return

        try:
            project = build_center_mate_project(
                moving_part_path=self.moving_part_path,
                fixed_part_path=self.fixed_part_path,
                moving_pair=self.moving_pair,
                fixed_pair=self.fixed_pair,
                contact_moving_pair=self.contact_moving_pair,
                contact_fixed_pair=self.contact_fixed_pair,
                array=self._current_array_spec(),
            )
            transform_rows = [
                "[" + ", ".join(f"{value:.3f}" for value in row) + "]"
                for row in project.center_mate_transform[:3]
            ]
            full_transform_rows = [
                "[" + ", ".join(f"{value:.3f}" for value in row) + "]"
                for row in project.full_transform[:3]
            ]
            self.summary_var.set(
                "Connector pin pair: "
                f"faces {project.moving_pair.triangle_a.face_index}, {project.moving_pair.triangle_b.face_index}\n"
                "Connector pin normals: "
                f"{format_vector(project.moving_pair.triangle_a.normal)} and "
                f"{format_vector(project.moving_pair.triangle_b.normal)}\n"
                "Plastic covering pair: "
                f"faces {project.fixed_pair.triangle_a.face_index}, {project.fixed_pair.triangle_b.face_index}\n"
                "Plastic covering normals: "
                f"{format_vector(project.fixed_pair.triangle_a.normal)} and "
                f"{format_vector(project.fixed_pair.triangle_b.normal)}\n"
                "Center-plane normal: "
                + ", ".join(f"{value:.3f}" for value in project.fixed_pair.plane_normal)
                + "\nCenter-mate transform:\n"
                + "\n".join(transform_rows)
                + f"\nArray: {project.array.count_x} x {project.array.count_y}"
                + f" at pitch ({project.array.pitch_x_mm:.3f}, {project.array.pitch_y_mm:.3f}) mm"
                + f"\nInstances: {len(project.instance_transforms)}"
            )
            if project.contact_moving_pair is not None and project.contact_fixed_pair is not None:
                self.summary_var.set(
                    self.summary_var.get()
                    + "\nContact pin pair: "
                    + f"faces {project.contact_moving_pair.triangle_a.face_index}, {project.contact_moving_pair.triangle_b.face_index}\n"
                    + "Contact pin normals: "
                    + f"{format_vector(project.contact_moving_pair.triangle_a.normal)} and "
                    + f"{format_vector(project.contact_moving_pair.triangle_b.normal)}\n"
                    + "Contact plastic pair: "
                    + f"faces {project.contact_fixed_pair.triangle_a.face_index}, {project.contact_fixed_pair.triangle_b.face_index}\n"
                    + "Contact plastic normals: "
                    + f"{format_vector(project.contact_fixed_pair.triangle_a.normal)} and "
                    + f"{format_vector(project.contact_fixed_pair.triangle_b.normal)}\n"
                    + "Full transform after contact mate:\n"
                    + "\n".join(full_transform_rows)
                )
            self.status_var.set("Center mate solved.")
            if project.contact_moving_pair is not None and project.contact_fixed_pair is not None:
                self.status_var.set("Center mate + contact mate solved.")
        except Exception as exc:
            self.summary_var.set("Picked triangles exist, but the center mate could not be solved.")
            self.status_var.set(str(exc))

        self._push_preview_payload()

    def _save_project_json(self) -> None:
        if self.moving_pair is None or self.fixed_pair is None:
            messagebox.showerror("Save Blocked", "Pick the center mate first.", parent=self.root)
            return
        try:
            project = build_center_mate_project(
                moving_part_path=self.moving_part_path,
                fixed_part_path=self.fixed_part_path,
                moving_pair=self.moving_pair,
                fixed_pair=self.fixed_pair,
                contact_moving_pair=self.contact_moving_pair,
                contact_fixed_pair=self.contact_fixed_pair,
                array=self._current_array_spec(),
            )
        except Exception as exc:
            messagebox.showerror("Save Blocked", str(exc), parent=self.root)
            return

        project_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save center mate JSON",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            initialfile=DEFAULT_PROJECT_PATH.name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not project_path:
            return
        output_path = Path(project_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(project), indent=2), encoding="utf-8")
        self.status_var.set(f"Saved center mate project to {output_path}.")

    def _load_project_json(self) -> None:
        project_path = filedialog.askopenfilename(
            parent=self.root,
            title="Load center mate JSON",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not project_path:
            return
        try:
            project = load_center_mate_project(Path(project_path))
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc), parent=self.root)
            self.status_var.set(f"Load failed: {exc}")
            return

        self.moving_part_path = Path(project.moving_part_path).expanduser().resolve()
        self.fixed_part_path = Path(project.fixed_part_path).expanduser().resolve()
        self.moving_part_var.set(str(self.moving_part_path))
        self.fixed_part_var.set(str(self.fixed_part_path))
        self.moving_pair = project.moving_pair
        self.fixed_pair = project.fixed_pair
        self.contact_moving_pair = project.contact_moving_pair
        self.contact_fixed_pair = project.contact_fixed_pair
        self.array_count_x_var.set(project.array.count_x)
        self.array_count_y_var.set(project.array.count_y)
        self.array_pitch_x_var.set(project.array.pitch_x_mm)
        self.array_pitch_y_var.set(project.array.pitch_y_mm)
        self.status_var.set(f"Loaded center mate project from {project_path}.")
        self._refresh_summary()

    def _reset_state(self) -> None:
        self.moving_part_path = DEFAULT_MOVING_PART_PATH
        self.fixed_part_path = DEFAULT_FIXED_PART_PATH
        self.moving_part_var.set(str(self.moving_part_path))
        self.fixed_part_var.set(str(self.fixed_part_path))
        self.moving_pair = None
        self.fixed_pair = None
        self.contact_moving_pair = None
        self.contact_fixed_pair = None
        self.array_count_x_var.set(5)
        self.array_count_y_var.set(5)
        self.array_pitch_x_var.set(2.54)
        self.array_pitch_y_var.set(2.54)
        self.status_var.set("Reset the center mate state.")
        self._refresh_summary()

    def _push_preview_payload(self) -> None:
        try:
            write_bridge_payload(
                self.bridge_path,
                build_viewer_payload(
                    moving_part_path=Path(self.moving_part_var.get()).expanduser().resolve(),
                    fixed_part_path=Path(self.fixed_part_var.get()).expanduser().resolve(),
                    moving_pair=self.moving_pair,
                    fixed_pair=self.fixed_pair,
                    contact_moving_pair=self.contact_moving_pair,
                    contact_fixed_pair=self.contact_fixed_pair,
                    array=self._current_array_spec(),
                    status_message=self.status_var.get(),
                    summary_message=self.summary_var.get(),
                ),
            )
        except Exception as exc:
            self.status_var.set(f"Viewer sync failed: {exc}")

    def _start_viewer(self) -> None:
        try:
            self._push_preview_payload()
            self.viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--viewer-bridge",
                    str(self.bridge_path),
                ],
                cwd=str(REPO_ROOT),
            )
            self.status_var.set("Viewer launched.")
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

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    args = parse_args()
    if args.viewer_bridge is not None:
        viewer = CenterMateViewer(args.viewer_bridge.expanduser().resolve())
        viewer.run()
        return 0
    panel = CenterMateControlPanel(args)
    return panel.run()


if __name__ == "__main__":
    raise SystemExit(main())
