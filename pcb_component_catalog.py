from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tkinter as tk
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import trimesh
from vedo import Box, Line, Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PCB = REPO_ROOT / "DataSet" / "KICAD" / "Arduino hat" / "Arduino_hat.kicad_pcb"
HEIGHT_OVERRIDE_PATH = REPO_ROOT / "output" / "component_height_overrides.json"
PLACEMENT_OVERRIDE_PATH = REPO_ROOT / "output" / "component_placement_overrides.json"
ASSET_ROTATION_OVERRIDE_PATH = REPO_ROOT / "output" / "component_asset_rotation_overrides.json"
BOARD_MESH_OVERRIDE_PATH = REPO_ROOT / "output" / "board_mesh_transform_overrides.json"
COMPONENT_REFERENCE_ASSET_OVERRIDE_PATH = REPO_ROOT / "output" / "component_reference_asset_overrides.json"
BOUNDING_BOX_ONLY_ASSET_OVERRIDE = "__BOUNDING_BOX_ONLY__"
VIEWER_BRIDGE_DIR = REPO_ROOT / "output" / "component_viewer_bridge"
COMPONENT_LIBRARY_DIR = REPO_ROOT / "output" / "component_maker" / "library"
FEMALE_PIN_1X1_DIR = REPO_ROOT / "output" / "component_maker" / "FemalePinConnector1x1"
FEMALE_PIN_HOLDER_PATH = FEMALE_PIN_1X1_DIR / "female_pin_holder.stl"
FEMALE_PIN_CONTACT_PATH = FEMALE_PIN_1X1_DIR / "female_pin_internal_contact.stl"
FEMALE_PIN_ARRAY_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "female_pin_arrays"
MATE_BUILDER_PROJECT_PATH = REPO_ROOT / "output" / "component_maker" / "female_pin_array_builder" / "center_mate_project.json"
LEGACY_MATE_BUILDER_PROJECT_PATH = REPO_ROOT / "output" / "component_maker" / "mate_builder" / "center_mate_project.json"


@dataclass(slots=True)
class ComponentRecord:
    reference: str
    value: str
    footprint: str
    layer: str
    x_mm: float
    y_mm: float
    rotation_deg: float
    body_width_mm: float
    body_height_mm: float
    body_thickness_mm: float
    body_center_offset_x_mm: float
    body_center_offset_y_mm: float
    placement_offset_x_mm: float
    placement_offset_y_mm: float
    pad_span_width_mm: float
    pad_span_height_mm: float
    pad_count: int
    model_path: str
    bounds_source: str
    height_source: str


@dataclass(slots=True)
class OutlineSegment:
    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    mid: tuple[float, float] | None = None


@dataclass(slots=True)
class BoardScene:
    board_path: Path
    components: list[ComponentRecord]
    outline_segments: list[OutlineSegment]
    board_bounds: tuple[float, float, float, float]


@dataclass(slots=True)
class ComponentAssetPart:
    part_name: str
    stl_path: Path
    color: str
    height_mm: float


@dataclass(slots=True)
class ComponentAssetDefinition:
    component_family: str
    component_name: str
    definition_path: Path
    native_bbox_mm: tuple[float, float, float]
    native_center_mm: tuple[float, float, float]
    native_rotation_deg: tuple[float, float, float]
    fit_part_names: tuple[str, ...]
    parts: list[ComponentAssetPart]


@dataclass(slots=True)
class BoardMeshAsset:
    name: str
    path: Path


@dataclass(slots=True)
class BoardMeshPackage:
    manifest_path: Path
    source_board: str
    meshes: list[BoardMeshAsset]


def infer_component_family(component: ComponentRecord) -> str:
    reference = component.reference.upper()
    footprint = component.footprint.upper()
    value = component.value.upper()
    if reference.startswith("C") or "CAPACITOR" in footprint or value.endswith("F"):
        return "capacitor"
    if reference.startswith("R") or "RESISTOR" in footprint or value.endswith("OHM") or "OHM" in value:
        return "resistor"
    return "generic"


def slugify_board_name(board_path: Path) -> str:
    raw_name = board_path.stem.strip().lower()
    return "".join(char if char.isalnum() else "_" for char in raw_name).strip("_") or "board"


def load_board_mesh_package(manifest_path: Path) -> BoardMeshPackage | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if "variants" in payload and isinstance(payload.get("variants"), dict):
        variants = payload["variants"]
        preferred_path_text = str(variants.get("defect") or variants.get("clean") or "").strip()
        if not preferred_path_text:
            return None
        return load_board_mesh_package(Path(preferred_path_text).expanduser().resolve())
    meshes_payload = payload.get("meshes", [])
    if not isinstance(meshes_payload, list):
        return None
    meshes: list[BoardMeshAsset] = []
    for item in meshes_payload:
        if not isinstance(item, dict):
            continue
        mesh_name = str(item.get("name", "")).strip()
        mesh_path_text = str(item.get("path", "")).strip()
        if not mesh_name or not mesh_path_text:
            continue
        mesh_path = Path(mesh_path_text).expanduser().resolve()
        if mesh_path.suffix.lower() != ".stl" or not mesh_path.exists():
            continue
        meshes.append(BoardMeshAsset(name=mesh_name, path=mesh_path))
    if not meshes:
        return None
    return BoardMeshPackage(
        manifest_path=manifest_path.expanduser().resolve(),
        source_board=str(payload.get("source", "")).strip(),
        meshes=meshes,
    )


def auto_detect_board_mesh_package(board_path: Path) -> BoardMeshPackage | None:
    board_key = slugify_board_name(board_path)
    candidates = [
        REPO_ROOT / "output" / board_key / "material_partition_defects" / "material_partition_manifest.json",
        REPO_ROOT / "output" / board_key / "material_partition" / "material_partition_manifest.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        package = load_board_mesh_package(candidate)
        if package is not None:
            return package
    return None


def load_component_asset_library() -> dict[str, list[ComponentAssetDefinition]]:
    library: dict[str, list[ComponentAssetDefinition]] = {}
    if not COMPONENT_LIBRARY_DIR.exists():
        return library
    for definition_path in sorted(COMPONENT_LIBRARY_DIR.glob("*.json")):
        try:
            payload = json.loads(definition_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        component_family = str(payload.get("component_family", "")).strip().lower()
        component_name = str(payload.get("component_name", definition_path.stem)).strip()
        native_bbox = payload.get("native_bbox_mm", {})
        native_center = payload.get("native_center_mm", {})
        native_rotation = payload.get("native_rotation_deg", {})
        fit_part_names_payload = payload.get("fit_part_names", [])
        parts_payload = payload.get("parts", [])
        if not component_family or not isinstance(parts_payload, list):
            continue
        parts: list[ComponentAssetPart] = []
        for item in parts_payload:
            if not isinstance(item, dict):
                continue
            stl_path_text = str(item.get("stl_path", "")).strip()
            if not stl_path_text:
                continue
            stl_path = (definition_path.parent / stl_path_text).resolve()
            if not stl_path.exists():
                continue
            parts.append(
                ComponentAssetPart(
                    part_name=str(item.get("part_name", stl_path.stem)),
                    stl_path=stl_path,
                    color=str(item.get("color", "#b8b8b8")),
                    height_mm=float(item.get("height_mm", 0.0)),
                )
            )
        if not parts:
            continue
        asset = ComponentAssetDefinition(
            component_family=component_family,
            component_name=component_name,
            definition_path=definition_path,
            native_bbox_mm=(
                float(native_bbox.get("x", 1.0)),
                float(native_bbox.get("y", 1.0)),
                float(native_bbox.get("z", 1.0)),
            ),
            native_center_mm=(
                float(native_center.get("x", 0.0)),
                float(native_center.get("y", 0.0)),
                float(native_center.get("z", 0.0)),
            ),
            native_rotation_deg=(
                float(native_rotation.get("x", 0.0)),
                float(native_rotation.get("y", 0.0)),
                float(native_rotation.get("z", 0.0)),
            ),
            fit_part_names=tuple(
                str(item).strip()
                for item in fit_part_names_payload
                if str(item).strip()
            ),
            parts=parts,
        )
        library.setdefault(component_family, []).append(asset)
    return library


def rotate_point_xyz(point_xyz: tuple[float, float, float], rotation_deg_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x_coord, y_coord, z_coord = point_xyz
    rot_x = math.radians(rotation_deg_xyz[0])
    rot_y = math.radians(rotation_deg_xyz[1])
    rot_z = math.radians(rotation_deg_xyz[2])

    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    y_coord, z_coord = (y_coord * cos_x) - (z_coord * sin_x), (y_coord * sin_x) + (z_coord * cos_x)

    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    x_coord, z_coord = (x_coord * cos_y) + (z_coord * sin_y), (-x_coord * sin_y) + (z_coord * cos_y)

    cos_z, sin_z = math.cos(rot_z), math.sin(rot_z)
    x_coord, y_coord = (x_coord * cos_z) - (y_coord * sin_z), (x_coord * sin_z) + (y_coord * cos_z)
    return (x_coord, y_coord, z_coord)


def load_mesh_mm(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected STL mesh at {path}, got {type(mesh)!r}.")
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    if float(max(mesh.extents)) < 1.0:
        mesh.apply_scale(1000.0)
    return mesh


def translated_mesh_copy(mesh: trimesh.Trimesh, dx: float, dy: float, dz: float = 0.0) -> trimesh.Trimesh:
    copied = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    copied.apply_translation([dx, dy, dz])
    return copied


def concatenate_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("No meshes were provided for concatenation.")
    if len(meshes) == 1:
        mesh = meshes[0]
        return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    merged = trimesh.util.concatenate(meshes)
    return trimesh.Trimesh(vertices=merged.vertices.copy(), faces=merged.faces.copy(), process=False)


def scale_mesh_xyz(mesh: trimesh.Trimesh, scale_x: float, scale_y: float, scale_z: float) -> trimesh.Trimesh:
    scaled = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    scaled.vertices[:, 0] *= scale_x
    scaled.vertices[:, 1] *= scale_y
    scaled.vertices[:, 2] *= scale_z
    return scaled


def rotate_mesh_xyz(mesh: trimesh.Trimesh, rotation_deg_xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    rotated = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    rotated_vertices = [
        list(
            rotate_point_xyz(
                (float(vertex[0]), float(vertex[1]), float(vertex[2])),
                rotation_deg_xyz,
            )
        )
        for vertex in rotated.vertices.tolist()
    ]
    rotated.vertices = rotated_vertices
    return rotated


def shift_mesh_to_negative_z_contact(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    shifted = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    max_z = float(shifted.bounds[1][2])
    shifted.apply_translation([0.0, 0.0, -max_z])
    return shifted


def center_mesh_to_centroid(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    centered = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    centered.apply_translation(-centered.centroid)
    return centered


def apply_transform_to_mesh(mesh: trimesh.Trimesh, transform: list[list[float]]) -> trimesh.Trimesh:
    transformed = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    transformed.apply_transform(transform)
    return transformed


def parse_array_size(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if match is None:
        raise ValueError("Enter array size like 2x5.")
    count_x = int(match.group(1))
    count_y = int(match.group(2))
    if count_x <= 0 or count_y <= 0:
        raise ValueError("Array counts must be positive integers.")
    return count_x, count_y


def parse_rotation_xyz(text: str) -> tuple[float, float, float]:
    cleaned = text.strip()
    if not cleaned:
        return (0.0, 0.0, 0.0)
    parts = [part.strip() for part in re.split(r"[, ]+", cleaned) if part.strip()]
    if len(parts) == 1:
        value = float(parts[0])
        return (0.0, 0.0, value)
    if len(parts) != 3:
        raise ValueError("Enter rotation like 90 or 0,0,90.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def safe_slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or fallback


def color_for_name(name: str) -> str:
    palette = [
        "#202020",
        "#b87333",
        "#6d6875",
        "#537d5d",
        "#4a7a96",
        "#b17a3e",
        "#c14953",
        "#7d8a91",
    ]
    return palette[sum(ord(char) for char in name) % len(palette)]


def build_female_pin_array_meshes(
    count_x: int,
    count_y: int,
    *,
    pitch_x_mm: float = 2.54,
    pitch_y_mm: float = 2.54,
) -> dict[str, trimesh.Trimesh]:
    if not FEMALE_PIN_HOLDER_PATH.exists() or not FEMALE_PIN_CONTACT_PATH.exists():
        raise FileNotFoundError(
            "Female pin 1x1 STL assets were not found. Generate them first in "
            f"{FEMALE_PIN_1X1_DIR}."
        )
    holder_mesh = load_mesh_mm(FEMALE_PIN_HOLDER_PATH)
    contact_mesh = load_mesh_mm(FEMALE_PIN_CONTACT_PATH)
    holder_instances: list[trimesh.Trimesh] = []
    contact_instances: list[trimesh.Trimesh] = []
    x_origin = ((count_x - 1) * pitch_x_mm) / 2.0
    y_origin = ((count_y - 1) * pitch_y_mm) / 2.0
    for x_index in range(count_x):
        for y_index in range(count_y):
            offset_x = (x_index * pitch_x_mm) - x_origin
            offset_y = (y_index * pitch_y_mm) - y_origin
            holder_instances.append(translated_mesh_copy(holder_mesh, offset_x, offset_y, 0.0))
            contact_instances.append(translated_mesh_copy(contact_mesh, offset_x, offset_y, 0.0))
    holder_array = concatenate_meshes(holder_instances)
    contact_array = concatenate_meshes(contact_instances)
    assembly_array = concatenate_meshes([holder_array, contact_array])
    return {
        "holder": holder_array,
        "contact": contact_array,
        "assembly": assembly_array,
    }


def load_mate_builder_project_payload(project_path: Path) -> dict:
    payload = json.loads(project_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mate builder project JSON must contain an object payload.")
    return payload


def resolve_mate_builder_project_path() -> Path:
    if MATE_BUILDER_PROJECT_PATH.exists():
        return MATE_BUILDER_PROJECT_PATH
    return LEGACY_MATE_BUILDER_PROJECT_PATH


def build_female_pin_array_meshes_from_mate_project(
    project_path: Path,
    count_x: int,
    count_y: int,
) -> dict[str, trimesh.Trimesh]:
    payload = load_mate_builder_project_payload(project_path)
    moving_part_path = Path(str(payload["moving_part_path"])).expanduser().resolve()
    fixed_part_path = Path(str(payload["fixed_part_path"])).expanduser().resolve()
    full_transform = payload["full_transform"]
    array_payload = payload.get("array", {}) if isinstance(payload.get("array", {}), dict) else {}
    pitch_x_mm = float(array_payload.get("pitch_x_mm", 2.54))
    pitch_y_mm = float(array_payload.get("pitch_y_mm", 2.54))
    fixed_mesh = center_mesh_to_centroid(load_mesh_mm(fixed_part_path))
    moving_mesh = center_mesh_to_centroid(load_mesh_mm(moving_part_path))
    holder_instances: list[trimesh.Trimesh] = []
    contact_instances: list[trimesh.Trimesh] = []
    x_origin = ((count_x - 1) * pitch_x_mm) / 2.0
    y_origin = ((count_y - 1) * pitch_y_mm) / 2.0
    for x_index in range(count_x):
        for y_index in range(count_y):
            offset_x = (x_index * pitch_x_mm) - x_origin
            offset_y = (y_index * pitch_y_mm) - y_origin
            holder_instances.append(translated_mesh_copy(fixed_mesh, offset_x, offset_y, 0.0))
            moved_instance = apply_transform_to_mesh(moving_mesh, full_transform)
            moved_instance.apply_translation([offset_x, offset_y, 0.0])
            contact_instances.append(moved_instance)
    holder_array = concatenate_meshes(holder_instances)
    contact_array = concatenate_meshes(contact_instances)
    assembly_array = concatenate_meshes([holder_array, contact_array])
    return {
        "holder": holder_array,
        "contact": contact_array,
        "assembly": assembly_array,
    }


def fit_female_pin_array_to_component(
    meshes: dict[str, trimesh.Trimesh],
    component: ComponentRecord,
    rotation_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, trimesh.Trimesh]:
    rotated_meshes = {
        name: rotate_mesh_xyz(mesh, rotation_deg_xyz)
        for name, mesh in meshes.items()
    }
    holder_bounds = rotated_meshes["holder"].bounds
    extent_x = max(float(holder_bounds[1][0] - holder_bounds[0][0]), 1e-6)
    extent_y = max(float(holder_bounds[1][1] - holder_bounds[0][1]), 1e-6)
    extent_z = max(float(holder_bounds[1][2] - holder_bounds[0][2]), 1e-6)
    scale_x = max(component.body_width_mm, 0.2) / extent_x
    scale_y = max(component.body_height_mm, 0.2) / extent_y
    scale_z = max(component.body_thickness_mm, 0.2) / extent_z
    fitted = {
        name: scale_mesh_xyz(mesh, scale_x, scale_y, scale_z)
        for name, mesh in rotated_meshes.items()
    }
    holder_max_z = float(fitted["holder"].bounds[1][2])
    result: dict[str, trimesh.Trimesh] = {}
    for name, mesh in fitted.items():
        shifted = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        shifted.apply_translation([0.0, 0.0, -holder_max_z])
        result[name] = shifted
    return result


class ViewerEntityBrowser:
    def __init__(self, board_name: str, on_toggle: callable) -> None:
        self.root = tk.Tk()
        self.root.title(f"{board_name} Displayed Entities")
        self.root.geometry("360x620")
        self.root.minsize(280, 320)
        self.closed = False
        self.on_toggle = on_toggle
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Displayed Entities", font=("Georgia", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(frame, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = ttk.Frame(self.canvas, padding=2)
        self.content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.vars: dict[str, tk.BooleanVar] = {}
        self.parent_by_key: dict[str, str | None] = {}
        self.children_by_parent: dict[str, list[str]] = {}
        self.widgets_by_key: dict[str, ttk.Checkbutton] = {}

    def set_entities(
        self,
        entities: list[tuple[str, str, list[tuple[str, str]]]],
        visibility: dict[str, bool],
    ) -> None:
        if self.closed:
            return
        for child in self.content.winfo_children():
            child.destroy()
        self.parent_by_key.clear()
        self.children_by_parent.clear()
        self.widgets_by_key.clear()
        row = 0
        for parent_key, parent_label, child_items in entities:
            parent_var = self.vars.setdefault(parent_key, tk.BooleanVar(value=visibility.get(parent_key, True)))
            parent_var.set(bool(visibility.get(parent_key, True)))
            parent_button = ttk.Checkbutton(
                self.content,
                text=parent_label,
                variable=parent_var,
                command=lambda key=parent_key: self._on_parent_toggle(key),
            )
            parent_button.grid(row=row, column=0, sticky="w", pady=(2, 0))
            self.widgets_by_key[parent_key] = parent_button
            self.parent_by_key[parent_key] = None
            self.children_by_parent[parent_key] = []
            row += 1
            for child_key, child_label in child_items:
                child_var = self.vars.setdefault(child_key, tk.BooleanVar(value=visibility.get(child_key, True)))
                child_var.set(bool(visibility.get(child_key, True)))
                child_button = ttk.Checkbutton(
                    self.content,
                    text=child_label,
                    variable=child_var,
                    command=lambda key=child_key: self._on_child_toggle(key),
                )
                child_button.grid(row=row, column=0, sticky="w", padx=(24, 0), pady=(2, 0))
                self.widgets_by_key[child_key] = child_button
                self.parent_by_key[child_key] = parent_key
                self.children_by_parent[parent_key].append(child_key)
                row += 1
            self._sync_child_widget_states(parent_key)

    def _sync_child_widget_states(self, parent_key: str) -> None:
        parent_enabled = bool(self.vars.get(parent_key).get()) if parent_key in self.vars else True
        for child_key in self.children_by_parent.get(parent_key, []):
            widget = self.widgets_by_key.get(child_key)
            if widget is None:
                continue
            if parent_enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])

    def _on_parent_toggle(self, parent_key: str) -> None:
        is_visible = bool(self.vars[parent_key].get())
        self._sync_child_widget_states(parent_key)
        self.on_toggle(parent_key, is_visible)

    def _on_child_toggle(self, child_key: str) -> None:
        is_visible = bool(self.vars[child_key].get())
        self.on_toggle(child_key, is_visible)

    def _on_content_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self.content_window, width=event.width)

    def pump(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class Component3DViewer:
    def __init__(
        self,
        scene: BoardScene,
        asset_library: dict[str, list[ComponentAssetDefinition]] | None = None,
        board_mesh_package: BoardMeshPackage | None = None,
        bridge_path: Path | None = None,
    ) -> None:
        self.scene = scene
        self.asset_library = asset_library or {}
        self.board_mesh_package = board_mesh_package
        self.bridge_path = bridge_path
        self.plotter = Plotter(
            title=f"{scene.board_path.stem} Component Viewer",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1400, 900),
        )
        self.info = Text2D("", pos="top-left", s=0.8, c="#2d241f", bg=None, font="Courier")
        self.actors: list = []
        self.component_actor_by_reference: dict[str, list[object]] = {}
        self.reference_by_actor_id: dict[int, str] = {}
        self.actor_base_color: dict[int, str] = {}
        self.selected_reference: str | None = None
        self.is_shown = False
        self._last_bridge_selected: str | None = None
        self.asset_mesh_cache: dict[str, trimesh.Trimesh] = {}
        self._last_rotation_override_signature = ""
        self._last_board_mesh_transform_signature = ""
        self._last_component_asset_override_signature = ""
        self._last_click_reference: str | None = None
        self._last_click_time = 0.0
        self.entity_browser: ViewerEntityBrowser | None = None
        self.entity_tree_payload: list[tuple[str, str, list[tuple[str, str]]]] = []
        self.entity_visibility: dict[str, bool] = {}
        self.entity_parent_by_key: dict[str, str | None] = {}
        self.entity_actor_groups: dict[str, list[object]] = {}
        self.asset_definition_cache: dict[str, ComponentAssetDefinition] = {}

    def _component_color(self, component: ComponentRecord) -> str:
        footprint = component.footprint.upper()
        if "CONNECTOR" in footprint or "PINHEADER" in footprint or "PINSOCKET" in footprint:
            return "#537d5d"
        if "CRYSTAL" in footprint:
            return "#b17a3e"
        if footprint.startswith("LED") or "LED_" in footprint:
            return "#c14953"
        if "SOIC" in footprint or "SOIJ" in footprint or "PLCC" in footprint:
            return "#6d6875"
        return "#4a7a96"

    def _build_outline_actors(self) -> list:
        actors: list = []
        for segment in self.scene.outline_segments:
            points = outline_segment_points(segment)
            coords = [(x, y, 0.0) for x, y in points]
            actors.append(Line(coords).c("#2d241f").lw(3))
        self._register_entity_group("pcb::outline", actors, "pcb")
        return actors

    def _component_display_name(self, component: ComponentRecord) -> str:
        value_text = component.value.strip() if component.value.strip() else component.footprint
        return f"{component.reference} {value_text}"

    def _register_entity_group(self, key: str, actors: list[object], parent_key: str | None = None) -> None:
        self.entity_parent_by_key[key] = parent_key
        self.entity_actor_groups[key] = actors
        self.entity_visibility.setdefault(key, True)

    def _entity_effective_visible(self, key: str) -> bool:
        current_key: str | None = key
        while current_key is not None:
            if not self.entity_visibility.get(current_key, True):
                return False
            current_key = self.entity_parent_by_key.get(current_key)
        return True

    def _set_actor_visible(self, actor: object, visible: bool) -> None:
        try:
            if visible:
                actor.on()
            else:
                actor.off()
            return
        except Exception:
            pass
        try:
            actor.SetVisibility(1 if visible else 0)
        except Exception:
            pass

    def _apply_entity_visibility(self) -> None:
        for key, actors in self.entity_actor_groups.items():
            is_visible = self._entity_effective_visible(key)
            for actor in actors:
                self._set_actor_visible(actor, is_visible)

    def _on_entity_visibility_changed(self, key: str, is_visible: bool) -> None:
        self.entity_visibility[key] = is_visible
        self._apply_entity_visibility()
        if self.is_shown:
            self.plotter.render()

    def _board_mesh_color(self, mesh_name: str) -> str:
        name = mesh_name.lower()
        if "fr4" in name:
            return "#2d7a52"
        if "solder_mask" in name:
            return "#245d3c"
        if "copper" in name or "trace" in name or "pad_barrels" in name:
            return "#b87333"
        if "pad_air" in name:
            return "#d9d1c4"
        return "#7d8a91"

    def _board_mesh_alignment_offset_xy(self) -> tuple[float, float]:
        if self.board_mesh_package is None:
            return (0.0, 0.0)
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        found_mesh = False
        for mesh_asset in self.board_mesh_package.meshes:
            try:
                mesh = self._load_asset_mesh(mesh_asset.path)
            except Exception:
                continue
            bounds = mesh.bounds
            if bounds is None or len(bounds) != 2:
                continue
            found_mesh = True
            min_x = min(min_x, float(bounds[0][0]))
            min_y = min(min_y, float(bounds[0][1]))
            max_x = max(max_x, float(bounds[1][0]))
            max_y = max(max_y, float(bounds[1][1]))
        if not found_mesh:
            return (0.0, 0.0)
        target_min_x, target_min_y, _target_max_x, _target_max_y = self.scene.board_bounds
        return (target_min_x - min_x, target_min_y - min_y)

    def _build_board_mesh_actors(self) -> list:
        self.entity_parent_by_key.clear()
        self.entity_actor_groups.clear()
        self.entity_visibility.setdefault("pcb", True)
        if self.board_mesh_package is None:
            self.entity_visibility.setdefault("pcb::outline", True)
            self.entity_tree_payload = [("pcb", "PCB", [("pcb::outline", "Outline")])]
            return []
        actors: list = []
        pcb_children: list[tuple[str, str]] = []
        offset_x, offset_y = self._board_mesh_alignment_offset_xy()
        transform_override = self._board_mesh_transform_override()
        rotation_deg_xyz = (
            float(transform_override.get("x_deg", 0.0)),
            float(transform_override.get("y_deg", 0.0)),
            float(transform_override.get("z_deg", 0.0)),
        )
        flip_x = bool(transform_override.get("flip_x", False))
        flip_y = bool(transform_override.get("flip_y", False))
        flip_z = bool(transform_override.get("flip_z", False))
        transformed_meshes: list[tuple[str, list[list[float]], list[list[int]]]] = []
        max_z = -math.inf
        min_z = math.inf
        for mesh_asset in self.board_mesh_package.meshes:
            try:
                mesh = self._load_asset_mesh(mesh_asset.path)
            except Exception:
                continue
            transformed_vertices = [
                list(
                    self._transform_board_mesh_vertex(
                        (float(vertex[0]), float(vertex[1]), float(vertex[2])),
                        xy_offset=(offset_x, offset_y),
                        rotation_deg_xyz=rotation_deg_xyz,
                        flip_x=flip_x,
                        flip_y=flip_y,
                        flip_z=flip_z,
                    )
                )
                for vertex in mesh.vertices.tolist()
            ]
            if transformed_vertices:
                max_z = max(max_z, max(vertex[2] for vertex in transformed_vertices))
                min_z = min(min_z, min(vertex[2] for vertex in transformed_vertices))
            transformed_meshes.append((mesh_asset.name, transformed_vertices, mesh.faces.tolist()))
        if max_z == -math.inf or min_z == math.inf:
            z_offset = 0.0
        else:
            z_offset = -min_z
        for mesh_name, transformed_vertices, faces in transformed_meshes:
            shifted_vertices = [
                [vertex[0], vertex[1], vertex[2] + z_offset]
                for vertex in transformed_vertices
            ]
            actor = Mesh([shifted_vertices, faces]).c(self._board_mesh_color(mesh_name)).alpha(1.0)
            actor.linewidth(0)
            actors.append(actor)
            child_key = f"pcb::{mesh_name}"
            self._register_entity_group(child_key, [actor], "pcb")
            pcb_children.append((child_key, mesh_name))
        self.entity_visibility.setdefault("pcb::outline", True)
        pcb_children.append(("pcb::outline", "Outline"))
        self.entity_tree_payload = [("pcb", "PCB", pcb_children)]
        return actors

    def _build_component_actors(self) -> list:
        actors: list = []
        self.component_actor_by_reference.clear()
        self.reference_by_actor_id.clear()
        self.actor_base_color.clear()
        entity_payload = list(self.entity_tree_payload)
        for component in self.scene.components:
            parent_key = f"component::{component.reference}"
            self.entity_visibility.setdefault(parent_key, True)
            component_actors = self._build_component_asset_actors(component)
            component_children: list[tuple[str, str]] = []
            if not component_actors:
                component_actors = self._build_component_box_actors(component)
                component_children.append((f"{parent_key}::body", "Body"))
            else:
                asset = self._asset_override_for_component(component)
                if asset is None:
                    family = infer_component_family(component)
                    asset_options = self.asset_library.get(family, [])
                    asset = asset_options[0] if asset_options else None
                if asset is not None:
                    component_children.extend(
                        (f"{parent_key}::{part.part_name}", part.part_name)
                        for part in asset.parts
                    )
            actors.extend(component_actors)
            self.component_actor_by_reference[component.reference] = component_actors
            for actor in component_actors:
                self.reference_by_actor_id[id(actor)] = component.reference
            entity_payload.append(
                (parent_key, self._component_display_name(component), component_children or [(f"{parent_key}::body", "Body")])
            )
        self.entity_tree_payload = entity_payload
        return actors

    def _rotation_override_signature(self) -> str:
        board_overrides = load_asset_rotation_overrides().get(board_override_key(self.scene.board_path), {})
        return json.dumps(board_overrides, sort_keys=True)

    def _component_rotation_override(self, component: ComponentRecord) -> tuple[float, float, float]:
        board_overrides = load_asset_rotation_overrides().get(board_override_key(self.scene.board_path), {})
        values = board_overrides.get(component.reference, {})
        return (
            float(values.get("x_deg", 0.0)),
            float(values.get("y_deg", 0.0)),
            float(values.get("z_deg", 0.0)),
        )

    def _board_mesh_transform_override(self) -> dict[str, object]:
        return load_board_mesh_transform_overrides().get(
            board_override_key(self.scene.board_path),
            {
                "x_deg": 0.0,
                "y_deg": 0.0,
                "z_deg": 0.0,
                "flip_x": False,
                "flip_y": False,
                "flip_z": False,
            },
        )

    def _board_mesh_transform_signature(self) -> str:
        return json.dumps(self._board_mesh_transform_override(), sort_keys=True)

    def _component_asset_override_signature(self) -> str:
        board_overrides = load_component_reference_asset_overrides().get(board_override_key(self.scene.board_path), {})
        return json.dumps(board_overrides, sort_keys=True)

    def _component_forces_bounding_box(self, component: ComponentRecord) -> bool:
        board_overrides = load_component_reference_asset_overrides().get(board_override_key(self.scene.board_path), {})
        return board_overrides.get(component.reference, "").strip() == BOUNDING_BOX_ONLY_ASSET_OVERRIDE

    def _transform_board_mesh_vertex(
        self,
        vertex_xyz: tuple[float, float, float],
        *,
        xy_offset: tuple[float, float],
        rotation_deg_xyz: tuple[float, float, float],
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
    ) -> tuple[float, float, float]:
        x_coord = float(vertex_xyz[0]) + xy_offset[0]
        y_coord = float(vertex_xyz[1]) + xy_offset[1]
        z_coord = float(vertex_xyz[2])
        center_x = (self.scene.board_bounds[0] + self.scene.board_bounds[2]) / 2.0
        center_y = (self.scene.board_bounds[1] + self.scene.board_bounds[3]) / 2.0
        local_x = x_coord - center_x
        local_y = y_coord - center_y
        local_z = z_coord
        if flip_x:
            local_x *= -1.0
        if flip_y:
            local_y *= -1.0
        if flip_z:
            local_z *= -1.0
        rotated_x, rotated_y, rotated_z = rotate_point_xyz((local_x, local_y, local_z), rotation_deg_xyz)
        return (center_x + rotated_x, center_y + rotated_y, rotated_z)

    def _transform_board_mesh_direction(
        self,
        direction_xyz: tuple[float, float, float],
        *,
        rotation_deg_xyz: tuple[float, float, float],
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
    ) -> tuple[float, float, float]:
        dir_x, dir_y, dir_z = direction_xyz
        if flip_x:
            dir_x *= -1.0
        if flip_y:
            dir_y *= -1.0
        if flip_z:
            dir_z *= -1.0
        return rotate_point_xyz((dir_x, dir_y, dir_z), rotation_deg_xyz)

    def _refresh_scene_actors(self) -> None:
        for actor in self.actors:
            self.plotter.remove(actor)
        self.actors = self._build_board_mesh_actors() + self._build_outline_actors() + self._build_component_actors()
        for actor in self.actors:
            self.plotter += actor
        if self.entity_browser is not None:
            self.entity_browser.set_entities(self.entity_tree_payload, self.entity_visibility)
        self._apply_entity_visibility()
        self.set_selected_reference(self.selected_reference)
        if self.is_shown:
            self.plotter.render()

    def _build_component_box_actors(self, component: ComponentRecord) -> list:
        width = max(component.body_width_mm, 0.2)
        depth = max(component.body_height_mm, 0.2)
        height = max(component.body_thickness_mm, 0.2)
        center_x, center_y = component_body_center_in_world(component)
        actor = Box(
            pos=(center_x, center_y, -height / 2.0),
            length=width,
            width=depth,
            height=height,
        ).c(self._component_color(component)).alpha(0.9)
        actor.linewidth(0)
        if abs(component.rotation_deg) > 1e-6:
            actor.rotate_z(kicad_rotation_to_math(component.rotation_deg), around=(center_x, center_y, -height / 2.0))
        self.actor_base_color[id(actor)] = self._component_color(component)
        self._register_entity_group(f"component::{component.reference}::body", [actor], f"component::{component.reference}")
        return [actor]

    def _load_asset_mesh(self, stl_path: Path) -> trimesh.Trimesh:
        cache_key = str(stl_path)
        cached = self.asset_mesh_cache.get(cache_key)
        if cached is not None:
            return trimesh.Trimesh(vertices=cached.vertices.copy(), faces=cached.faces.copy(), process=False)
        mesh = trimesh.load(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected STL mesh at {stl_path}, got {type(mesh)!r}.")
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        if float(max(mesh.extents)) < 1.0:
            mesh.apply_scale(1000.0)
        self.asset_mesh_cache[cache_key] = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        return mesh

    def _asset_rotated_bbox_mm(
        self,
        asset: ComponentAssetDefinition,
        rotation_deg_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        rotated_vertices: list[tuple[float, float, float]] = []
        for part in asset.parts:
            if asset.fit_part_names and part.part_name not in asset.fit_part_names:
                continue
            try:
                mesh = self._load_asset_mesh(part.stl_path)
            except Exception:
                continue
            for x_coord, y_coord, z_coord in mesh.vertices.tolist():
                rotated_vertices.append(
                    rotate_point_xyz(
                        (
                            x_coord - asset.native_center_mm[0],
                            y_coord - asset.native_center_mm[1],
                            z_coord - asset.native_center_mm[2],
                        ),
                        rotation_deg_xyz,
                    )
                )
        if not rotated_vertices:
            return asset.native_bbox_mm
        x_values = [vertex[0] for vertex in rotated_vertices]
        y_values = [vertex[1] for vertex in rotated_vertices]
        z_values = [vertex[2] for vertex in rotated_vertices]
        return (
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
            max(z_values) - min(z_values),
        )

    def _asset_override_for_component(self, component: ComponentRecord) -> ComponentAssetDefinition | None:
        board_overrides = load_component_reference_asset_overrides().get(board_override_key(self.scene.board_path), {})
        definition_path_text = board_overrides.get(component.reference, "").strip()
        if definition_path_text == BOUNDING_BOX_ONLY_ASSET_OVERRIDE:
            return None
        if not definition_path_text:
            return None
        definition_path = Path(definition_path_text).expanduser().resolve()
        cached = self.asset_definition_cache.get(str(definition_path))
        if cached is not None:
            return cached
        for asset_list in self.asset_library.values():
            for asset in asset_list:
                if asset.definition_path == definition_path:
                    self.asset_definition_cache[str(definition_path)] = asset
                    return asset
        try:
            payload = json.loads(definition_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        component_family = str(payload.get("component_family", "generic")).strip().lower() or "generic"
        component_name = str(payload.get("component_name", definition_path.stem)).strip() or definition_path.stem
        native_bbox = payload.get("native_bbox_mm", {})
        native_center = payload.get("native_center_mm", {})
        native_rotation = payload.get("native_rotation_deg", {})
        fit_part_names_payload = payload.get("fit_part_names", [])
        parts_payload = payload.get("parts", [])
        parts: list[ComponentAssetPart] = []
        for item in parts_payload:
            if not isinstance(item, dict):
                continue
            stl_path_text = str(item.get("stl_path", "")).strip()
            if not stl_path_text:
                continue
            stl_path = (definition_path.parent / stl_path_text).resolve()
            if not stl_path.exists():
                continue
            parts.append(
                ComponentAssetPart(
                    part_name=str(item.get("part_name", stl_path.stem)),
                    stl_path=stl_path,
                    color=str(item.get("color", "#b8b8b8")),
                    height_mm=float(item.get("height_mm", 0.0)),
                )
            )
        if not parts:
            return None
        asset = ComponentAssetDefinition(
            component_family=component_family,
            component_name=component_name,
            definition_path=definition_path,
            native_bbox_mm=(
                float(native_bbox.get("x", 1.0)),
                float(native_bbox.get("y", 1.0)),
                float(native_bbox.get("z", 1.0)),
            ),
            native_center_mm=(
                float(native_center.get("x", 0.0)),
                float(native_center.get("y", 0.0)),
                float(native_center.get("z", 0.0)),
            ),
            native_rotation_deg=(
                float(native_rotation.get("x", 0.0)),
                float(native_rotation.get("y", 0.0)),
                float(native_rotation.get("z", 0.0)),
            ),
            fit_part_names=tuple(
                str(item).strip()
                for item in fit_part_names_payload
                if str(item).strip()
            ),
            parts=parts,
        )
        self.asset_definition_cache[str(definition_path)] = asset
        return asset

    def _build_component_asset_actors(self, component: ComponentRecord) -> list:
        if self._component_forces_bounding_box(component):
            return []
        asset = self._asset_override_for_component(component)
        if asset is None:
            family = infer_component_family(component)
            asset_options = self.asset_library.get(family, [])
            if not asset_options:
                return []
            asset = asset_options[0]
        center_x, center_y = component_body_center_in_world(component)
        target_x = max(component.body_width_mm, 0.2)
        target_y = max(component.body_height_mm, 0.2)
        target_z = max(component.body_thickness_mm, 0.2)
        override_rotation_deg = self._component_rotation_override(component)
        total_local_rotation = (
            asset.native_rotation_deg[0] + override_rotation_deg[0],
            asset.native_rotation_deg[1] + override_rotation_deg[1],
            asset.native_rotation_deg[2] + override_rotation_deg[2],
        )
        rotated_bbox_x, rotated_bbox_y, rotated_bbox_z = self._asset_rotated_bbox_mm(asset, total_local_rotation)
        scale_x = target_x / max(rotated_bbox_x, 1e-6)
        scale_y = target_y / max(rotated_bbox_y, 1e-6)
        scale_z = target_z / max(rotated_bbox_z, 1e-6)
        angle_rad = math.radians(kicad_rotation_to_math(component.rotation_deg))
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        actors: list = []
        for part in asset.parts:
            try:
                mesh = self._load_asset_mesh(part.stl_path)
            except Exception:
                return []
            transformed_vertices: list[list[float]] = []
            for x_coord, y_coord, z_coord in mesh.vertices.tolist():
                rotated_x, rotated_y, rotated_z = rotate_point_xyz(
                    (
                        x_coord - asset.native_center_mm[0],
                        y_coord - asset.native_center_mm[1],
                        z_coord - asset.native_center_mm[2],
                    ),
                    total_local_rotation,
                )
                centered_x = rotated_x * scale_x
                centered_y = rotated_y * scale_y
                centered_z = rotated_z * scale_z
                rotated_x = (centered_x * cosine) - (centered_y * sine)
                rotated_y = (centered_x * sine) + (centered_y * cosine)
                transformed_vertices.append([
                    center_x + rotated_x,
                    center_y + rotated_y,
                    (-target_z / 2.0) + centered_z,
                ])
            actor = Mesh([transformed_vertices, mesh.faces.tolist()]).c(part.color).alpha(1.0)
            actor.linewidth(0)
            self.actor_base_color[id(actor)] = part.color
            actors.append(actor)
            self._register_entity_group(
                f"component::{component.reference}::{part.part_name}",
                [actor],
                f"component::{component.reference}",
            )
        return actors

    def _on_left_click(self, event) -> None:
        actor = getattr(event, "actor", None)
        if actor is None:
            return
        reference = self.reference_by_actor_id.get(id(actor))
        if reference is None:
            return
        now = time.monotonic()
        self.set_selected_reference(reference)
        self._write_bridge_selection(reference)
        if reference == self._last_click_reference and (now - self._last_click_time) <= 0.7:
            self._write_bridge_rotation_request(reference)
            self._last_click_reference = None
            self._last_click_time = 0.0
            return
        self._last_click_reference = reference
        self._last_click_time = now

    def _on_left_double_click(self, event) -> None:
        actor = getattr(event, "actor", None)
        if actor is None:
            return
        reference = self.reference_by_actor_id.get(id(actor))
        if reference is None:
            return
        self.set_selected_reference(reference)
        self._write_bridge_selection(reference)
        self._write_bridge_rotation_request(reference)
        self._last_click_reference = None
        self._last_click_time = 0.0

    def _write_bridge_rotation_request(self, reference: str) -> None:
        if self.bridge_path is None:
            return
        payload = self._read_bridge_payload()
        payload["rotation_request_from_viewer"] = reference
        self.bridge_path.parent.mkdir(parents=True, exist_ok=True)
        self.bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_selected_reference(self, reference: str | None) -> None:
        self.selected_reference = reference
        for component in self.scene.components:
            actors = self.component_actor_by_reference.get(component.reference)
            if not actors:
                continue
            is_selected = component.reference == reference
            for actor in actors:
                actor.c("#f28f3b" if is_selected else self.actor_base_color.get(id(actor), self._component_color(component)))
                actor.alpha(1.0)
                actor.linewidth(0)
        selected_text = reference or "none"
        self.info.text(
            f"PCB: {self.scene.board_path.name}\n"
            f"Components: {len(self.scene.components)}\n"
            f"Selected: {selected_text}\n"
            "Showing board outline and component models when library matches are available"
        )
        if self.is_shown:
            self.plotter.render()

    def _read_bridge_payload(self) -> dict:
        if self.bridge_path is None or not self.bridge_path.exists():
            return {}
        try:
            payload = json.loads(self.bridge_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_bridge_selection(self, reference: str | None) -> None:
        if self.bridge_path is None:
            return
        payload = self._read_bridge_payload()
        payload["selected_reference_from_viewer"] = reference
        self.bridge_path.parent.mkdir(parents=True, exist_ok=True)
        self.bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_timer(self, _event) -> None:
        if self.entity_browser is not None:
            self.entity_browser.pump()
        payload = self._read_bridge_payload()
        selected_reference = payload.get("selected_reference_from_catalog")
        if selected_reference != self._last_bridge_selected:
            self._last_bridge_selected = selected_reference
            self.set_selected_reference(selected_reference if isinstance(selected_reference, str) else None)
        rotation_signature = self._rotation_override_signature()
        board_transform_signature = self._board_mesh_transform_signature()
        asset_override_signature = self._component_asset_override_signature()
        if (
            rotation_signature != self._last_rotation_override_signature
            or board_transform_signature != self._last_board_mesh_transform_signature
            or asset_override_signature != self._last_component_asset_override_signature
        ):
            self._last_rotation_override_signature = rotation_signature
            self._last_board_mesh_transform_signature = board_transform_signature
            self._last_component_asset_override_signature = asset_override_signature
            self._refresh_scene_actors()

    def show(self) -> None:
        self.actors = self._build_board_mesh_actors() + self._build_outline_actors() + self._build_component_actors()
        self.entity_browser = ViewerEntityBrowser(self.scene.board_path.stem, self._on_entity_visibility_changed)
        self.entity_browser.set_entities(self.entity_tree_payload, self.entity_visibility)
        self.plotter.show(*self.actors, self.info, zoom="tight", interactive=False)
        self.is_shown = True
        self._last_rotation_override_signature = self._rotation_override_signature()
        self._last_board_mesh_transform_signature = self._board_mesh_transform_signature()
        self._last_component_asset_override_signature = self._component_asset_override_signature()
        self.plotter.add_callback("LeftButtonPress", self._on_left_click)
        self.plotter.add_callback("LeftButtonDoubleClick", self._on_left_double_click)
        self.plotter.add_callback("Timer", self._on_timer)
        self.plotter.timer_callback("create", dt=150)
        self._apply_entity_visibility()
        self.set_selected_reference(self.selected_reference)
        self.plotter.interactive()

    def close(self) -> None:
        if self.entity_browser is not None:
            self.entity_browser.close()
            self.entity_browser = None
        if self.is_shown:
            self.plotter.close()
            self.is_shown = False


def preview_female_pin_array(holder_path: Path, contact_path: Path, title: str) -> None:
    holder_mesh = load_mesh_mm(holder_path)
    contact_mesh = load_mesh_mm(contact_path)
    plotter = Plotter(title=title, bg="#efe7d2", bg2="#f6f0e2", axes=1, size=(1200, 820))
    info = Text2D(
        f"{title}\nPreview: generated female pin array STL",
        pos="top-left",
        s=0.8,
        c="#2d241f",
        bg=None,
        font="Courier",
    )
    holder_actor = Mesh([holder_mesh.vertices.tolist(), holder_mesh.faces.tolist()]).c("#202020").alpha(0.72)
    contact_actor = Mesh([contact_mesh.vertices.tolist(), contact_mesh.faces.tolist()]).c("#d08a35").alpha(1.0)
    holder_actor.linewidth(0)
    contact_actor.linewidth(0)
    plotter.show(holder_actor, contact_actor, info, zoom="tight", interactive=True)


def tokenize_sexpr(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "(" or char == ")":
            tokens.append(char)
            index += 1
            continue
        if char == '"':
            index += 1
            buffer: list[str] = []
            while index < length:
                current = text[index]
                if current == "\\" and index + 1 < length:
                    buffer.append(text[index + 1])
                    index += 2
                    continue
                if current == '"':
                    index += 1
                    break
                buffer.append(current)
                index += 1
            tokens.append("".join(buffer))
            continue
        start = index
        while index < length and not text[index].isspace() and text[index] not in "()":
            index += 1
        tokens.append(text[start:index])
    return tokens


def parse_sexpr(tokens: list[str]) -> list:
    index = 0

    def parse_node() -> list | str:
        nonlocal index
        token = tokens[index]
        if token == "(":
            index += 1
            children: list = []
            while index < len(tokens) and tokens[index] != ")":
                children.append(parse_node())
            if index >= len(tokens):
                raise ValueError("Unbalanced S-expression: missing ')'.")
            index += 1
            return children
        index += 1
        return token

    nodes: list = []
    while index < len(tokens):
        nodes.append(parse_node())
    return nodes


def node_head(node: list) -> str | None:
    if isinstance(node, list) and node and isinstance(node[0], str):
        return node[0]
    return None


def child_nodes(node: list, head: str) -> list[list]:
    return [child for child in node if isinstance(child, list) and node_head(child) == head]


def first_child(node: list, head: str) -> list | None:
    for child in node:
        if isinstance(child, list) and node_head(child) == head:
            return child
    return None


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_xy_node(node: list | None, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if node is None or len(node) < 3:
        return default
    return (safe_float(node[1]), safe_float(node[2]))


def rotate_point(point: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return (
        (point[0] * cos_angle) - (point[1] * sin_angle),
        (point[0] * sin_angle) + (point[1] * cos_angle),
    )


def kicad_rotation_to_math(angle_deg: float) -> float:
    # KiCad board coordinates behave like screen coordinates, so footprint
    # rotation needs the opposite sign when we map local footprint geometry
    # into our math-style rotation helper.
    return -angle_deg


def rect_corners(center: tuple[float, float], width: float, height: float, angle_deg: float) -> list[tuple[float, float]]:
    half_w = width / 2.0
    half_h = height / 2.0
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    return [
        (center[0] + rotated[0], center[1] + rotated[1])
        for rotated in (rotate_point(corner, angle_deg) for corner in corners)
    ]


def translate_points(points: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(point[0] + dx, point[1] + dy) for point in points]


def bounds_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def merge_bounds(bounds: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def arc_extrema_points(start: tuple[float, float], mid: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
    x1, y1 = start
    x2, y2 = mid
    x3, y3 = end
    determinant = 2.0 * ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))
    if abs(determinant) < 1e-9:
        return [start, mid, end]
    ux = (
        ((x1 * x1 + y1 * y1) * (y2 - y3))
        + ((x2 * x2 + y2 * y2) * (y3 - y1))
        + ((x3 * x3 + y3 * y3) * (y1 - y2))
    ) / determinant
    uy = (
        ((x1 * x1 + y1 * y1) * (x3 - x2))
        + ((x2 * x2 + y2 * y2) * (x1 - x3))
        + ((x3 * x3 + y3 * y3) * (x2 - x1))
    ) / determinant
    radius = math.hypot(x1 - ux, y1 - uy)
    start_angle = math.atan2(y1 - uy, x1 - ux)
    mid_angle = math.atan2(y2 - uy, x2 - ux)
    end_angle = math.atan2(y3 - uy, x3 - ux)

    def normalize(angle: float) -> float:
        while angle < 0:
            angle += math.tau
        while angle >= math.tau:
            angle -= math.tau
        return angle

    start_angle = normalize(start_angle)
    mid_angle = normalize(mid_angle)
    end_angle = normalize(end_angle)

    def ccw_delta(a: float, b: float) -> float:
        delta = b - a
        if delta < 0:
            delta += math.tau
        return delta

    ccw_total = ccw_delta(start_angle, end_angle)
    ccw_mid = ccw_delta(start_angle, mid_angle)
    clockwise = ccw_mid > ccw_total

    def angle_on_arc(angle: float) -> bool:
        if clockwise:
            return ccw_delta(angle, start_angle) <= ccw_delta(end_angle, start_angle) + 1e-9
        return ccw_delta(start_angle, angle) <= ccw_total + 1e-9

    points = [start, mid, end]
    for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        candidate = normalize(angle)
        if angle_on_arc(candidate):
            points.append((ux + (radius * math.cos(candidate)), uy + (radius * math.sin(candidate))))
    return points


def primitive_bounds(primitive: list) -> tuple[float, float, float, float] | None:
    head = node_head(primitive)
    if head == "fp_line":
        return bounds_from_points(
            [
                parse_xy_node(first_child(primitive, "start")),
                parse_xy_node(first_child(primitive, "end")),
            ]
        )
    if head == "fp_rect":
        start = parse_xy_node(first_child(primitive, "start"))
        end = parse_xy_node(first_child(primitive, "end"))
        return bounds_from_points([start, end])
    if head == "fp_circle":
        center = parse_xy_node(first_child(primitive, "center"))
        end = parse_xy_node(first_child(primitive, "end"))
        radius = math.hypot(end[0] - center[0], end[1] - center[1])
        return (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    if head == "fp_arc":
        points = arc_extrema_points(
            parse_xy_node(first_child(primitive, "start")),
            parse_xy_node(first_child(primitive, "mid")),
            parse_xy_node(first_child(primitive, "end")),
        )
        return bounds_from_points(points)
    if head == "fp_poly":
        pts_node = first_child(primitive, "pts")
        if pts_node is None:
            return None
        points = [
            parse_xy_node(point_node)
            for point_node in pts_node
            if isinstance(point_node, list) and node_head(point_node) == "xy"
        ]
        return bounds_from_points(points)
    return None


def primitive_on_fab_layer(primitive: list) -> bool:
    layer_node = first_child(primitive, "layer")
    if layer_node is None or len(layer_node) < 2:
        return False
    return str(layer_node[1]) in {"F.Fab", "B.Fab"}


def pad_bounds(pad_node: list) -> tuple[float, float, float, float] | None:
    at_node = first_child(pad_node, "at")
    size_node = first_child(pad_node, "size")
    if at_node is None or size_node is None or len(size_node) < 3:
        return None
    center = parse_xy_node(at_node)
    angle_deg = safe_float(at_node[3]) if len(at_node) >= 4 else 0.0
    width = safe_float(size_node[1])
    height = safe_float(size_node[2])
    return bounds_from_points(rect_corners(center, width, height, angle_deg))


def footprint_at(footprint_node: list) -> tuple[float, float, float]:
    at_node = first_child(footprint_node, "at")
    if at_node is None or len(at_node) < 3:
        return (0.0, 0.0, 0.0)
    return (
        safe_float(at_node[1]),
        safe_float(at_node[2]),
        safe_float(at_node[3]) if len(at_node) >= 4 else 0.0,
    )


def footprint_property_map(footprint_node: list) -> dict[str, str]:
    values: dict[str, str] = {}
    for property_node in child_nodes(footprint_node, "property"):
        if len(property_node) >= 3 and isinstance(property_node[1], str):
            values[str(property_node[1])] = str(property_node[2])
    return values


def footprint_models(footprint_node: list) -> list[str]:
    models: list[str] = []
    for model_node in child_nodes(footprint_node, "model"):
        if len(model_node) >= 2:
            models.append(str(model_node[1]))
    return models


def estimate_height_from_footprint(footprint_name: str, value: str) -> tuple[float, str]:
    footprint_upper = footprint_name.upper()
    value_upper = value.upper()

    rules = [
        (("LED_0805",), 0.80, "heuristic: LED 0805"),
        (("R_0805", "C_0805"), 0.80, "heuristic: 0805 passive"),
        (("LED_0603",), 0.60, "heuristic: LED 0603"),
        (("R_0603", "C_0603"), 0.60, "heuristic: 0603 passive"),
        (("LED_1206",), 1.10, "heuristic: LED 1206"),
        (("R_1206", "C_1206"), 0.90, "heuristic: 1206 passive"),
        (("HC49",), 4.50, "heuristic: HC49 crystal"),
        (("SOIC-8", "SOIJ-8", "SO-8"), 1.75, "heuristic: SO-8 package"),
        (("POWERINTEGRATIONS_SO-8",), 1.75, "heuristic: SO-8 package"),
        (("PLCC-28",), 4.60, "heuristic: PLCC socket"),
        (("PINHEADER", "PINSOCKET"), 8.50, "heuristic: pin header/socket"),
    ]
    for patterns, height_mm, source in rules:
        if any(pattern in footprint_upper for pattern in patterns):
            return height_mm, source

    if "CONNECTOR" in footprint_upper:
        return 8.50, "heuristic: generic connector"
    if "CRYSTAL" in footprint_upper:
        return 4.50, "heuristic: generic crystal"
    if "LOGO" in footprint_upper:
        return 0.20, "heuristic: graphic/logo"
    if reference_prefix(value_upper) == "LED":
        return 0.80, "heuristic: LED"
    return 1.60, "heuristic: fallback"


def reference_prefix(value: str) -> str:
    letters = []
    for char in value:
        if char.isalpha():
            letters.append(char)
        elif letters:
            break
    return "".join(letters)


def board_override_key(board_path: Path) -> str:
    return str(board_path.expanduser().resolve())


def load_height_overrides() -> dict[str, dict[str, float]]:
    if not HEIGHT_OVERRIDE_PATH.exists():
        return {}
    try:
        payload = json.loads(HEIGHT_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, float]] = {}
    for board_key, entries in payload.items():
        if not isinstance(entries, dict):
            continue
        cleaned[board_key] = {}
        for reference, height in entries.items():
            parsed = safe_float_or_none(height)
            if parsed is not None:
                cleaned[board_key][str(reference)] = parsed
    return cleaned


def save_height_overrides(overrides: dict[str, dict[str, float]]) -> None:
    HEIGHT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEIGHT_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def load_placement_overrides() -> dict[str, dict[str, dict[str, float]]]:
    if not PLACEMENT_OVERRIDE_PATH.exists():
        return {}
    try:
        payload = json.loads(PLACEMENT_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, dict[str, float]]] = {}
    for board_key, entries in payload.items():
        if not isinstance(entries, dict):
            continue
        cleaned[board_key] = {}
        for reference, placement in entries.items():
            if not isinstance(placement, dict):
                continue
            cleaned[board_key][str(reference)] = {
                "x_mm": safe_float(placement.get("x_mm"), 0.0),
                "y_mm": safe_float(placement.get("y_mm"), 0.0),
            }
    return cleaned


def save_placement_overrides(overrides: dict[str, dict[str, dict[str, float]]]) -> None:
    PLACEMENT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLACEMENT_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def load_asset_rotation_overrides() -> dict[str, dict[str, dict[str, float]]]:
    if not ASSET_ROTATION_OVERRIDE_PATH.exists():
        return {}
    try:
        payload = json.loads(ASSET_ROTATION_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for board_key, board_overrides in payload.items():
        if not isinstance(board_overrides, dict):
            continue
        normalized_board: dict[str, dict[str, float]] = {}
        for reference, rotation_values in board_overrides.items():
            if not isinstance(rotation_values, dict):
                continue
            normalized_board[str(reference)] = {
                "x_deg": float(rotation_values.get("x_deg", 0.0)),
                "y_deg": float(rotation_values.get("y_deg", 0.0)),
                "z_deg": float(rotation_values.get("z_deg", 0.0)),
            }
        result[str(board_key)] = normalized_board
    return result


def save_asset_rotation_overrides(overrides: dict[str, dict[str, dict[str, float]]]) -> None:
    ASSET_ROTATION_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSET_ROTATION_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def load_board_mesh_transform_overrides() -> dict[str, dict[str, object]]:
    if not BOARD_MESH_OVERRIDE_PATH.exists():
        return {}
    try:
        payload = json.loads(BOARD_MESH_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, object]] = {}
    for board_key, values in payload.items():
        if not isinstance(values, dict):
            continue
        cleaned[str(board_key)] = {
            "x_deg": float(values.get("x_deg", 0.0)),
            "y_deg": float(values.get("y_deg", 0.0)),
            "z_deg": float(values.get("z_deg", 0.0)),
            "flip_x": bool(values.get("flip_x", False)),
            "flip_y": bool(values.get("flip_y", False)),
            "flip_z": bool(values.get("flip_z", False)),
        }
    return cleaned


def save_board_mesh_transform_overrides(overrides: dict[str, dict[str, object]]) -> None:
    BOARD_MESH_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOARD_MESH_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def load_component_reference_asset_overrides() -> dict[str, dict[str, str]]:
    if not COMPONENT_REFERENCE_ASSET_OVERRIDE_PATH.exists():
        return {}
    try:
        payload = json.loads(COMPONENT_REFERENCE_ASSET_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for board_key, values in payload.items():
        if not isinstance(values, dict):
            continue
        cleaned[str(board_key)] = {
            str(reference): str(definition_path).strip()
            for reference, definition_path in values.items()
            if str(definition_path).strip()
        }
    return cleaned


def save_component_reference_asset_overrides(overrides: dict[str, dict[str, str]]) -> None:
    COMPONENT_REFERENCE_ASSET_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPONENT_REFERENCE_ASSET_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def extract_outline_segment(node: list) -> OutlineSegment | None:
    layer_node = first_child(node, "layer")
    if layer_node is None or len(layer_node) < 2 or str(layer_node[1]) != "Edge.Cuts":
        return None
    head = node_head(node)
    if head == "gr_line":
        return OutlineSegment(
            kind="line",
            start=parse_xy_node(first_child(node, "start")),
            end=parse_xy_node(first_child(node, "end")),
        )
    if head == "gr_arc":
        return OutlineSegment(
            kind="arc",
            start=parse_xy_node(first_child(node, "start")),
            mid=parse_xy_node(first_child(node, "mid")),
            end=parse_xy_node(first_child(node, "end")),
        )
    return None


def arc_outline_points(start: tuple[float, float], mid: tuple[float, float], end: tuple[float, float], segments: int = 24) -> list[tuple[float, float]]:
    x1, y1 = start
    x2, y2 = mid
    x3, y3 = end
    determinant = 2.0 * ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))
    if abs(determinant) < 1e-9:
        return [start, end]
    ux = (
        ((x1 * x1 + y1 * y1) * (y2 - y3))
        + ((x2 * x2 + y2 * y2) * (y3 - y1))
        + ((x3 * x3 + y3 * y3) * (y1 - y2))
    ) / determinant
    uy = (
        ((x1 * x1 + y1 * y1) * (x3 - x2))
        + ((x2 * x2 + y2 * y2) * (x1 - x3))
        + ((x3 * x3 + y3 * y3) * (x2 - x1))
    ) / determinant
    radius = math.hypot(x1 - ux, y1 - uy)
    start_angle = math.atan2(y1 - uy, x1 - ux)
    mid_angle = math.atan2(y2 - uy, x2 - ux)
    end_angle = math.atan2(y3 - uy, x3 - ux)

    def normalize(angle: float) -> float:
        while angle < 0:
            angle += math.tau
        while angle >= math.tau:
            angle -= math.tau
        return angle

    start_angle = normalize(start_angle)
    mid_angle = normalize(mid_angle)
    end_angle = normalize(end_angle)

    def ccw_delta(a: float, b: float) -> float:
        delta = b - a
        if delta < 0:
            delta += math.tau
        return delta

    ccw_total = ccw_delta(start_angle, end_angle)
    ccw_mid = ccw_delta(start_angle, mid_angle)
    clockwise = ccw_mid > ccw_total
    if clockwise:
        total = ccw_delta(end_angle, start_angle)
        return [
            (ux + (radius * math.cos(start_angle - (total * index / segments))), uy + (radius * math.sin(start_angle - (total * index / segments))))
            for index in range(segments + 1)
        ]
    return [
        (ux + (radius * math.cos(start_angle + (ccw_total * index / segments))), uy + (radius * math.sin(start_angle + (ccw_total * index / segments))))
        for index in range(segments + 1)
    ]


def outline_segment_points(segment: OutlineSegment) -> list[tuple[float, float]]:
    if segment.kind == "line":
        return [segment.start, segment.end]
    if segment.kind == "arc" and segment.mid is not None:
        return arc_outline_points(segment.start, segment.mid, segment.end)
    return [segment.start, segment.end]


def component_local_corners(component: ComponentRecord) -> list[tuple[float, float]]:
    return rect_corners(
        (component.body_center_offset_x_mm, component.body_center_offset_y_mm),
        component.body_width_mm,
        component.body_height_mm,
        0.0,
    )


def component_body_center_on_board(component: ComponentRecord) -> tuple[float, float]:
    rotated_offset = rotate_point(
        (component.body_center_offset_x_mm, component.body_center_offset_y_mm),
        component.rotation_deg,
    )
    return (
        component.x_mm + component.placement_offset_x_mm + rotated_offset[0],
        component.y_mm + component.placement_offset_y_mm + rotated_offset[1],
    )


def component_body_center_in_world(component: ComponentRecord) -> tuple[float, float]:
    rotated_offset = rotate_point(
        (component.body_center_offset_x_mm, component.body_center_offset_y_mm),
        kicad_rotation_to_math(component.rotation_deg),
    )
    return (
        component.x_mm + component.placement_offset_x_mm + rotated_offset[0],
        component.y_mm + component.placement_offset_y_mm + rotated_offset[1],
    )


def component_board_corners(component: ComponentRecord) -> list[tuple[float, float]]:
    center_x, center_y = component_body_center_on_board(component)
    return rect_corners(
        (center_x, center_y),
        component.body_width_mm,
        component.body_height_mm,
        component.rotation_deg,
    )


def component_world_corners(component: ComponentRecord) -> list[tuple[float, float]]:
    center_x, center_y = component_body_center_in_world(component)
    return rect_corners(
        (center_x, center_y),
        component.body_width_mm,
        component.body_height_mm,
        kicad_rotation_to_math(component.rotation_deg),
    )


def extract_component_record(
    footprint_node: list,
    height_override_mm: float | None = None,
    placement_override_x_mm: float = 0.0,
    placement_override_y_mm: float = 0.0,
) -> ComponentRecord | None:
    if len(footprint_node) < 2:
        return None
    properties = footprint_property_map(footprint_node)
    reference = properties.get("Reference", "").strip()
    if not reference:
        return None
    value = properties.get("Value", "").strip()
    footprint_name = str(footprint_node[1])
    layer_node = first_child(footprint_node, "layer")
    layer = str(layer_node[1]) if layer_node is not None and len(layer_node) >= 2 else "Unknown"
    x_mm, y_mm, rotation_deg = footprint_at(footprint_node)

    fab_bounds_list = [
        bounds
        for bounds in (
            primitive_bounds(child)
            for child in footprint_node
            if isinstance(child, list) and node_head(child) in {"fp_line", "fp_rect", "fp_circle", "fp_arc", "fp_poly"} and primitive_on_fab_layer(child)
        )
        if bounds is not None
    ]
    fab_bounds = merge_bounds(fab_bounds_list)

    pad_bounds_list = [
        bounds
        for bounds in (pad_bounds(child) for child in child_nodes(footprint_node, "pad"))
        if bounds is not None
    ]
    pads_merged = merge_bounds(pad_bounds_list)

    chosen_bounds = fab_bounds or pads_merged
    if chosen_bounds is None:
        chosen_bounds = (0.0, 0.0, 0.0, 0.0)
        bounds_source = "none"
    else:
        bounds_source = "fab" if fab_bounds is not None else "pads"

    body_width_mm = max(chosen_bounds[2] - chosen_bounds[0], 0.0)
    body_height_mm = max(chosen_bounds[3] - chosen_bounds[1], 0.0)
    body_center_offset_x_mm = (chosen_bounds[0] + chosen_bounds[2]) / 2.0
    body_center_offset_y_mm = (chosen_bounds[1] + chosen_bounds[3]) / 2.0
    if pads_merged is None:
        pad_span_width_mm = 0.0
        pad_span_height_mm = 0.0
    else:
        pad_span_width_mm = max(pads_merged[2] - pads_merged[0], 0.0)
        pad_span_height_mm = max(pads_merged[3] - pads_merged[1], 0.0)

    if height_override_mm is not None:
        body_thickness_mm = height_override_mm
        height_source = "manual override"
    else:
        body_thickness_mm, height_source = estimate_height_from_footprint(footprint_name, value)

    return ComponentRecord(
        reference=reference,
        value=value,
        footprint=footprint_name,
        layer=layer,
        x_mm=x_mm,
        y_mm=y_mm,
        rotation_deg=rotation_deg,
        body_width_mm=body_width_mm,
        body_height_mm=body_height_mm,
        body_thickness_mm=body_thickness_mm,
        body_center_offset_x_mm=body_center_offset_x_mm,
        body_center_offset_y_mm=body_center_offset_y_mm,
        placement_offset_x_mm=placement_override_x_mm,
        placement_offset_y_mm=placement_override_y_mm,
        pad_span_width_mm=pad_span_width_mm,
        pad_span_height_mm=pad_span_height_mm,
        pad_count=len(child_nodes(footprint_node, "pad")),
        model_path="; ".join(footprint_models(footprint_node)),
        bounds_source=bounds_source,
        height_source=height_source,
    )


def load_board_scene(board_path: Path) -> BoardScene:
    text = board_path.read_text(encoding="utf-8")
    parsed = parse_sexpr(tokenize_sexpr(text))
    root = parsed[0] if len(parsed) == 1 and isinstance(parsed[0], list) else parsed
    footprints = [
        child
        for child in root
        if isinstance(child, list) and node_head(child) in {"footprint", "module"}
    ]
    board_key = board_override_key(board_path)
    overrides = load_height_overrides().get(board_key, {})
    placement_overrides = load_placement_overrides().get(board_key, {})
    components = [
        record
        for record in (
            extract_component_record(
                node,
                height_override_mm=overrides.get(
                    footprint_property_map(node).get("Reference", "").strip()
                ),
                placement_override_x_mm=placement_overrides.get(
                    footprint_property_map(node).get("Reference", "").strip(),
                    {},
                ).get("x_mm", 0.0),
                placement_override_y_mm=placement_overrides.get(
                    footprint_property_map(node).get("Reference", "").strip(),
                    {},
                ).get("y_mm", 0.0),
            )
            for node in footprints
        )
        if record is not None
    ]
    outline_segments = [
        segment
        for segment in (
            extract_outline_segment(child)
            for child in root
            if isinstance(child, list) and node_head(child) in {"gr_line", "gr_arc"}
        )
        if segment is not None
    ]
    outline_points = [point for segment in outline_segments for point in outline_segment_points(segment)]
    board_bounds = bounds_from_points(outline_points) or (0.0, 0.0, 1.0, 1.0)
    return BoardScene(
        board_path=board_path,
        components=sorted(components, key=lambda item: item.reference),
        outline_segments=outline_segments,
        board_bounds=board_bounds,
    )


def load_components_from_board(board_path: Path) -> list[ComponentRecord]:
    return load_board_scene(board_path).components


class ComponentCatalogApp:
    def __init__(self, board_path: Path) -> None:
        self.root = tk.Tk()
        self.root.title("PCB Component Catalog")
        self.root.geometry("1280x760")
        self.root.minsize(1080, 640)

        self.board_path = board_path.expanduser().resolve()
        self.all_components: list[ComponentRecord] = []
        self.filtered_components: list[ComponentRecord] = []
        self.outline_segments: list[OutlineSegment] = []
        self.board_bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self.component_item_ids: dict[str, int] = {}
        self.canvas_item_to_reference: dict[int, str] = {}
        self.current_scene: BoardScene | None = None
        self.viewer_bridge_path: Path | None = None
        self.viewer_process: subprocess.Popen | None = None
        self._last_viewer_selected_reference: str | None = None
        self._last_viewer_rotation_request: str | None = None
        self.board_mesh_package: BoardMeshPackage | None = auto_detect_board_mesh_package(self.board_path)

        self.board_var = tk.StringVar(value=str(self.board_path))
        self.board_mesh_manifest_var = tk.StringVar(
            value=str(self.board_mesh_package.manifest_path) if self.board_mesh_package is not None else ""
        )
        self.status_var = tk.StringVar(value="Loading components...")
        self.search_var = tk.StringVar()
        self.height_override_var = tk.StringVar()
        self.offset_x_var = tk.StringVar()
        self.offset_y_var = tk.StringVar()
        self.rotation_x_var = tk.StringVar()
        self.rotation_y_var = tk.StringVar()
        self.rotation_z_var = tk.StringVar()
        self.board_rotation_x_var = tk.StringVar()
        self.board_rotation_y_var = tk.StringVar()
        self.board_rotation_z_var = tk.StringVar()
        self.board_flip_x_var = tk.BooleanVar(value=False)
        self.board_flip_y_var = tk.BooleanVar(value=False)
        self.board_flip_z_var = tk.BooleanVar(value=False)
        self.pending_female_pin_definition_path: Path | None = None
        self.pending_female_pin_reference: str | None = None

        self._build_ui()
        self._load_board(self.board_path)
        self.root.after(150, self._poll_viewer_bridge)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(2, weight=1)

        header = ttk.Label(outer, text="PCB Component Catalog", font=("Georgia", 18, "bold"))
        header.grid(row=0, column=0, columnspan=2, sticky="w")

        source_box = ttk.LabelFrame(outer, text="Board Source", padding=10)
        source_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 12))
        source_box.columnconfigure(0, weight=1)
        ttk.Entry(source_box, textvariable=self.board_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(source_box, text="Browse Board", command=self._browse_board).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(source_box, text="Reload", command=self._reload_current_board).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(source_box, text="Export JSON", command=self._export_json).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(source_box, text="View (3D)", command=self._open_3d_viewer).grid(row=0, column=4, padx=(8, 0))
        ttk.Label(source_box, text="Board STL Package").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_box, textvariable=self.board_mesh_manifest_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(source_box, text="Browse Package", command=self._browse_board_mesh_package).grid(row=1, column=4, padx=(8, 0), pady=(8, 0))
        board_transform_box = ttk.LabelFrame(source_box, text="Board STL Transform", padding=8)
        board_transform_box.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(10, 0))
        board_transform_box.columnconfigure(1, weight=1)
        ttk.Label(board_transform_box, text="X / Y / Z (deg)").grid(row=0, column=0, sticky="w")
        board_rotation_row = ttk.Frame(board_transform_box)
        board_rotation_row.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        board_rotation_row.columnconfigure(0, weight=1)
        board_rotation_row.columnconfigure(1, weight=1)
        board_rotation_row.columnconfigure(2, weight=1)
        ttk.Entry(board_rotation_row, textvariable=self.board_rotation_x_var, width=8).grid(row=0, column=0, sticky="ew")
        ttk.Entry(board_rotation_row, textvariable=self.board_rotation_y_var, width=8).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Entry(board_rotation_row, textvariable=self.board_rotation_z_var, width=8).grid(row=0, column=2, sticky="ew")
        board_button_row = ttk.Frame(board_transform_box)
        board_button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(board_button_row, text="X -90", command=lambda: self._nudge_board_mesh_rotation("x_deg", -90.0)).pack(side="left", fill="x", expand=True)
        ttk.Button(board_button_row, text="X +90", command=lambda: self._nudge_board_mesh_rotation("x_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(board_button_row, text="Y -90", command=lambda: self._nudge_board_mesh_rotation("y_deg", -90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(board_button_row, text="Y +90", command=lambda: self._nudge_board_mesh_rotation("y_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(board_button_row, text="Z -90", command=lambda: self._nudge_board_mesh_rotation("z_deg", -90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(board_button_row, text="Z +90", command=lambda: self._nudge_board_mesh_rotation("z_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        board_flip_row = ttk.Frame(board_transform_box)
        board_flip_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(board_flip_row, text="Flip X", variable=self.board_flip_x_var).pack(side="left")
        ttk.Checkbutton(board_flip_row, text="Flip Y", variable=self.board_flip_y_var).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(board_flip_row, text="Flip Z", variable=self.board_flip_z_var).pack(side="left", padx=(12, 0))
        ttk.Button(board_transform_box, text="Save Board Transform", command=self._save_board_mesh_transform_override).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(board_transform_box, text="Clear Board Transform", command=self._clear_board_mesh_transform_override).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(source_box, textvariable=self.status_var, wraplength=1100).grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))

        filter_box = ttk.LabelFrame(outer, text="Components", padding=10)
        filter_box.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        filter_box.columnconfigure(0, weight=1)
        filter_box.rowconfigure(1, weight=1)
        filter_box.rowconfigure(2, weight=2)
        ttk.Label(filter_box, text="Search by reference, value, or footprint").grid(row=0, column=0, sticky="w")
        search_entry = ttk.Entry(filter_box, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        search_entry.bind("<KeyRelease>", lambda _event: self._apply_filter())

        columns = (
            "reference",
            "value",
            "footprint",
            "layer",
            "body",
            "height_source",
            "pads",
            "position",
        )
        self.tree = ttk.Treeview(filter_box, columns=columns, show="headings", height=22)
        headings = {
            "reference": "Ref",
            "value": "Value",
            "footprint": "Footprint",
            "layer": "Layer",
            "body": "Body L x W x H (mm)",
            "height_source": "Height Source",
            "pads": "Pads",
            "position": "X, Y (mm)",
        }
        widths = {
            "reference": 90,
            "value": 120,
            "footprint": 320,
            "layer": 80,
            "body": 150,
            "height_source": 130,
            "pads": 60,
            "position": 140,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.tree.bind("<<TreeviewSelect>>", self._show_selected_component)

        scrollbar = ttk.Scrollbar(filter_box, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=2, sticky="ns", pady=(10, 0))
        self.tree.configure(yscrollcommand=scrollbar.set)

        map_box = ttk.LabelFrame(filter_box, text="Board Map", padding=8)
        map_box.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        map_box.columnconfigure(0, weight=1)
        map_box.rowconfigure(0, weight=1)
        self.board_canvas = tk.Canvas(map_box, bg="#f5efdf", highlightthickness=0, height=320)
        self.board_canvas.grid(row=0, column=0, sticky="nsew")
        self.board_canvas.bind("<Configure>", lambda _event: self._redraw_board_map())
        self.board_canvas.bind("<Button-1>", self._on_board_canvas_click)

        detail_container = ttk.LabelFrame(outer, text="Component Details", padding=0)
        detail_container.grid(row=2, column=1, sticky="nsew")
        detail_container.columnconfigure(0, weight=1)
        detail_container.rowconfigure(0, weight=1)
        self.detail_canvas = tk.Canvas(detail_container, highlightthickness=0)
        detail_scrollbar = ttk.Scrollbar(detail_container, orient="vertical", command=self.detail_canvas.yview)
        detail_scrollbar.grid(row=0, column=1, sticky="ns")
        self.detail_canvas.grid(row=0, column=0, sticky="nsew")
        self.detail_canvas.configure(yscrollcommand=detail_scrollbar.set)
        detail_box = ttk.Frame(self.detail_canvas, padding=12)
        self.detail_canvas_window = self.detail_canvas.create_window((0, 0), window=detail_box, anchor="nw")
        detail_box.columnconfigure(1, weight=1)
        detail_box.bind("<Configure>", self._on_detail_content_configure)
        self.detail_canvas.bind("<Configure>", self._on_detail_canvas_configure)
        self.detail_canvas.bind("<MouseWheel>", self._on_detail_mousewheel)
        self.detail_canvas.bind("<Button-4>", self._on_detail_mousewheel)
        self.detail_canvas.bind("<Button-5>", self._on_detail_mousewheel)
        detail_box.bind("<MouseWheel>", self._on_detail_mousewheel)
        detail_box.bind("<Button-4>", self._on_detail_mousewheel)
        detail_box.bind("<Button-5>", self._on_detail_mousewheel)

        self.detail_vars = {
            "reference": tk.StringVar(value="-"),
            "value": tk.StringVar(value="-"),
            "footprint": tk.StringVar(value="-"),
            "layer": tk.StringVar(value="-"),
            "position": tk.StringVar(value="-"),
            "rotation": tk.StringVar(value="-"),
            "body": tk.StringVar(value="-"),
            "height": tk.StringVar(value="-"),
            "pad_span": tk.StringVar(value="-"),
            "pad_count": tk.StringVar(value="-"),
            "bounds_source": tk.StringVar(value="-"),
            "height_source": tk.StringVar(value="-"),
            "model_path": tk.StringVar(value="-"),
        }

        row = 0
        for label, key in (
            ("Reference", "reference"),
            ("Value", "value"),
            ("Footprint", "footprint"),
            ("Layer", "layer"),
            ("Board Position", "position"),
            ("Rotation", "rotation"),
            ("Body Size", "body"),
            ("Body Height", "height"),
            ("Pad Span", "pad_span"),
            ("Pad Count", "pad_count"),
            ("Dimension Source", "bounds_source"),
            ("Height Source", "height_source"),
            ("KiCad 3D Model", "model_path"),
        ):
            ttk.Label(detail_box, text=label, font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="nw", pady=(0, 8))
            ttk.Label(detail_box, textvariable=self.detail_vars[key], wraplength=360, justify="left").grid(row=row, column=1, sticky="nw", pady=(0, 8))
            row += 1

        override_box = ttk.LabelFrame(detail_box, text="Manual Height Override", padding=8)
        override_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        override_box.columnconfigure(1, weight=1)
        ttk.Label(override_box, text="Height (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(override_box, textvariable=self.height_override_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(override_box, text="Save Height", command=self._save_height_override).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(override_box, text="Clear Override", command=self._clear_height_override).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        row += 1

        placement_box = ttk.LabelFrame(detail_box, text="Manual Placement Override", padding=8)
        placement_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        placement_box.columnconfigure(1, weight=1)
        ttk.Label(placement_box, text="X Offset (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(placement_box, textvariable=self.offset_x_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(placement_box, text="Y Offset (mm)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(placement_box, textvariable=self.offset_y_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(placement_box, text="Save Placement", command=self._save_placement_override).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(placement_box, text="Clear Placement", command=self._clear_placement_override).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        row += 1

        rotation_box = ttk.LabelFrame(detail_box, text="3D Model Rotation Override", padding=8)
        rotation_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        rotation_box.columnconfigure(1, weight=1)
        ttk.Label(rotation_box, text="X / Y / Z (deg)").grid(row=0, column=0, sticky="w")
        rotation_entry_row = ttk.Frame(rotation_box)
        rotation_entry_row.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        rotation_entry_row.columnconfigure(0, weight=1)
        rotation_entry_row.columnconfigure(1, weight=1)
        rotation_entry_row.columnconfigure(2, weight=1)
        ttk.Entry(rotation_entry_row, textvariable=self.rotation_x_var, width=8).grid(row=0, column=0, sticky="ew")
        ttk.Entry(rotation_entry_row, textvariable=self.rotation_y_var, width=8).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Entry(rotation_entry_row, textvariable=self.rotation_z_var, width=8).grid(row=0, column=2, sticky="ew")
        rotate_button_row = ttk.Frame(rotation_box)
        rotate_button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(rotate_button_row, text="X -90", command=lambda: self._nudge_asset_rotation("x_deg", -90.0)).pack(side="left", fill="x", expand=True)
        ttk.Button(rotate_button_row, text="X +90", command=lambda: self._nudge_asset_rotation("x_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(rotate_button_row, text="Y -90", command=lambda: self._nudge_asset_rotation("y_deg", -90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(rotate_button_row, text="Y +90", command=lambda: self._nudge_asset_rotation("y_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(rotate_button_row, text="Z -90", command=lambda: self._nudge_asset_rotation("z_deg", -90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(rotate_button_row, text="Z +90", command=lambda: self._nudge_asset_rotation("z_deg", 90.0)).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(rotation_box, text="Save Rotation", command=self._save_asset_rotation_override).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(rotation_box, text="Clear Rotation", command=self._clear_asset_rotation_override).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        row += 1

        attach_stl_box = ttk.LabelFrame(detail_box, text="Attach STL", padding=8)
        attach_stl_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        attach_stl_box.columnconfigure(0, weight=1)
        ttk.Label(
            attach_stl_box,
            text="Load one or more STL files that are already positioned relative to each other, then fit their combined bbox to the selected component.",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        attach_button_row = ttk.Frame(attach_stl_box)
        attach_button_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        attach_button_row.columnconfigure(0, weight=1)
        ttk.Button(attach_button_row, text="Attach STL", command=self._attach_stl_to_component).grid(row=0, column=0, sticky="ew")
        row += 1

        female_pin_box = ttk.LabelFrame(detail_box, text="Female Pin Generator", padding=8)
        female_pin_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        female_pin_box.columnconfigure(0, weight=1)
        ttk.Label(
            female_pin_box,
            text="Use the mate-builder project JSON to create a female pin socket array like 2x5 for the selected component.",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(female_pin_box, text="Female Pin", command=self._generate_female_pin_array).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.place_female_pin_button = ttk.Button(female_pin_box, text="Place", command=self._place_pending_female_pin, state="disabled")
        self.place_female_pin_button.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        row += 1

        note = (
            "Dimensions come from the footprint body outline on F.Fab/B.Fab when present. "
            "If the footprint has no Fab outline, the script falls back to pad span bounds. "
            "Height starts from footprint heuristics and can be manually overridden per component. "
            "Placement overrides help for connector housings that do not line up perfectly with the footprint Fab outline. "
            "3D model rotation overrides adjust library STL orientation for the selected component in the viewer."
        )
        ttk.Label(detail_box, text=note, wraplength=380, justify="left").grid(row=row, column=0, columnspan=2, sticky="sw", pady=(10, 0))

    def _load_board(self, board_path: Path) -> None:
        try:
            scene = load_board_scene(board_path)
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Could not parse board:\n{board_path}\n\n{exc}", parent=self.root)
            return
        self.board_path = board_path
        self.board_var.set(str(board_path))
        self.current_scene = scene
        self.all_components = scene.components
        self.outline_segments = scene.outline_segments
        self.board_bounds = scene.board_bounds
        auto_package = auto_detect_board_mesh_package(board_path)
        self.board_mesh_package = auto_package
        self.board_mesh_manifest_var.set(str(auto_package.manifest_path) if auto_package is not None else "")
        self._load_board_mesh_transform_into_vars()
        self._set_pending_female_pin_definition(None, None)
        self.status_var.set(f"Loaded {len(scene.components)} components from {board_path.name}")
        self._close_3d_viewer()
        self._apply_filter()

    def _reload_current_board(self) -> None:
        self._load_board(Path(self.board_var.get().strip()).expanduser().resolve())

    def _browse_board(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a KiCad PCB board",
            initialdir=str(self.board_path.parent),
            filetypes=[("KiCad PCB files", "*.kicad_pcb"), ("All files", "*.*")],
        )
        if selected:
            self._load_board(Path(selected).expanduser().resolve())

    def _browse_board_mesh_package(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a board STL manifest or PCB package manifest",
            initialdir=str((self.board_mesh_package.manifest_path.parent if self.board_mesh_package is not None else REPO_ROOT / "output")),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        package = load_board_mesh_package(Path(selected).expanduser().resolve())
        if package is None:
            messagebox.showerror("Invalid Package", "Could not load the selected board STL package manifest.", parent=self.root)
            return
        self.board_mesh_package = package
        self.board_mesh_manifest_var.set(str(package.manifest_path))
        self._load_board_mesh_transform_into_vars()
        self.status_var.set(f"Using board STL package {package.manifest_path.name}.")

    def _load_board_mesh_transform_into_vars(self) -> None:
        values = load_board_mesh_transform_overrides().get(board_override_key(self.board_path), {})
        x_deg = float(values.get("x_deg", 0.0))
        y_deg = float(values.get("y_deg", 0.0))
        z_deg = float(values.get("z_deg", 0.0))
        self.board_rotation_x_var.set(f"{x_deg:.1f}" if abs(x_deg) > 1e-9 else "")
        self.board_rotation_y_var.set(f"{y_deg:.1f}" if abs(y_deg) > 1e-9 else "")
        self.board_rotation_z_var.set(f"{z_deg:.1f}" if abs(z_deg) > 1e-9 else "")
        self.board_flip_x_var.set(bool(values.get("flip_x", False)))
        self.board_flip_y_var.set(bool(values.get("flip_y", False)))
        self.board_flip_z_var.set(bool(values.get("flip_z", False)))

    def _set_pending_female_pin_definition(self, reference: str | None, definition_path: Path | None) -> None:
        self.pending_female_pin_reference = reference
        self.pending_female_pin_definition_path = definition_path
        if definition_path is not None and reference is not None:
            self.place_female_pin_button.configure(state="normal")
        else:
            self.place_female_pin_button.configure(state="disabled")

    def _apply_filter(self) -> None:
        query = self.search_var.get().strip().lower()
        if query:
            self.filtered_components = [
                component
                for component in self.all_components
                if query in component.reference.lower()
                or query in component.value.lower()
                or query in component.footprint.lower()
            ]
        else:
            self.filtered_components = list(self.all_components)

        for item_id in self.tree.get_children():
            self.tree.delete(item_id)

        for index, component in enumerate(self.filtered_components):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    component.reference,
                    component.value,
                    component.footprint,
                    component.layer,
                    f"{component.body_width_mm:.2f} x {component.body_height_mm:.2f} x {component.body_thickness_mm:.2f}",
                    component.height_source,
                    component.pad_count,
                    f"{component.x_mm:.2f}, {component.y_mm:.2f}",
                ),
            )

        if self.filtered_components:
            first_id = self.tree.get_children()[0]
            self.tree.selection_set(first_id)
            self.tree.focus(first_id)
            self._show_selected_component()
        else:
            self._clear_details()

        self.status_var.set(
            f"Loaded {len(self.all_components)} components from {self.board_path.name} | Showing {len(self.filtered_components)}"
        )
        self._redraw_board_map()

    def _show_selected_component(self, _event: object | None = None) -> None:
        selection = self.tree.selection()
        if not selection:
            self._clear_details()
            return
        component = self.filtered_components[int(selection[0])]
        self.detail_vars["reference"].set(component.reference)
        self.detail_vars["value"].set(component.value or "-")
        self.detail_vars["footprint"].set(component.footprint)
        self.detail_vars["layer"].set(component.layer)
        self.detail_vars["position"].set(f"{component.x_mm:.3f} mm, {component.y_mm:.3f} mm")
        self.detail_vars["rotation"].set(f"{component.rotation_deg:.1f} deg")
        self.detail_vars["body"].set(
            f"{component.body_width_mm:.3f} mm x {component.body_height_mm:.3f} mm x {component.body_thickness_mm:.3f} mm"
        )
        self.detail_vars["height"].set(f"{component.body_thickness_mm:.3f} mm")
        self.detail_vars["pad_span"].set(f"{component.pad_span_width_mm:.3f} mm x {component.pad_span_height_mm:.3f} mm")
        self.detail_vars["pad_count"].set(str(component.pad_count))
        self.detail_vars["bounds_source"].set(component.bounds_source)
        self.detail_vars["height_source"].set(component.height_source)
        self.detail_vars["model_path"].set(component.model_path or "-")
        self.height_override_var.set(f"{component.body_thickness_mm:.3f}" if component.height_source == "manual override" else "")
        self.offset_x_var.set(f"{component.placement_offset_x_mm:.3f}" if abs(component.placement_offset_x_mm) > 1e-9 else "")
        self.offset_y_var.set(f"{component.placement_offset_y_mm:.3f}" if abs(component.placement_offset_y_mm) > 1e-9 else "")
        rotation_overrides = load_asset_rotation_overrides().get(board_override_key(self.board_path), {}).get(component.reference, {})
        self.rotation_x_var.set(f"{float(rotation_overrides.get('x_deg', 0.0)):.1f}" if abs(float(rotation_overrides.get("x_deg", 0.0))) > 1e-9 else "")
        self.rotation_y_var.set(f"{float(rotation_overrides.get('y_deg', 0.0)):.1f}" if abs(float(rotation_overrides.get("y_deg", 0.0))) > 1e-9 else "")
        self.rotation_z_var.set(f"{float(rotation_overrides.get('z_deg', 0.0)):.1f}" if abs(float(rotation_overrides.get("z_deg", 0.0))) > 1e-9 else "")
        self._highlight_component_on_map(component.reference)
        self._write_catalog_selection_to_bridge(component.reference)

    def _clear_details(self) -> None:
        for variable in self.detail_vars.values():
            variable.set("-")
        self.height_override_var.set("")
        self.offset_x_var.set("")
        self.offset_y_var.set("")
        self.rotation_x_var.set("")
        self.rotation_y_var.set("")
        self.rotation_z_var.set("")
        self._highlight_component_on_map(None)

    def _selected_component(self) -> ComponentRecord | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return self.filtered_components[int(selection[0])]

    def _save_height_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        height_mm = safe_float_or_none(self.height_override_var.get().strip())
        if height_mm is None or height_mm <= 0.0:
            messagebox.showerror("Invalid Height", "Enter a positive numeric height in mm.", parent=self.root)
            return
        overrides = load_height_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[component.reference] = height_mm
        save_height_overrides(overrides)
        self._load_board(self.board_path)
        self._reselect_reference(component.reference)

    def _clear_height_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        overrides = load_height_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.get(board_key, {})
        if component.reference in board_overrides:
            del board_overrides[component.reference]
            if not board_overrides and board_key in overrides:
                del overrides[board_key]
            save_height_overrides(overrides)
        self._load_board(self.board_path)
        self._reselect_reference(component.reference)

    def _save_placement_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        offset_x_mm = safe_float_or_none(self.offset_x_var.get().strip())
        offset_y_mm = safe_float_or_none(self.offset_y_var.get().strip())
        overrides = load_placement_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[component.reference] = {
            "x_mm": offset_x_mm if offset_x_mm is not None else 0.0,
            "y_mm": offset_y_mm if offset_y_mm is not None else 0.0,
        }
        save_placement_overrides(overrides)
        self._load_board(self.board_path)
        self._reselect_reference(component.reference)

    def _clear_placement_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        overrides = load_placement_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.get(board_key, {})
        if component.reference in board_overrides:
            del board_overrides[component.reference]
            if not board_overrides and board_key in overrides:
                del overrides[board_key]
            save_placement_overrides(overrides)
        self._load_board(self.board_path)
        self._reselect_reference(component.reference)

    def _save_asset_rotation_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        try:
            x_deg = float(self.rotation_x_var.get() or "0")
            y_deg = float(self.rotation_y_var.get() or "0")
            z_deg = float(self.rotation_z_var.get() or "0")
        except ValueError:
            messagebox.showerror("Invalid Rotation", "Enter valid X, Y, and Z rotation values.", parent=self.root)
            return
        overrides = load_asset_rotation_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[component.reference] = {
            "x_deg": x_deg,
            "y_deg": y_deg,
            "z_deg": z_deg,
        }
        save_asset_rotation_overrides(overrides)
        self.status_var.set(f"Saved 3D rotation override for {component.reference}.")
        self._show_selected_component()

    def _save_asset_rotation_override_for_reference(self, reference: str, x_deg: float, y_deg: float, z_deg: float) -> None:
        overrides = load_asset_rotation_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[reference] = {
            "x_deg": float(x_deg),
            "y_deg": float(y_deg),
            "z_deg": float(z_deg),
        }
        save_asset_rotation_overrides(overrides)
        self.status_var.set(f"Saved 3D rotation override for {reference}.")
        self._show_selected_component()

    def _clear_asset_rotation_override(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        overrides = load_asset_rotation_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.get(board_key, {})
        if component.reference in board_overrides:
            del board_overrides[component.reference]
        if not board_overrides and board_key in overrides:
            del overrides[board_key]
        save_asset_rotation_overrides(overrides)
        self.rotation_x_var.set("")
        self.rotation_y_var.set("")
        self.rotation_z_var.set("")
        self.status_var.set(f"Cleared 3D rotation override for {component.reference}.")
        self._show_selected_component()

    def _nudge_asset_rotation(self, axis_key: str, delta_deg: float) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        var_map = {
            "x_deg": self.rotation_x_var,
            "y_deg": self.rotation_y_var,
            "z_deg": self.rotation_z_var,
        }
        target_var = var_map[axis_key]
        current_value = float(target_var.get() or "0")
        target_var.set(f"{current_value + delta_deg:.1f}")
        self._save_asset_rotation_override()

    def _ask_xyz_rotation(
        self,
        *,
        parent: tk.Misc,
        title: str,
        reference: str,
        initial_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float] | str | None:
        result: dict[str, tuple[float, float, float] | str | None] = {"value": None}
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.attributes("-topmost", True)
        dialog.protocol("WM_DELETE_WINDOW", lambda: dialog.destroy())

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=f"Set X / Y / Z rotation for {reference}.", wraplength=300).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        x_var = tk.StringVar(value=f"{initial_xyz[0]:.1f}" if abs(initial_xyz[0]) > 1e-9 else "0")
        y_var = tk.StringVar(value=f"{initial_xyz[1]:.1f}" if abs(initial_xyz[1]) > 1e-9 else "0")
        z_var = tk.StringVar(value=f"{initial_xyz[2]:.1f}" if abs(initial_xyz[2]) > 1e-9 else "0")

        ttk.Label(frame, text="X (deg)").grid(row=1, column=0, sticky="w", pady=(10, 0))
        x_entry = ttk.Entry(frame, textvariable=x_var, width=12)
        x_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))

        ttk.Label(frame, text="Y (deg)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=y_var, width=12).grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))

        ttk.Label(frame, text="Z (deg)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=z_var, width=12).grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))

        button_row = ttk.Frame(frame)
        button_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        def submit() -> None:
            try:
                result["value"] = (float(x_var.get() or "0"), float(y_var.get() or "0"), float(z_var.get() or "0"))
            except ValueError:
                messagebox.showerror("Invalid Rotation", "Enter valid numeric X, Y, and Z rotation values.", parent=dialog)
                return
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        def clear_attached_stl() -> None:
            result["value"] = "clear_attached_stl"
            dialog.destroy()

        ttk.Button(button_row, text="OK", command=submit).grid(row=0, column=0, sticky="ew")
        ttk.Button(button_row, text="Cancel", command=cancel).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(button_row, text="Clear Attached STL", command=clear_attached_stl).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        dialog.bind("<Return>", lambda _event: submit())
        dialog.bind("<Escape>", lambda _event: cancel())
        x_entry.focus_set()
        dialog.wait_window()
        return result["value"]

    def _handle_viewer_rotation_request(self, reference: str) -> None:
        self._reselect_reference(reference)
        current_values = load_asset_rotation_overrides().get(board_override_key(self.board_path), {}).get(reference, {})
        response = self._ask_xyz_rotation(
            parent=self.root,
            title="Rotate Part",
            reference=reference,
            initial_xyz=(
                float(current_values.get("x_deg", 0.0)),
                float(current_values.get("y_deg", 0.0)),
                float(current_values.get("z_deg", 0.0)),
            ),
        )
        if response == "clear_attached_stl":
            self._clear_attached_stl_for_reference(reference)
            return
        if response is None:
            return
        self._save_asset_rotation_override_for_reference(reference, *response)

    def _save_board_mesh_transform_override(self) -> None:
        try:
            x_deg = float(self.board_rotation_x_var.get() or "0")
            y_deg = float(self.board_rotation_y_var.get() or "0")
            z_deg = float(self.board_rotation_z_var.get() or "0")
        except ValueError:
            messagebox.showerror("Invalid Board Transform", "Enter valid X, Y, and Z rotation values for the board STL.", parent=self.root)
            return
        overrides = load_board_mesh_transform_overrides()
        overrides[board_override_key(self.board_path)] = {
            "x_deg": x_deg,
            "y_deg": y_deg,
            "z_deg": z_deg,
            "flip_x": bool(self.board_flip_x_var.get()),
            "flip_y": bool(self.board_flip_y_var.get()),
            "flip_z": bool(self.board_flip_z_var.get()),
        }
        save_board_mesh_transform_overrides(overrides)
        self.status_var.set("Saved board STL transform override.")

    def _clear_board_mesh_transform_override(self) -> None:
        overrides = load_board_mesh_transform_overrides()
        board_key = board_override_key(self.board_path)
        if board_key in overrides:
            del overrides[board_key]
            save_board_mesh_transform_overrides(overrides)
        self.board_rotation_x_var.set("")
        self.board_rotation_y_var.set("")
        self.board_rotation_z_var.set("")
        self.board_flip_x_var.set(False)
        self.board_flip_y_var.set(False)
        self.board_flip_z_var.set(False)
        self.status_var.set("Cleared board STL transform override.")

    def _nudge_board_mesh_rotation(self, axis_key: str, delta_deg: float) -> None:
        var_map = {
            "x_deg": self.board_rotation_x_var,
            "y_deg": self.board_rotation_y_var,
            "z_deg": self.board_rotation_z_var,
        }
        target_var = var_map[axis_key]
        current_value = float(target_var.get() or "0")
        target_var.set(f"{current_value + delta_deg:.1f}")
        self._save_board_mesh_transform_override()

    def _generate_female_pin_array(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        size_text = simpledialog.askstring(
            "Female Pin Array",
            "Enter array size like 2x5:",
            parent=self.root,
            initialvalue="2x5",
        )
        if size_text is None:
            return
        try:
            count_x, count_y = parse_array_size(size_text)
            rotation_text = simpledialog.askstring(
                "Female Pin Rotation",
                "Enter rotation X,Y,Z in degrees.\nExamples: 90 or 0,0,90",
                parent=self.root,
                initialvalue="0,0,0",
            )
            if rotation_text is None:
                return
            rotation_deg_xyz = parse_rotation_xyz(rotation_text)
            meshes = build_female_pin_array_meshes_from_mate_project(
                resolve_mate_builder_project_path(),
                count_x,
                count_y,
            )
            fitted_meshes = fit_female_pin_array_to_component(meshes, component, rotation_deg_xyz)
            output_dir = FEMALE_PIN_ARRAY_OUTPUT_DIR / f"{component.reference}_{count_x}x{count_y}"
            output_dir.mkdir(parents=True, exist_ok=True)
            holder_path = output_dir / "female_pin_holder_array.stl"
            contact_path = output_dir / "female_pin_contact_array.stl"
            assembly_path = output_dir / "female_pin_array_assembly.stl"
            fitted_meshes["holder"].export(holder_path)
            fitted_meshes["contact"].export(contact_path)
            fitted_meshes["assembly"].export(assembly_path)
            definition_path = self._write_female_pin_definition(
                component=component,
                count_x=count_x,
                count_y=count_y,
                holder_path=holder_path,
                contact_path=contact_path,
                rotation_deg_xyz=rotation_deg_xyz,
            )
            self._set_pending_female_pin_definition(component.reference, definition_path)
            self.status_var.set(
                f"Preview ready for {component.reference}: female pin array {count_x}x{count_y}. Click Place to assign it to the board."
            )
            self._open_female_pin_array_preview(holder_path, contact_path, f"{component.reference} Female Pin {count_x}x{count_y}")
        except Exception as exc:
            messagebox.showerror("Female Pin Generation Failed", str(exc), parent=self.root)

    def _open_female_pin_array_preview(self, holder_path: Path, contact_path: Path, title: str) -> None:
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--preview-female-pin-holder",
                str(holder_path),
                "--preview-female-pin-contact",
                str(contact_path),
                "--preview-title",
                title,
            ],
            cwd=str(REPO_ROOT),
        )

    def _place_pending_female_pin(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        if self.pending_female_pin_definition_path is None or self.pending_female_pin_reference != component.reference:
            messagebox.showinfo(
                "No Preview Ready",
                "Generate a female pin preview for the selected component first, then click Place.",
                parent=self.root,
            )
            return
        self._assign_definition_to_reference(component.reference, self.pending_female_pin_definition_path)
        self.status_var.set(f"Placed generated female pin array on {component.reference}.")

    def _write_female_pin_definition(
        self,
        *,
        component: ComponentRecord,
        count_x: int,
        count_y: int,
        holder_path: Path,
        contact_path: Path,
        rotation_deg_xyz: tuple[float, float, float],
    ) -> Path:
        holder_mesh = load_mesh_mm(holder_path)
        holder_bounds = holder_mesh.bounds
        center_x = float((holder_bounds[0][0] + holder_bounds[1][0]) / 2.0)
        center_y = float((holder_bounds[0][1] + holder_bounds[1][1]) / 2.0)
        center_z = float((holder_bounds[0][2] + holder_bounds[1][2]) / 2.0)
        bbox_x = float(holder_bounds[1][0] - holder_bounds[0][0])
        bbox_y = float(holder_bounds[1][1] - holder_bounds[0][1])
        bbox_z = float(holder_bounds[1][2] - holder_bounds[0][2])
        component_family = "connector"
        component_name = safe_slug(f"{component.reference}_female_pin_{count_x}x{count_y}", "female_pin_array")
        definition_path = COMPONENT_LIBRARY_DIR / f"{component_family}__{component_name}.json"
        definition_payload = {
            "component_family": component_family,
            "component_name": component_name,
            "native_bbox_mm": {"x": bbox_x, "y": bbox_y, "z": bbox_z},
            "native_center_mm": {"x": center_x, "y": center_y, "z": center_z},
            "native_rotation_deg": {"x": rotation_deg_xyz[0], "y": rotation_deg_xyz[1], "z": rotation_deg_xyz[2]},
            "fit_part_names": ["holder"],
            "parts": [
                {
                    "part_name": "holder",
                    "stl_path": str(holder_path.resolve()),
                    "color": "#202020",
                    "height_mm": bbox_z,
                },
                {
                    "part_name": "contact",
                    "stl_path": str(contact_path.resolve()),
                    "color": "#d08a35",
                    "height_mm": bbox_z,
                },
            ],
        }
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(json.dumps(definition_payload, indent=2), encoding="utf-8")
        return definition_path

    def _write_attached_stl_definition(self, component: ComponentRecord, stl_paths: list[Path]) -> Path:
        meshes = [load_mesh_mm(path) for path in stl_paths]
        if not meshes:
            raise ValueError("No STL meshes were loaded.")
        combined_bounds = concatenate_meshes(meshes).bounds
        center_x = float((combined_bounds[0][0] + combined_bounds[1][0]) / 2.0)
        center_y = float((combined_bounds[0][1] + combined_bounds[1][1]) / 2.0)
        center_z = float((combined_bounds[0][2] + combined_bounds[1][2]) / 2.0)
        bbox_x = float(combined_bounds[1][0] - combined_bounds[0][0])
        bbox_y = float(combined_bounds[1][1] - combined_bounds[0][1])
        bbox_z = float(combined_bounds[1][2] - combined_bounds[0][2])
        component_family = "generic"
        component_name = safe_slug(f"{component.reference}_attached_stl", "attached_stl")
        definition_path = COMPONENT_LIBRARY_DIR / f"{component_family}__{component_name}.json"
        definition_payload = {
            "component_family": component_family,
            "component_name": component_name,
            "native_bbox_mm": {"x": bbox_x, "y": bbox_y, "z": bbox_z},
            "native_center_mm": {"x": center_x, "y": center_y, "z": center_z},
            "native_rotation_deg": {"x": 0.0, "y": 0.0, "z": 0.0},
            "parts": [
                {
                    "part_name": path.stem,
                    "stl_path": str(path.resolve()),
                    "color": color_for_name(path.stem),
                    "height_mm": bbox_z,
                }
                for path in stl_paths
            ],
        }
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(json.dumps(definition_payload, indent=2), encoding="utf-8")
        return definition_path

    def _assign_definition_to_reference(self, reference: str, definition_path: Path) -> None:
        overrides = load_component_reference_asset_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[reference] = str(definition_path.resolve())
        save_component_reference_asset_overrides(overrides)

    def _attach_stl_to_component(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        selected_paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Choose STL files to attach",
            initialdir=str(REPO_ROOT / "output"),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if not selected_paths:
            return
        try:
            stl_paths = [Path(path).expanduser().resolve() for path in selected_paths]
            definition_path = self._write_attached_stl_definition(component, stl_paths)
            self._assign_definition_to_reference(component.reference, definition_path)
            self.status_var.set(f"Attached {len(stl_paths)} STL file(s) to {component.reference}.")
            self._show_selected_component()
        except Exception as exc:
            messagebox.showerror("Attach STL Failed", str(exc), parent=self.root)

    def _clear_attached_stl_for_reference(self, reference: str) -> bool:
        overrides = load_component_reference_asset_overrides()
        board_key = board_override_key(self.board_path)
        board_overrides = overrides.setdefault(board_key, {})
        board_overrides[reference] = BOUNDING_BOX_ONLY_ASSET_OVERRIDE
        save_component_reference_asset_overrides(overrides)
        self.asset_definition_cache.clear()
        self.status_var.set(f"{reference} now uses its own bounding box instead of an attached/shared STL.")
        self._show_selected_component()
        return True

    def _clear_attached_stl_for_component(self) -> None:
        component = self._selected_component()
        if component is None:
            messagebox.showinfo("No Selection", "Select a component first.", parent=self.root)
            return
        self._clear_attached_stl_for_reference(component.reference)

    def _reselect_reference(self, reference: str) -> None:
        for index, component in enumerate(self.filtered_components):
            if component.reference == reference:
                item_id = str(index)
                self.tree.selection_set(item_id)
                self.tree.focus(item_id)
                self.tree.see(item_id)
                self._show_selected_component()
                break

    def _canvas_point(self, point: tuple[float, float], scale: float, origin_x: float, origin_y: float) -> tuple[float, float]:
        x_mm, y_mm = point
        return (origin_x + (x_mm * scale), origin_y - (y_mm * scale))

    def _board_map_transform(self) -> tuple[float, float, float]:
        width = max(self.board_canvas.winfo_width(), 40)
        height = max(self.board_canvas.winfo_height(), 40)
        min_x, min_y, max_x, max_y = self.board_bounds
        board_width = max(max_x - min_x, 1.0)
        board_height = max(max_y - min_y, 1.0)
        padding = 20.0
        scale = min((width - (2 * padding)) / board_width, (height - (2 * padding)) / board_height)
        origin_x = padding - (min_x * scale)
        origin_y = padding + (max_y * scale)
        return scale, origin_x, origin_y

    def _redraw_board_map(self) -> None:
        if not hasattr(self, "board_canvas"):
            return
        self.board_canvas.delete("all")
        self.component_item_ids.clear()
        self.canvas_item_to_reference.clear()
        scale, origin_x, origin_y = self._board_map_transform()

        for segment in self.outline_segments:
            points = outline_segment_points(segment)
            canvas_points: list[float] = []
            for point in points:
                canvas_x, canvas_y = self._canvas_point(point, scale, origin_x, origin_y)
                canvas_points.extend([canvas_x, canvas_y])
            if len(canvas_points) >= 4:
                self.board_canvas.create_line(*canvas_points, fill="#2d241f", width=2.0, smooth=False)

        for component in self.filtered_components:
            corners = component_world_corners(component)
            flat_points: list[float] = []
            for point in corners + [corners[0]]:
                canvas_x, canvas_y = self._canvas_point(point, scale, origin_x, origin_y)
                flat_points.extend([canvas_x, canvas_y])
            fill_points = flat_points[:-2]
            fill_id = self.board_canvas.create_polygon(
                *fill_points,
                fill="#2c7a6b",
                outline="",
                stipple="gray25",
            )
            outline_id = self.board_canvas.create_line(*flat_points, fill="#2c7a6b", width=1.5)
            label_x, label_y = self._canvas_point(component_body_center_in_world(component), scale, origin_x, origin_y)
            text_id = self.board_canvas.create_text(
                label_x,
                label_y,
                text=component.reference,
                fill="#5a1b1b",
                font=("Segoe UI", 7, "bold"),
            )
            self.component_item_ids[component.reference] = outline_id
            self.component_item_ids[f"{component.reference}__label"] = text_id
            self.component_item_ids[f"{component.reference}__fill"] = fill_id
            self.canvas_item_to_reference[fill_id] = component.reference
            self.canvas_item_to_reference[outline_id] = component.reference
            self.canvas_item_to_reference[text_id] = component.reference

        selected = self._selected_component()
        self._highlight_component_on_map(selected.reference if selected is not None else None)

    def _highlight_component_on_map(self, reference: str | None) -> None:
        if not hasattr(self, "board_canvas"):
            return
        for component in self.filtered_components:
            outline_id = self.component_item_ids.get(component.reference)
            label_id = self.component_item_ids.get(f"{component.reference}__label")
            fill_id = self.component_item_ids.get(f"{component.reference}__fill")
            if outline_id is not None:
                self.board_canvas.itemconfigure(
                    outline_id,
                    fill="#cc5a1a" if component.reference == reference else "#2c7a6b",
                    width=3.0 if component.reference == reference else 1.5,
                )
            if fill_id is not None:
                self.board_canvas.itemconfigure(
                    fill_id,
                    fill="#f28f3b" if component.reference == reference else "#2c7a6b",
                )
            if label_id is not None:
                self.board_canvas.itemconfigure(
                    label_id,
                    fill="#111111" if component.reference == reference else "#5a1b1b",
                )

    def _on_board_canvas_click(self, event) -> None:
        item_ids = self.board_canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if not item_ids:
            return
        for item_id in reversed(item_ids):
            reference = self.canvas_item_to_reference.get(int(item_id))
            if reference is not None:
                self._reselect_reference(reference)
                return

    def _on_detail_content_configure(self, _event) -> None:
        self.detail_canvas.configure(scrollregion=self.detail_canvas.bbox("all"))

    def _on_detail_canvas_configure(self, event) -> None:
        self.detail_canvas.itemconfigure(self.detail_canvas_window, width=event.width)

    def _on_detail_mousewheel(self, event) -> str | None:
        if getattr(event, "delta", 0):
            step = -1 * int(event.delta / 120) if event.delta else 0
        elif getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = 0
        if step != 0:
            self.detail_canvas.yview_scroll(step, "units")
            return "break"
        return None

    def _export_json(self) -> None:
        if not self.all_components:
            messagebox.showinfo("Nothing To Export", "No components are loaded.", parent=self.root)
            return
        default_name = f"{self.board_path.stem}_components.json"
        output_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export component catalog as JSON",
            initialdir=str(REPO_ROOT / "output"),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not output_path:
            return
        payload = {
            "source_board": str(self.board_path),
            "component_count": len(self.all_components),
            "components": [asdict(component) for component in self.all_components],
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        messagebox.showinfo("Export Complete", f"Saved component catalog to:\n{output_path}", parent=self.root)

    def _open_3d_viewer(self) -> None:
        if self.current_scene is None:
            messagebox.showinfo("No Board Loaded", "Load a board first.", parent=self.root)
            return
        try:
            self._close_3d_viewer()
            VIEWER_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
            self.viewer_bridge_path = VIEWER_BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
            selected = self._selected_component()
            initial_reference = selected.reference if selected is not None else None
            self.viewer_bridge_path.write_text(
                json.dumps(
                    {
                        "selected_reference_from_catalog": initial_reference,
                        "selected_reference_from_viewer": None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--input-pcb",
                    str(self.board_path),
                    "--viewer-bridge",
                    str(self.viewer_bridge_path),
                    *(["--board-mesh-package", str(self.board_mesh_package.manifest_path)] if self.board_mesh_package is not None else []),
                ],
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            messagebox.showerror("3D Viewer Failed", f"Could not open the 3D viewer.\n\n{exc}", parent=self.root)

    def _close_3d_viewer(self) -> None:
        if self.viewer_process is not None:
            if self.viewer_process.poll() is None:
                self.viewer_process.terminate()
            self.viewer_process = None
        if self.viewer_bridge_path is not None and self.viewer_bridge_path.exists():
            try:
                self.viewer_bridge_path.unlink()
            except OSError:
                pass
        self.viewer_bridge_path = None
        self._last_viewer_selected_reference = None

    def _write_catalog_selection_to_bridge(self, reference: str | None) -> None:
        if self.viewer_bridge_path is None or not self.viewer_bridge_path.exists():
            return
        try:
            payload = json.loads(self.viewer_bridge_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload["selected_reference_from_catalog"] = reference
        self.viewer_bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _poll_viewer_bridge(self) -> None:
        try:
            if self.viewer_process is not None and self.viewer_process.poll() is not None:
                self._close_3d_viewer()
            if self.viewer_bridge_path is not None and self.viewer_bridge_path.exists():
                payload = json.loads(self.viewer_bridge_path.read_text(encoding="utf-8"))
                selected_reference = payload.get("selected_reference_from_viewer")
                rotation_request = payload.get("rotation_request_from_viewer")
                if (
                    isinstance(selected_reference, str)
                    and selected_reference
                    and selected_reference != self._last_viewer_selected_reference
                ):
                    self._last_viewer_selected_reference = selected_reference
                    self._reselect_reference(selected_reference)
                if (
                    isinstance(rotation_request, str)
                    and rotation_request
                    and rotation_request != self._last_viewer_rotation_request
                ):
                    self._last_viewer_rotation_request = rotation_request
                    payload["rotation_request_from_viewer"] = None
                    self.viewer_bridge_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    self._handle_viewer_rotation_request(rotation_request)
        except Exception:
            pass
        self.root.after(150, self._poll_viewer_bridge)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._close_3d_viewer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Browse PCB components and footprint-derived dimensions.")
    parser.add_argument("--input-pcb", type=Path, default=DEFAULT_INPUT_PCB, help="Path to a KiCad .kicad_pcb file.")
    parser.add_argument("--dump-json", action="store_true", help="Print the parsed component catalog as JSON and exit.")
    parser.add_argument("--viewer-bridge", type=Path, default=None, help="Internal: bridge file for the standalone 3D viewer.")
    parser.add_argument("--board-mesh-package", type=Path, default=None, help="Optional board STL package manifest JSON for the 3D viewer.")
    parser.add_argument("--preview-female-pin-holder", type=Path, default=None, help="Internal: preview generated female pin holder array STL.")
    parser.add_argument("--preview-female-pin-contact", type=Path, default=None, help="Internal: preview generated female pin contact array STL.")
    parser.add_argument("--preview-title", default="Female Pin Array Preview", help="Internal: preview window title.")
    args = parser.parse_args(argv)

    if args.preview_female_pin_holder is not None and args.preview_female_pin_contact is not None:
        preview_female_pin_array(
            args.preview_female_pin_holder.expanduser().resolve(),
            args.preview_female_pin_contact.expanduser().resolve(),
            str(args.preview_title),
        )
        return 0

    board_path = args.input_pcb.expanduser().resolve()
    if args.dump_json:
        components = load_components_from_board(board_path)
        print(json.dumps([asdict(component) for component in components], indent=2))
        return 0

    if args.viewer_bridge is not None:
        scene = load_board_scene(board_path)
        board_mesh_package = load_board_mesh_package(args.board_mesh_package.expanduser().resolve()) if args.board_mesh_package is not None else auto_detect_board_mesh_package(board_path)
        viewer = Component3DViewer(
            scene,
            asset_library=load_component_asset_library(),
            board_mesh_package=board_mesh_package,
            bridge_path=args.viewer_bridge.expanduser().resolve(),
        )
        viewer.show()
        return 0

    app = ComponentCatalogApp(board_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
