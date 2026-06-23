from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tkinter as tk
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import trimesh
from vedo import Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "part_generator"
DEFAULT_PROJECT_PATH = DEFAULT_OUTPUT_DIR / "part_generator_project.json"
DEFAULT_EXPORT_DIR = DEFAULT_OUTPUT_DIR / "exports"
DEFAULT_HOTKEY_SETTINGS_PATH = DEFAULT_OUTPUT_DIR / "hotkey_settings.json"
DEFAULT_COMPONENT_LIBRARY_DIR = REPO_ROOT / "output" / "component_maker" / "library"
BRIDGE_DIR = DEFAULT_OUTPUT_DIR / "viewer_bridge"
CANVAS_BACKGROUND = "#f7f0e4"
CANVAS_GRID = "#e6dac7"
CANVAS_AXIS = "#c7b39b"
DEFAULT_CANVAS_WIDTH = 900
DEFAULT_CANVAS_HEIGHT = 640
DEFAULT_SCALE_PX_PER_MM = 20.0
DEFAULT_PART_HEIGHT_MM = 0.25
DEFAULT_PART_COLOR = "#d38b5d"
DEFAULT_MATE_SNAP_DISTANCE_PX = 14.0
DEFAULT_ALIGNMENT_SNAP_DISTANCE_PX = 10.0
PARALLEL_LINE_TOLERANCE = 1e-3
DEFAULT_ARROW_NUDGE_MM = 0.1
DEFAULT_ARROW_NUDGE_LARGE_MM = 1.0
MAX_UNDO_HISTORY = 50

HOTKEY_ACTIONS: list[tuple[str, str, str]] = [
    ("create_new_part", "Create New Part", ""),
    ("save_part", "Add / Update Part", ""),
    ("delete_active_part", "Delete Part", ""),
    ("undo_point", "Undo Point", "<KeyPress-b>"),
    ("undo_last_action", "Undo Last Action", "<Control-z>"),
    ("finish_closed_shape", "Finish Closed Shape", "<Return>"),
    ("finish_open_line", "Finish Open Line", "<Shift-Return>"),
    ("cancel_draft", "Cancel Draft", "<Escape>"),
    ("assign_selected_to_active_part", "Assign Selected To Active Part", ""),
    ("delete_selected_contour", "Delete Selected Contour", "<Delete>"),
    ("center_selected_contour", "Center Selected To Origin", "<Control-o>"),
    ("mirror_selected_x", "Mirror Selected Across X", ""),
    ("mirror_selected_y", "Mirror Selected Across Y", ""),
    ("pick_distance_points", "Pick Distance Points", "<KeyPress-d>"),
    ("apply_distance_mate", "Apply Distance Mate", "<Control-d>"),
    ("clear_distance_mate_points", "Clear Distance Picks", ""),
    ("pick_parallel_lines", "Pick Parallel Lines", "<KeyPress-p>"),
    ("apply_parallel_line_distance", "Apply Parallel Line Distance", "<Control-p>"),
    ("clear_parallel_line_points", "Clear Parallel Picks", ""),
    ("save_project", "Save Project", ""),
    ("load_project", "Load Project", ""),
    ("preview_3d", "Preview 3D", ""),
    ("export_stls", "Export STLs", ""),
]


@dataclass(slots=True)
class PartDefinition:
    name: str
    color: str
    height_mm: float


@dataclass(slots=True)
class ContourRecord:
    contour_id: str
    part_name: str
    points_mm: list[tuple[float, float]]
    closed: bool


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
            raise ValueError("Failed to triangulate shape. Check for self-intersections or overlapping edges.")

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


def validate_contours(contours: list[ContourRecord]) -> None:
    open_contours = [contour for contour in contours if not contour.closed]
    if open_contours:
        raise ValueError(
            "All sketches must be closed before generating 3D geometry. "
            f"Open contour count: {len(open_contours)}."
        )

    for contour in contours:
        simplified = _simplify_profile_points(contour.points_mm)
        if len(simplified) < 3:
            raise ValueError(f"Contour {contour.contour_id} does not have enough unique points.")
        if abs(_signed_area(simplified)) <= 1e-9:
            raise ValueError(f"Contour {contour.contour_id} has zero area.")
        _triangulate_polygon(simplified)


def build_part_meshes(parts: dict[str, PartDefinition], contours: list[ContourRecord]) -> dict[str, trimesh.Trimesh]:
    validate_contours(contours)
    grouped: dict[str, list[trimesh.Trimesh]] = {}
    for contour in contours:
        part = parts.get(contour.part_name)
        if part is None:
            raise ValueError(f"Contour {contour.contour_id} references unknown part {contour.part_name!r}.")
        grouped.setdefault(part.name, []).append(extrude_closed_polygon(contour.points_mm, part.height_mm))

    result: dict[str, trimesh.Trimesh] = {}
    for part_name, meshes in grouped.items():
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)
        result[part_name] = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    return result


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


def _serialize_parts(parts: dict[str, PartDefinition]) -> list[dict]:
    return [asdict(part) for part in parts.values()]


def _serialize_contours(contours: list[ContourRecord]) -> list[dict]:
    return [asdict(contour) for contour in contours]


def _deserialize_parts(payload_parts: list[dict]) -> dict[str, PartDefinition]:
    return {item["name"]: PartDefinition(**item) for item in payload_parts}


def _deserialize_contours(payload_contours: list[dict]) -> list[ContourRecord]:
    return [
        ContourRecord(
            contour_id=item["contour_id"],
            part_name=item["part_name"],
            points_mm=[tuple(point) for point in item["points_mm"]],
            closed=bool(item["closed"]),
        )
        for item in payload_contours
    ]


def _build_state_snapshot(
    parts: dict[str, PartDefinition],
    contours: list[ContourRecord],
    *,
    action_label: str = "",
    selected_contour_id: str | None = None,
    active_part_name: str = "",
) -> dict:
    return {
        "action_label": action_label,
        "parts": _serialize_parts(parts),
        "contours": _serialize_contours(contours),
        "selected_contour_id": selected_contour_id,
        "active_part_name": active_part_name,
    }


def build_payload(
    parts: dict[str, PartDefinition],
    contours: list[ContourRecord],
    status_message: str = "",
    history: list[dict] | None = None,
    component_export: dict | None = None,
) -> dict:
    return {
        "parts": _serialize_parts(parts),
        "contours": _serialize_contours(contours),
        "status_message": status_message,
        "history": history or [],
        "component_export": component_export or {},
    }


class PartExtrusionViewer:
    def __init__(self, bridge_path: Path) -> None:
        self.bridge_path = bridge_path
        self.plotter = Plotter(
            title="PCB Part 2D Extrusion Preview",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1200, 820),
        )
        self.info = Text2D("", pos="top-left", s=0.8, c="#2d241f", bg=None, font="Courier")
        self.actors: list = []
        self.last_signature: tuple | None = None

    def _payload_signature(self, payload: dict) -> tuple:
        return (
            tuple((part["name"], part["color"], part["height_mm"]) for part in payload.get("parts", [])),
            tuple(
                (
                    contour["contour_id"],
                    contour["part_name"],
                    contour["closed"],
                    tuple(tuple(point) for point in contour["points_mm"]),
                )
                for contour in payload.get("contours", [])
            ),
            payload.get("status_message", ""),
        )

    def _build_scene(self, payload: dict) -> None:
        for actor in self.actors:
            self.plotter.remove(actor)
        self.actors.clear()

        parts = {item["name"]: PartDefinition(**item) for item in payload.get("parts", [])}
        contours = [
            ContourRecord(
                contour_id=item["contour_id"],
                part_name=item["part_name"],
                points_mm=[tuple(point) for point in item["points_mm"]],
                closed=bool(item["closed"]),
            )
            for item in payload.get("contours", [])
        ]

        status_message = str(payload.get("status_message", "")).strip()
        if parts and contours:
            try:
                part_meshes = build_part_meshes(parts, contours)
                for part_name, mesh in part_meshes.items():
                    part = parts[part_name]
                    actor = Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(part.color).alpha(1.0)
                    self.actors.append(actor)
                    self.plotter += actor
                summary = f"Parts: {len(parts)}  Closed contours: {len(contours)}"
            except Exception as exc:
                summary = f"Preview blocked: {exc}"
        else:
            summary = "Draw at least one closed contour to preview."

        self.info.text(
            "PCB Part Generator Preview\n"
            f"{summary}\n"
            f"{status_message or 'Use the editor to draw, assign parts, and extrude.'}"
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


class PartGeneratorControlPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("PCB Part 2D Extruder")
        self.root.geometry("1400x860")
        self.root.minsize(1200, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.scale_px_per_mm = DEFAULT_SCALE_PX_PER_MM
        self.parts: dict[str, PartDefinition] = {}
        self.contours: list[ContourRecord] = []
        self.canvas_item_to_contour: dict[int, str] = {}
        self.canvas_item_to_vertex: dict[int, tuple[str, int]] = {}
        self.selected_contour_id: str | None = None
        self.current_points_px: list[tuple[float, float]] = []
        self.draft_part_name: str | None = None
        self.preview_line_id: int | None = None
        self.preview_cursor_px: tuple[float, float] | None = None
        self.mated_target_px: tuple[float, float] | None = None
        self.dragging_vertex_ref: tuple[str, int] | None = None
        self.alignment_snap_x_px: float | None = None
        self.alignment_snap_y_px: float | None = None
        self.distance_mate_pick_active = False
        self.distance_mate_points: list[tuple[str, int]] = []
        self.distance_mate_has_scaled = False
        self.parallel_line_pick_active = False
        self.parallel_line_points: list[tuple[str, int]] = []
        self.point_lasso_enabled = False
        self.selected_vertex_refs: set[tuple[str, int]] = set()
        self.lasso_start_px: tuple[float, float] | None = None
        self.lasso_current_px: tuple[float, float] | None = None
        self.group_drag_anchor_px: tuple[float, float] | None = None
        self.alt_pressed = False
        self._syncing_part_selection = False
        self.view_offset_px = (0.0, 0.0)
        self.is_panning_canvas = False
        self.last_pan_anchor_px: tuple[float, float] | None = None

        self.active_part_var = tk.StringVar(value="")
        self.part_name_var = tk.StringVar(value="")
        self.part_color_var = tk.StringVar(value=DEFAULT_PART_COLOR)
        self.part_height_var = tk.DoubleVar(value=DEFAULT_PART_HEIGHT_MM)
        self.point_mate_enabled_var = tk.BooleanVar(value=False)
        self.point_mate_button_var = tk.StringVar(value="Point Auto-Mate: OFF")
        self.axis_snap_enabled_var = tk.BooleanVar(value=True)
        self.angle_constraint_enabled_var = tk.BooleanVar(value=False)
        self.angle_constraint_deg_var = tk.DoubleVar(value=90.0)
        self.distance_mate_distance_var = tk.DoubleVar(value=1.0)
        self.parallel_line_distance_var = tk.DoubleVar(value=1.0)
        self.scale_bbox_x_var = tk.DoubleVar(value=1.0)
        self.scale_bbox_y_var = tk.DoubleVar(value=1.0)
        self.scale_bbox_z_var = tk.DoubleVar(value=1.0)
        self.component_family_var = tk.StringVar(value="generic")
        self.component_name_var = tk.StringVar(value="generic_component")
        self.component_rotation_x_var = tk.DoubleVar(value=0.0)
        self.component_rotation_y_var = tk.DoubleVar(value=0.0)
        self.component_rotation_z_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Select an active part or create one before drawing.")
        self.current_draft_axis_mode: str | None = None
        self.hotkey_vars: dict[str, tk.StringVar] = {
            action_id: tk.StringVar(value=default_sequence)
            for action_id, _label, default_sequence in HOTKEY_ACTIONS
        }
        self.hotkey_binding_ids: dict[str, str] = {}
        self.hotkey_capture_target: str | None = None
        self.hotkey_capture_modifiers: set[str] = set()
        self.hotkey_capture_status_var = tk.StringVar(value="")
        self.hotkey_display_vars: dict[str, tk.StringVar] = {
            action_id: tk.StringVar(value="")
            for action_id, _label, _default_sequence in HOTKEY_ACTIONS
        }
        self.undo_history: list[dict] = []
        self.pending_drag_undo_snapshot: dict | None = None
        self.pending_drag_description = ""

        self.bridge_path = BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
        self.viewer_process: subprocess.Popen | None = None
        self.current_project_path: Path | None = args.project.expanduser().resolve() if args.project is not None else None

        self._build_ui()
        self._load_hotkey_settings()
        self._apply_hotkey_bindings(announce=False, persist=False)
        self._refresh_part_tree()
        self._redraw_canvas()
        self._start_viewer()
        if args.project is not None and args.project.exists():
            self._load_project(args.project)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer, padding=(0, 0, 10, 0))
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(left, highlightthickness=0, width=360)
        left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar.grid(row=0, column=1, sticky="ns")

        left_content = ttk.Frame(self.left_canvas, padding=2)
        left_window_id = self.left_canvas.create_window((0, 0), window=left_content, anchor="nw")

        def on_left_frame_configure(_event) -> None:
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def on_left_canvas_configure(event) -> None:
            self.left_canvas.itemconfigure(left_window_id, width=event.width)

        def on_left_mousewheel(event) -> None:
            if getattr(event, "delta", 0):
                step = -1 * int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                step = -1
            elif getattr(event, "num", None) == 5:
                step = 1
            else:
                step = 0
            if step:
                self.left_canvas.yview_scroll(step, "units")

        def bind_left_mousewheel(_event) -> None:
            self.left_canvas.bind_all("<MouseWheel>", on_left_mousewheel)
            self.left_canvas.bind_all("<Button-4>", on_left_mousewheel)
            self.left_canvas.bind_all("<Button-5>", on_left_mousewheel)

        def unbind_left_mousewheel(_event) -> None:
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")

        left_content.bind("<Configure>", on_left_frame_configure)
        self.left_canvas.bind("<Configure>", on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", unbind_left_mousewheel)

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(left_content, text="PCB Part Generator", font=("Georgia", 16, "bold")).pack(anchor="w")
        ttk.Label(
            left_content,
            text="Draw 2D contours, assign each contour to a material part, and extrude each part into 3D.",
            wraplength=320,
        ).pack(anchor="w", pady=(4, 12))

        part_box = ttk.LabelFrame(left_content, text="Part / Material", padding=10)
        part_box.pack(fill="x")

        ttk.Label(part_box, text="Active Part").pack(anchor="w")
        self.active_part_combo = ttk.Combobox(
            part_box,
            textvariable=self.active_part_var,
            values=sorted(self.parts.keys()),
            state="readonly",
            width=28,
        )
        self.active_part_combo.pack(fill="x", pady=(2, 8))
        self.active_part_combo.bind("<<ComboboxSelected>>", self._on_active_part_changed)

        ttk.Label(part_box, text="Part Name").pack(anchor="w")
        ttk.Entry(part_box, textvariable=self.part_name_var).pack(fill="x", pady=(2, 8))

        ttk.Label(part_box, text="Material Color").pack(anchor="w")
        color_row = ttk.Frame(part_box)
        color_row.pack(fill="x", pady=(2, 8))
        ttk.Entry(color_row, textvariable=self.part_color_var, width=18).pack(side="left", fill="x", expand=True)
        ttk.Button(color_row, text="Pick", command=self._choose_part_color).pack(side="left", padx=(8, 0))

        ttk.Label(part_box, text="Extrusion Height (mm)").pack(anchor="w")
        ttk.Entry(part_box, textvariable=self.part_height_var).pack(fill="x", pady=(2, 10))

        button_row = ttk.Frame(part_box)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Add / Update Part", command=self._upsert_part).pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Delete Part", command=self._delete_active_part).pack(side="left", fill="x", expand=True, padx=(8, 0))

        part_list_header = ttk.Frame(left_content)
        part_list_header.pack(fill="x", pady=(10, 0))
        ttk.Label(part_list_header, text="Parts").pack(side="left")
        ttk.Button(part_list_header, text="+", width=3, command=self._create_new_part).pack(side="right")

        self.part_tree = ttk.Treeview(left_content, columns=("name", "height", "color"), show="headings", height=6, selectmode="extended")
        self.part_tree.heading("name", text="Part")
        self.part_tree.heading("height", text="Height (mm)")
        self.part_tree.heading("color", text="Color")
        self.part_tree.column("name", width=140, anchor="w")
        self.part_tree.column("height", width=90, anchor="center")
        self.part_tree.column("color", width=120, anchor="center")
        self.part_tree.pack(fill="x", pady=(6, 0))
        self.part_tree.bind("<<TreeviewSelect>>", self._on_part_tree_selected)

        part_ops_box = ttk.LabelFrame(left_content, text="Selected Parts", padding=10)
        part_ops_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            part_ops_box,
            text="Use Ctrl or Shift in the parts list to select multiple parts for transform operations.",
            wraplength=320,
        ).pack(anchor="w")
        ttk.Button(part_ops_box, text="Center Selected Parts To Origin", command=self._center_selected_parts_to_origin).pack(fill="x", pady=(10, 0))
        ttk.Label(part_ops_box, text="Target Bounding Box X / Y / Z (mm)").pack(anchor="w", pady=(10, 0))
        scale_row = ttk.Frame(part_ops_box)
        scale_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(scale_row, textvariable=self.scale_bbox_x_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(scale_row, textvariable=self.scale_bbox_y_var, width=8).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Entry(scale_row, textvariable=self.scale_bbox_z_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Button(part_ops_box, text="Scale Selected Parts To Bounding Box", command=self._scale_selected_parts_to_bbox).pack(fill="x", pady=(8, 0))

        draw_box = ttk.LabelFrame(left_content, text="Drawing", padding=10)
        draw_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            draw_box,
            text="Left click adds points. Finish as a closed shape for 3D. Open lines can stay on the canvas, but preview/export will reject them.",
            wraplength=320,
        ).pack(anchor="w")

        ttk.Button(draw_box, text="Undo Point", command=self._undo_point).pack(fill="x", pady=(10, 0))
        ttk.Button(draw_box, text="Finish Closed Shape", command=self._finish_closed_shape).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Finish Open Line", command=self._finish_open_line).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Cancel Draft", command=self._cancel_draft).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Assign Selected To Active Part", command=self._assign_selected_to_active_part).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Delete Selected Contour", command=self._delete_selected_contour).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Center Selected To Origin", command=self._center_selected_contour_to_origin).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Mirror Selected Across X", command=lambda: self._mirror_selected_contour("x")).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Mirror Selected Across Y", command=lambda: self._mirror_selected_contour("y")).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Lasso Select Points", command=self._toggle_point_lasso_mode).pack(fill="x", pady=(8, 0))
        ttk.Button(draw_box, text="Clear Point Selection", command=self._clear_selected_vertices).pack(fill="x", pady=(8, 0))

        mate_box = ttk.LabelFrame(left_content, text="Mate / Snap", padding=10)
        mate_box.pack(fill="x", pady=(12, 0))
        ttk.Button(
            mate_box,
            textvariable=self.point_mate_button_var,
            command=self._toggle_point_auto_mate,
        ).pack(fill="x")
        ttk.Label(
            mate_box,
            text="Move close to a vertex to mate the next point exactly to it. Toggle this off when you want freer point placement.",
            wraplength=320,
        ).pack(anchor="w", pady=(4, 0))

        distance_box = ttk.LabelFrame(left_content, text="Distance Mate", padding=10)
        distance_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            distance_box,
            text="Pick two closed-shape vertices. The first applied distance rescales the whole model; later ones either resize a picked line in the same contour or move another contour to match the target distance.",
            wraplength=320,
        ).pack(anchor="w")
        ttk.Button(distance_box, text="Pick Distance Points", command=self._toggle_distance_mate_pick_mode).pack(fill="x", pady=(10, 0))
        ttk.Label(distance_box, text="Target Distance (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(distance_box, textvariable=self.distance_mate_distance_var).pack(fill="x", pady=(2, 0))
        ttk.Button(distance_box, text="Apply Distance Mate", command=self._apply_distance_mate).pack(fill="x", pady=(8, 0))
        ttk.Button(distance_box, text="Clear Distance Picks", command=self._clear_distance_mate_points).pack(fill="x", pady=(8, 0))

        parallel_box = ttk.LabelFrame(left_content, text="Parallel Line Distance", padding=10)
        parallel_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            parallel_box,
            text="Pick 4 vertices: two adjacent points for line 1, then two adjacent points for line 2. The tool sets the shortest perpendicular distance between the parallel lines.",
            wraplength=320,
        ).pack(anchor="w")
        ttk.Button(parallel_box, text="Pick Parallel Lines", command=self._toggle_parallel_line_pick_mode).pack(fill="x", pady=(10, 0))
        ttk.Label(parallel_box, text="Target Distance (mm)").pack(anchor="w", pady=(8, 0))
        ttk.Entry(parallel_box, textvariable=self.parallel_line_distance_var).pack(fill="x", pady=(2, 0))
        ttk.Button(parallel_box, text="Apply Parallel Line Distance", command=self._apply_parallel_line_distance).pack(fill="x", pady=(8, 0))
        ttk.Button(parallel_box, text="Clear Parallel Picks", command=self._clear_parallel_line_points).pack(fill="x", pady=(8, 0))

        angle_box = ttk.LabelFrame(left_content, text="Angle Constraint", padding=10)
        angle_box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(
            angle_box,
            text="Auto snap horizontal / vertical from previous point",
            variable=self.axis_snap_enabled_var,
            command=self._redraw_canvas,
        ).pack(anchor="w")
        ttk.Checkbutton(
            angle_box,
            text="Lock next segment angle",
            variable=self.angle_constraint_enabled_var,
            command=self._redraw_canvas,
        ).pack(anchor="w")
        ttk.Label(
            angle_box,
            text="Angle relative to the previous line, in degrees. Angle lock overrides the auto horizontal/vertical snap.",
            wraplength=320,
            ).pack(anchor="w", pady=(4, 4))
        ttk.Entry(angle_box, textvariable=self.angle_constraint_deg_var).pack(fill="x")

        io_box = ttk.LabelFrame(left_content, text="Project", padding=10)
        io_box.pack(fill="x", pady=(12, 0))
        ttk.Button(io_box, text="Save Project", command=self._save_project).pack(fill="x")
        ttk.Button(io_box, text="Load Project", command=self._prompt_load_project).pack(fill="x", pady=(8, 0))

        export_box = ttk.LabelFrame(left_content, text="Component Export", padding=10)
        export_box.pack(fill="x", pady=(12, 0))
        ttk.Label(export_box, text="Component Family").pack(anchor="w")
        ttk.Combobox(
            export_box,
            textvariable=self.component_family_var,
            values=("generic", "capacitor", "resistor"),
            state="readonly",
        ).pack(fill="x", pady=(2, 8))
        ttk.Label(export_box, text="Component Name").pack(anchor="w")
        ttk.Entry(export_box, textvariable=self.component_name_var).pack(fill="x", pady=(2, 8))
        ttk.Label(export_box, text="Library Rotation X / Y / Z (deg)").pack(anchor="w")
        export_rotation_row = ttk.Frame(export_box)
        export_rotation_row.pack(fill="x", pady=(4, 8))
        ttk.Entry(export_rotation_row, textvariable=self.component_rotation_x_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Entry(export_rotation_row, textvariable=self.component_rotation_y_var, width=8).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Entry(export_rotation_row, textvariable=self.component_rotation_z_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Button(io_box, text="Preview 3D", command=self._push_payload).pack(fill="x", pady=(8, 0))
        ttk.Button(export_box, text="Export STLs + Library JSON", command=self._export_stls).pack(fill="x")

        ttk.Button(io_box, text="Hotkey Settings", command=self._open_hotkey_settings).pack(fill="x", pady=(8, 0))

        ttk.Label(left_content, textvariable=self.status_var, wraplength=320).pack(anchor="w", pady=(12, 0))

        ttk.Label(
            right,
            text="2D Sketch",
            font=("Georgia", 15, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

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
        self.root.bind("<KeyPress-Alt_L>", self._on_alt_press)
        self.root.bind("<KeyPress-Alt_R>", self._on_alt_press)
        self.root.bind("<KeyRelease-Alt_L>", self._on_alt_release)
        self.root.bind("<KeyRelease-Alt_R>", self._on_alt_release)
        self.root.bind("<Left>", lambda event: self._on_arrow_nudge(event, -1.0, 0.0))
        self.root.bind("<Right>", lambda event: self._on_arrow_nudge(event, 1.0, 0.0))
        self.root.bind("<Up>", lambda event: self._on_arrow_nudge(event, 0.0, 1.0))
        self.root.bind("<Down>", lambda event: self._on_arrow_nudge(event, 0.0, -1.0))

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

    def _part_tree_highlighted_name(self) -> str:
        selection = self.part_tree.selection()
        if not selection:
            return ""
        return str(selection[0]).strip()

    def _selected_part_names(self) -> list[str]:
        return [str(item_id).strip() for item_id in self.part_tree.selection() if str(item_id).strip()]

    def _selected_part_contours(self) -> list[ContourRecord]:
        selected_names = set(self._selected_part_names())
        if not selected_names:
            return []
        return [contour for contour in self.contours if contour.part_name in selected_names]

    def _selected_parts_bbox_mm(self) -> tuple[float, float, float, float, float]:
        contours = self._selected_part_contours()
        if not contours:
            raise ValueError("Select one or more parts from the parts list first.")
        x_values = [x_mm for contour in contours for x_mm, _y_mm in contour.points_mm]
        y_values = [y_mm for contour in contours for _x_mm, y_mm in contour.points_mm]
        if not x_values or not y_values:
            raise ValueError("Selected parts do not contain any contour points.")
        selected_names = set(self._selected_part_names())
        selected_heights = [self.parts[name].height_mm for name in selected_names if name in self.parts]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        max_z = max(selected_heights) if selected_heights else 0.0
        return (min_x, max_x, min_y, max_y, max_z)

    def _safe_slug(self, text: str, fallback: str = "item") -> str:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")
        return slug or fallback

    def _rotate_point_xyz(
        self,
        point_xyz: tuple[float, float, float],
        rotation_deg_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
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

    def _component_definition_payload(self, part_meshes: dict[str, trimesh.Trimesh], exported_paths: dict[str, Path]) -> tuple[dict, Path]:
        component_family = self._safe_slug(self.component_family_var.get().strip(), "generic")
        component_name = self._safe_slug(self.component_name_var.get().strip(), "component")
        combined_mesh = trimesh.util.concatenate(list(part_meshes.values()))
        min_corner = combined_mesh.bounds[0].tolist()
        max_corner = combined_mesh.bounds[1].tolist()
        bbox_center = [
            (min_corner[0] + max_corner[0]) / 2.0,
            (min_corner[1] + max_corner[1]) / 2.0,
            (min_corner[2] + max_corner[2]) / 2.0,
        ]
        rotation_deg_xyz = (
            float(self.component_rotation_x_var.get()),
            float(self.component_rotation_y_var.get()),
            float(self.component_rotation_z_var.get()),
        )
        rotated_vertices = [
            self._rotate_point_xyz(
                (x_coord - bbox_center[0], y_coord - bbox_center[1], z_coord - bbox_center[2]),
                rotation_deg_xyz,
            )
            for x_coord, y_coord, z_coord in combined_mesh.vertices.tolist()
        ]
        rotated_x = [vertex[0] for vertex in rotated_vertices]
        rotated_y = [vertex[1] for vertex in rotated_vertices]
        rotated_z = [vertex[2] for vertex in rotated_vertices]
        bbox_size = [
            max(rotated_x) - min(rotated_x),
            max(rotated_y) - min(rotated_y),
            max(rotated_z) - min(rotated_z),
        ]
        definition_path = DEFAULT_COMPONENT_LIBRARY_DIR / f"{component_family}__{component_name}.json"
        part_payloads = []
        for part_name, output_path in exported_paths.items():
            part = self.parts.get(part_name)
            part_payloads.append(
                {
                    "part_name": part_name,
                    "stl_path": os.path.relpath(output_path, definition_path.parent),
                    "color": part.color if part is not None else DEFAULT_PART_COLOR,
                    "height_mm": part.height_mm if part is not None else DEFAULT_PART_HEIGHT_MM,
                }
            )
        payload = {
            "component_family": component_family,
            "component_name": component_name,
            "source_project_path": str(self.current_project_path) if self.current_project_path is not None else "",
            "native_bbox_mm": {
                "x": bbox_size[0],
                "y": bbox_size[1],
                "z": bbox_size[2],
            },
            "native_center_mm": {
                "x": bbox_center[0],
                "y": bbox_center[1],
                "z": bbox_center[2],
            },
            "native_rotation_deg": {
                "x": rotation_deg_xyz[0],
                "y": rotation_deg_xyz[1],
                "z": rotation_deg_xyz[2],
            },
            "parts": part_payloads,
        }
        return payload, definition_path

    def _sync_part_tree_selection(self, part_name: str | None) -> None:
        target = (part_name or "").strip()
        current_selection = tuple(self.part_tree.selection())
        if target:
            if current_selection == (target,):
                return
            if target not in self.part_tree.get_children():
                return
        elif not current_selection:
            return

        self._syncing_part_selection = True
        if target:
            self.part_tree.selection_set(target)
            self.part_tree.focus(target)
            self.part_tree.see(target)
        else:
            self.part_tree.selection_remove(current_selection)
        self.root.after_idle(self._clear_part_selection_sync_flag)

    def _clear_part_selection_sync_flag(self) -> None:
        self._syncing_part_selection = False

    def _part_line_style(self, part_name: str) -> tuple[int, ...] | None:
        ordered_names = sorted(self.parts.keys(), key=str.lower)
        if part_name not in ordered_names:
            return None
        patterns: list[tuple[int, ...] | None] = [
            None,
            (10, 4),
            (3, 3),
            (12, 3, 3, 3),
            (8, 3, 2, 3, 2, 3),
        ]
        return patterns[ordered_names.index(part_name) % len(patterns)]

    def _contour_centroid_mm(self, contour: ContourRecord) -> tuple[float, float]:
        points = contour.points_mm
        if not points:
            return (0.0, 0.0)
        if contour.closed and len(points) >= 3:
            signed_area_twice = 0.0
            centroid_x = 0.0
            centroid_y = 0.0
            for index, (x1, y1) in enumerate(points):
                x2, y2 = points[(index + 1) % len(points)]
                cross = (x1 * y2) - (x2 * y1)
                signed_area_twice += cross
                centroid_x += (x1 + x2) * cross
                centroid_y += (y1 + y2) * cross
            if abs(signed_area_twice) > 1e-12:
                return (
                    centroid_x / (3.0 * signed_area_twice),
                    centroid_y / (3.0 * signed_area_twice),
                )
        average_x = sum(point[0] for point in points) / len(points)
        average_y = sum(point[1] for point in points) / len(points)
        return (average_x, average_y)

    def _reflect_world_point(self, point_mm: tuple[float, float], axis_name: str) -> tuple[float, float]:
        x_mm, y_mm = point_mm
        if axis_name == "y":
            return (-x_mm, y_mm)
        return (x_mm, -y_mm)

    def _contour_by_id(self, contour_id: str) -> ContourRecord | None:
        for contour in self.contours:
            if contour.contour_id == contour_id:
                return contour
        return None

    def _focused_on_text_input(self) -> bool:
        widget = self.root.focus_get()
        return isinstance(widget, (tk.Entry, ttk.Entry, ttk.Combobox))

    def _current_state_snapshot(self, action_label: str = "") -> dict:
        return _build_state_snapshot(
            self.parts,
            self.contours,
            action_label=action_label,
            selected_contour_id=self.selected_contour_id,
            active_part_name=self.active_part_var.get().strip(),
        )

    def _history_signature(self, snapshot: dict) -> tuple:
        return (
            tuple((part["name"], part["color"], part["height_mm"]) for part in snapshot.get("parts", [])),
            tuple(
                (
                    contour["contour_id"],
                    contour["part_name"],
                    contour["closed"],
                    tuple(tuple(point) for point in contour["points_mm"]),
                )
                for contour in snapshot.get("contours", [])
            ),
        )

    def _record_undo_state(self, action_label: str) -> None:
        snapshot = self._current_state_snapshot(action_label)
        if self.undo_history and self._history_signature(self.undo_history[-1]) == self._history_signature(snapshot):
            self.undo_history[-1]["action_label"] = action_label
            return
        self.undo_history.append(snapshot)
        if len(self.undo_history) > MAX_UNDO_HISTORY:
            self.undo_history = self.undo_history[-MAX_UNDO_HISTORY:]

    def _restore_state_snapshot(self, snapshot: dict) -> None:
        self.parts = _deserialize_parts(snapshot.get("parts", []))
        self.contours = _deserialize_contours(snapshot.get("contours", []))
        self.selected_contour_id = snapshot.get("selected_contour_id")
        self.selected_vertex_refs = {
            vertex_ref
            for vertex_ref in self.selected_vertex_refs
            if self._contour_by_id(vertex_ref[0]) is not None
        }
        active_part_name = str(snapshot.get("active_part_name", "")).strip()
        if active_part_name in self.parts:
            self.active_part_var.set(active_part_name)
        elif self.parts:
            self.active_part_var.set(sorted(self.parts.keys())[0])
        else:
            self.active_part_var.set("")
        self.dragging_vertex_ref = None
        self.group_drag_anchor_px = None
        self.pending_drag_undo_snapshot = None
        self.pending_drag_description = ""
        self.distance_mate_points = [
            vertex_ref for vertex_ref in self.distance_mate_points
            if self._contour_by_id(vertex_ref[0]) is not None
        ]
        self.parallel_line_points = [
            vertex_ref for vertex_ref in self.parallel_line_points
            if self._contour_by_id(vertex_ref[0]) is not None
        ]
        self._refresh_part_tree()
        self._on_active_part_changed()
        self._redraw_canvas()
        self._push_payload()

    def _undo_last_action(self) -> None:
        if not self.undo_history:
            self.status_var.set("No saved actions to undo.")
            return
        snapshot = self.undo_history.pop()
        action_label = str(snapshot.get("action_label", "")).strip() or "last action"
        self._restore_state_snapshot(snapshot)
        self.status_var.set(f"Undid {action_label}.")

    def _begin_pending_drag_undo(self, action_label: str) -> None:
        if self.pending_drag_undo_snapshot is None:
            self.pending_drag_undo_snapshot = self._current_state_snapshot(action_label)
            self.pending_drag_description = action_label

    def _finish_pending_drag_undo(self) -> None:
        if self.pending_drag_undo_snapshot is None:
            return
        if self._history_signature(self.pending_drag_undo_snapshot) != self._history_signature(self._current_state_snapshot()):
            self.undo_history.append(self.pending_drag_undo_snapshot)
            if len(self.undo_history) > MAX_UNDO_HISTORY:
                self.undo_history = self.undo_history[-MAX_UNDO_HISTORY:]
        self.pending_drag_undo_snapshot = None
        self.pending_drag_description = ""

    def _toggle_point_lasso_mode(self) -> None:
        self.point_lasso_enabled = not self.point_lasso_enabled
        self.lasso_start_px = None
        self.lasso_current_px = None
        if self.point_lasso_enabled:
            self.status_var.set("Point lasso is on. Drag a box over visible points to select them.")
        else:
            self.status_var.set("Point lasso is off.")
        self._redraw_canvas()

    def _clear_selected_vertices(self) -> None:
        self.selected_vertex_refs.clear()
        self.group_drag_anchor_px = None
        self.lasso_start_px = None
        self.lasso_current_px = None
        self._redraw_canvas()
        self.status_var.set("Point selection cleared.")

    def _toggle_point_auto_mate(self) -> None:
        enabled = not self.point_mate_enabled_var.get()
        self.point_mate_enabled_var.set(enabled)
        self.point_mate_button_var.set(f"Point Auto-Mate: {'ON' if enabled else 'OFF'}")
        self._redraw_canvas()
        self.status_var.set(
            "Point auto-mate enabled." if enabled else "Point auto-mate disabled."
        )

    def _visible_lasso_candidate_vertex_refs(self) -> list[tuple[str, int]]:
        highlighted_part_name = self._part_tree_highlighted_name()
        candidates: list[tuple[str, int]] = []
        for contour in self.contours:
            if not contour.closed:
                continue
            if highlighted_part_name and contour.part_name != highlighted_part_name:
                continue
            for point_index, _point_mm in enumerate(contour.points_mm):
                candidates.append((contour.contour_id, point_index))
        return candidates

    def _selected_vertex_points_mm(self) -> list[tuple[str, int, tuple[float, float]]]:
        points: list[tuple[str, int, tuple[float, float]]] = []
        for contour_id, point_index in sorted(self.selected_vertex_refs):
            contour = self._contour_by_id(contour_id)
            if contour is None or point_index < 0 or point_index >= len(contour.points_mm):
                continue
            points.append((contour_id, point_index, contour.points_mm[point_index]))
        return points

    def _lasso_selected_vertex_refs(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> set[tuple[str, int]]:
        selected_refs: set[tuple[str, int]] = set()
        for vertex_ref in self._visible_lasso_candidate_vertex_refs():
            point_mm = self._distance_mate_point_mm(vertex_ref)
            if point_mm is None:
                continue
            point_px = self._world_to_canvas(point_mm)
            if min_x <= point_px[0] <= max_x and min_y <= point_px[1] <= max_y:
                selected_refs.add(vertex_ref)
        return selected_refs

    def _move_selected_vertices_by_mm(self, dx_mm: float, dy_mm: float) -> None:
        moved = False
        updated_selection: set[tuple[str, int]] = set()
        for contour_id, point_index, point_mm in self._selected_vertex_points_mm():
            contour = self._contour_by_id(contour_id)
            if contour is None:
                continue
            contour.points_mm[point_index] = (point_mm[0] + dx_mm, point_mm[1] + dy_mm)
            updated_selection.add((contour_id, point_index))
            moved = True
        self.selected_vertex_refs = updated_selection
        if not moved:
            return
        self._redraw_canvas()
        self._push_payload()

    def _on_arrow_nudge(self, event, x_sign: float, y_sign: float) -> str | None:
        if self._focused_on_text_input() or not self.selected_vertex_refs:
            return None
        step_mm = DEFAULT_ARROW_NUDGE_LARGE_MM if (event.state & 0x0001) else DEFAULT_ARROW_NUDGE_MM
        self._record_undo_state("Move selected points")
        self._move_selected_vertices_by_mm(x_sign * step_mm, y_sign * step_mm)
        self.status_var.set(f"Moved {len(self.selected_vertex_refs)} selected point(s) by {step_mm:.3f} mm.")
        return "break"

    def _hotkey_handlers(self) -> dict[str, object]:
        return {
            "create_new_part": self._create_new_part,
            "save_part": self._upsert_part,
            "delete_active_part": self._delete_active_part,
            "undo_point": self._undo_point,
            "undo_last_action": self._undo_last_action,
            "finish_closed_shape": self._finish_closed_shape,
            "finish_open_line": self._finish_open_line,
            "cancel_draft": self._cancel_draft,
            "assign_selected_to_active_part": self._assign_selected_to_active_part,
            "delete_selected_contour": self._delete_selected_contour,
            "center_selected_contour": self._center_selected_contour_to_origin,
            "mirror_selected_x": lambda: self._mirror_selected_contour("x"),
            "mirror_selected_y": lambda: self._mirror_selected_contour("y"),
            "pick_distance_points": self._toggle_distance_mate_pick_mode,
            "apply_distance_mate": self._apply_distance_mate,
            "clear_distance_mate_points": self._clear_distance_mate_points,
            "pick_parallel_lines": self._toggle_parallel_line_pick_mode,
            "apply_parallel_line_distance": self._apply_parallel_line_distance,
            "clear_parallel_line_points": self._clear_parallel_line_points,
            "save_project": self._save_project,
            "load_project": self._prompt_load_project,
            "preview_3d": self._push_payload,
            "export_stls": self._export_stls,
        }

    def _wrap_hotkey_action(self, action):
        def handler(_event=None) -> str | None:
            if self._focused_on_text_input():
                return None
            action()
            return "break"
        return handler

    def _normalize_hotkey_sequence(self, sequence: str) -> str:
        normalized = sequence.strip()
        if not normalized:
            return ""
        if normalized.startswith("<") and normalized.endswith(">"):
            return normalized
        return f"<{normalized}>"

    def _hotkey_display_text(self, sequence: str) -> str:
        normalized = self._normalize_hotkey_sequence(sequence)
        if not normalized:
            return "Unbound"
        return normalized[1:-1]

    def _refresh_hotkey_display_vars(self) -> None:
        for action_id, _label, _default_sequence in HOTKEY_ACTIONS:
            self.hotkey_display_vars[action_id].set(
                self._hotkey_display_text(self.hotkey_vars[action_id].get())
            )

    def _event_to_hotkey_sequence(self, event) -> str:
        keysym = str(getattr(event, "keysym", "") or "").strip()
        if not keysym:
            return ""
        modifier_map = {
            "Shift_L": "Shift",
            "Shift_R": "Shift",
            "Control_L": "Control",
            "Control_R": "Control",
            "Alt_L": "Alt",
            "Alt_R": "Alt",
            "Meta_L": "Alt",
            "Meta_R": "Alt",
        }
        if keysym in modifier_map:
            self.hotkey_capture_modifiers.add(modifier_map[keysym])
            return ""
        modifiers = [name for name in ("Control", "Shift", "Alt") if name in self.hotkey_capture_modifiers]
        normalized_key = "space" if keysym == " " else keysym
        return "<" + "-".join([*modifiers, normalized_key]) + ">"

    def _on_hotkey_capture_release(self, event) -> None:
        keysym = str(getattr(event, "keysym", "") or "").strip()
        modifier_map = {
            "Shift_L": "Shift",
            "Shift_R": "Shift",
            "Control_L": "Control",
            "Control_R": "Control",
            "Alt_L": "Alt",
            "Alt_R": "Alt",
            "Meta_L": "Alt",
            "Meta_R": "Alt",
        }
        modifier = modifier_map.get(keysym)
        if modifier is not None:
            self.hotkey_capture_modifiers.discard(modifier)

    def _save_hotkey_settings(self) -> None:
        DEFAULT_HOTKEY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hotkeys": {
                action_id: self._normalize_hotkey_sequence(self.hotkey_vars[action_id].get())
                for action_id, _label, _default_sequence in HOTKEY_ACTIONS
            }
        }
        DEFAULT_HOTKEY_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_hotkey_settings(self) -> None:
        if not DEFAULT_HOTKEY_SETTINGS_PATH.exists():
            self._refresh_hotkey_display_vars()
            return
        try:
            payload = json.loads(DEFAULT_HOTKEY_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            self._refresh_hotkey_display_vars()
            return
        hotkeys = payload.get("hotkeys", {}) if isinstance(payload, dict) else {}
        if not isinstance(hotkeys, dict):
            self._refresh_hotkey_display_vars()
            return
        known_action_ids = {action_id for action_id, _label, _default_sequence in HOTKEY_ACTIONS}
        for action_id, sequence in hotkeys.items():
            if action_id in known_action_ids and action_id in self.hotkey_vars and isinstance(sequence, str):
                self.hotkey_vars[action_id].set(sequence)
        self._refresh_hotkey_display_vars()

    def _apply_hotkey_bindings(self, announce: bool = True, persist: bool = True) -> bool:
        for sequence in self.hotkey_binding_ids.values():
            self.root.unbind(sequence)
        self.hotkey_binding_ids.clear()

        used_sequences: dict[str, str] = {}
        handlers = self._hotkey_handlers()
        try:
            for action_id, label, _default_sequence in HOTKEY_ACTIONS:
                sequence = self._normalize_hotkey_sequence(self.hotkey_vars[action_id].get())
                if not sequence:
                    continue
                if sequence in used_sequences:
                    raise ValueError(f"{label} duplicates the hotkey for {used_sequences[sequence]}.")
                if action_id not in handlers:
                    continue
                self.root.bind(sequence, self._wrap_hotkey_action(handlers[action_id]))
                self.hotkey_binding_ids[action_id] = sequence
                used_sequences[sequence] = label
        except Exception as exc:
            messagebox.showerror("Hotkeys", str(exc), parent=self.root)
            return False
        self._refresh_hotkey_display_vars()
        if persist:
            try:
                self._save_hotkey_settings()
            except Exception as exc:
                messagebox.showerror("Hotkeys", f"Hotkeys applied but could not be saved: {exc}", parent=self.root)
                return False
        if announce:
            self.status_var.set("Hotkeys updated and saved.")
        return True

    def _open_hotkey_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Hotkey Settings")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("640x760")

        self.hotkey_capture_target = None
        self.hotkey_capture_modifiers.clear()
        self.hotkey_capture_status_var.set("Click Set on any action, then press the key combination you want.")
        self._refresh_hotkey_display_vars()

        outer = ttk.Frame(dialog, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(
            outer,
            textvariable=self.hotkey_capture_status_var,
            wraplength=580,
        ).grid(row=0, column=0, sticky="ew")

        table_canvas = tk.Canvas(outer, highlightthickness=0)
        table_canvas.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=table_canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        table_canvas.configure(yscrollcommand=scrollbar.set)

        table_frame = ttk.Frame(table_canvas, padding=2)
        table_window_id = table_canvas.create_window((0, 0), window=table_frame, anchor="nw")

        def on_frame_configure(_event) -> None:
            table_canvas.configure(scrollregion=table_canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            table_canvas.itemconfigure(table_window_id, width=event.width)

        table_frame.bind("<Configure>", on_frame_configure)
        table_canvas.bind("<Configure>", on_canvas_configure)

        ttk.Label(table_frame, text="Action").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(table_frame, text="Current").grid(row=0, column=1, sticky="w", padx=(0, 8))

        for row_index, (action_id, label, _default_sequence) in enumerate(HOTKEY_ACTIONS, start=1):
            ttk.Label(table_frame, text=label).grid(row=row_index, column=0, sticky="w", pady=(6, 0), padx=(0, 8))
            ttk.Label(table_frame, textvariable=self.hotkey_display_vars[action_id]).grid(
                row=row_index,
                column=1,
                sticky="w",
                pady=(6, 0),
                padx=(0, 8),
            )
            ttk.Button(
                table_frame,
                text="Set",
                command=lambda target=action_id: self._begin_hotkey_capture(target),
            ).grid(row=row_index, column=2, sticky="ew", pady=(6, 0), padx=(0, 6))
            ttk.Button(
                table_frame,
                text="Clear",
                command=lambda target=action_id: self._clear_hotkey_binding(target),
            ).grid(row=row_index, column=3, sticky="ew", pady=(6, 0))

        def handle_key_capture(event) -> str | None:
            if self.hotkey_capture_target is None:
                return None
            sequence = self._event_to_hotkey_sequence(event)
            if not sequence:
                return "break"
            self.hotkey_vars[self.hotkey_capture_target].set(sequence)
            self.hotkey_capture_target = None
            self._refresh_hotkey_display_vars()
            self.hotkey_capture_status_var.set(f"Captured {self._hotkey_display_text(sequence)}.")
            return "break"

        dialog.bind("<KeyPress>", handle_key_capture)
        dialog.bind("<KeyRelease>", lambda event: self._on_hotkey_capture_release(event))
        dialog.focus_force()

        button_row = ttk.Frame(outer)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        def apply_and_close() -> None:
            if self._apply_hotkey_bindings():
                dialog.destroy()

        def reload_saved() -> None:
            self.hotkey_capture_target = None
            self.hotkey_capture_modifiers.clear()
            self._load_hotkey_settings()
            self.hotkey_capture_status_var.set(f"Loaded saved hotkeys from {DEFAULT_HOTKEY_SETTINGS_PATH.name}.")

        ttk.Button(button_row, text="Cancel Capture", command=self._cancel_hotkey_capture).pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Load Saved", command=reload_saved).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(button_row, text="Apply", command=apply_and_close).pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Close", command=dialog.destroy).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _begin_hotkey_capture(self, action_id: str) -> None:
        action_label = next((label for current_id, label, _default in HOTKEY_ACTIONS if current_id == action_id), action_id)
        self.hotkey_capture_target = action_id
        self.hotkey_capture_modifiers.clear()
        self.hotkey_capture_status_var.set(f"Press a key for {action_label}. Use Cancel Capture to stop.")

    def _clear_hotkey_binding(self, action_id: str) -> None:
        self.hotkey_vars[action_id].set("")
        self._refresh_hotkey_display_vars()
        action_label = next((label for current_id, label, _default in HOTKEY_ACTIONS if current_id == action_id), action_id)
        self.hotkey_capture_status_var.set(f"Cleared hotkey for {action_label}.")

    def _cancel_hotkey_capture(self) -> None:
        self.hotkey_capture_target = None
        self.hotkey_capture_modifiers.clear()
        self.hotkey_capture_status_var.set("Hotkey capture cancelled.")

    def _distance_mate_point_mm(self, vertex_ref: tuple[str, int]) -> tuple[float, float] | None:
        contour = self._contour_by_id(vertex_ref[0])
        if contour is None:
            return None
        point_index = vertex_ref[1]
        if point_index < 0 or point_index >= len(contour.points_mm):
            return None
        return contour.points_mm[point_index]

    def _distance_mate_current_distance_mm(self) -> float | None:
        if len(self.distance_mate_points) != 2:
            return None
        first_point = self._distance_mate_point_mm(self.distance_mate_points[0])
        second_point = self._distance_mate_point_mm(self.distance_mate_points[1])
        if first_point is None or second_point is None:
            return None
        return math.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1])

    def _toggle_distance_mate_pick_mode(self) -> None:
        self.parallel_line_pick_active = False
        self.distance_mate_pick_active = not self.distance_mate_pick_active
        if self.distance_mate_pick_active:
            self.status_var.set("Distance mate pick mode is on. Click two closed-shape points.")
        else:
            self.status_var.set("Distance mate pick mode is off.")
        self._redraw_canvas()

    def _clear_distance_mate_points(self) -> None:
        self.distance_mate_points.clear()
        self.distance_mate_pick_active = False
        self._redraw_canvas()
        self.status_var.set("Distance mate picks cleared.")

    def _toggle_parallel_line_pick_mode(self) -> None:
        self.distance_mate_pick_active = False
        self.parallel_line_pick_active = not self.parallel_line_pick_active
        if self.parallel_line_pick_active:
            self.status_var.set("Parallel line pick mode is on. Pick 4 vertices: line 1 then line 2.")
        else:
            self.status_var.set("Parallel line pick mode is off.")
        self._redraw_canvas()

    def _clear_parallel_line_points(self) -> None:
        self.parallel_line_points.clear()
        self.parallel_line_pick_active = False
        self._redraw_canvas()
        self.status_var.set("Parallel line picks cleared.")

    def _register_parallel_line_point(self, vertex_ref: tuple[str, int]) -> None:
        contour = self._contour_by_id(vertex_ref[0])
        if contour is None or not contour.closed:
            self.status_var.set("Parallel line distance works only on vertices from closed contours.")
            return
        if self.parallel_line_points and self.parallel_line_points[-1] == vertex_ref:
            self.status_var.set("That point is already selected for the parallel line tool.")
            return
        if len(self.parallel_line_points) >= 4:
            self.parallel_line_points = self.parallel_line_points[-3:] + [vertex_ref]
        else:
            self.parallel_line_points.append(vertex_ref)
        self.selected_contour_id = contour.contour_id
        self.active_part_var.set(contour.part_name)
        self._on_active_part_changed()
        self._redraw_canvas()
        self.status_var.set(
            f"Parallel line picks: {len(self.parallel_line_points)}/4."
        )

    def _is_contour_edge(self, contour: ContourRecord, first_index: int, second_index: int) -> bool:
        point_count = len(contour.points_mm)
        if point_count < 2:
            return False
        if abs(first_index - second_index) == 1:
            return True
        return contour.closed and {first_index, second_index} == {0, point_count - 1}

    def _parallel_line_segments_mm(
        self,
    ) -> tuple[
        tuple[ContourRecord, tuple[int, int], tuple[tuple[float, float], tuple[float, float]]],
        tuple[ContourRecord, tuple[int, int], tuple[tuple[float, float], tuple[float, float]]],
    ]:
        if len(self.parallel_line_points) != 4:
            raise ValueError("Pick exactly 4 vertices for the parallel line tool.")

        segments = []
        for start in (0, 2):
            first_ref = self.parallel_line_points[start]
            second_ref = self.parallel_line_points[start + 1]
            if first_ref[0] != second_ref[0]:
                raise ValueError("Each picked line must come from a single contour.")
            contour = self._contour_by_id(first_ref[0])
            if contour is None:
                raise ValueError("A picked contour no longer exists.")
            first_index = first_ref[1]
            second_index = second_ref[1]
            if not self._is_contour_edge(contour, first_index, second_index):
                raise ValueError("Each picked line must use two adjacent contour vertices.")
            segments.append(
                (
                    contour,
                    (first_index, second_index),
                    (contour.points_mm[first_index], contour.points_mm[second_index]),
                )
            )
        return segments[0], segments[1]

    def _parallel_line_distance_mm(
        self,
        first_line: tuple[tuple[float, float], tuple[float, float]],
        second_line: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, tuple[float, float], float]:
        (x1, y1), (x2, y2) = first_line
        (x3, y3), (_x4, _y4) = second_line
        dx = x2 - x1
        dy = y2 - y1
        line_length = math.hypot(dx, dy)
        if line_length <= 1e-9:
            raise ValueError("A selected line has zero length.")
        direction = (dx / line_length, dy / line_length)
        normal = (-direction[1], direction[0])
        signed_distance = ((x3 - x1) * normal[0]) + ((y3 - y1) * normal[1])
        return abs(signed_distance), normal, signed_distance

    def _project_point_to_line(
        self,
        point_mm: tuple[float, float],
        line_mm: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float]:
        (x1, y1), (x2, y2) = line_mm
        dx = x2 - x1
        dy = y2 - y1
        line_length_sq = (dx * dx) + (dy * dy)
        if line_length_sq <= 1e-12:
            raise ValueError("Cannot project onto a zero-length line.")
        t_value = (((point_mm[0] - x1) * dx) + ((point_mm[1] - y1) * dy)) / line_length_sq
        return (x1 + (t_value * dx), y1 + (t_value * dy))

    def _register_distance_mate_point(self, vertex_ref: tuple[str, int]) -> None:
        contour = self._contour_by_id(vertex_ref[0])
        if contour is None or not contour.closed:
            self.status_var.set("Distance mate works only on vertices from closed contours.")
            return
        if self.distance_mate_points and self.distance_mate_points[-1] == vertex_ref:
            self.status_var.set("That point is already selected for distance mate.")
            return
        if len(self.distance_mate_points) >= 2:
            self.distance_mate_points = [self.distance_mate_points[-1], vertex_ref]
        else:
            self.distance_mate_points.append(vertex_ref)
        self.selected_contour_id = contour.contour_id
        self.active_part_var.set(contour.part_name)
        self._on_active_part_changed()
        current_distance = self._distance_mate_current_distance_mm()
        if current_distance is None:
            self.status_var.set("Distance mate point 1 selected. Pick one more point.")
        else:
            self.status_var.set(f"Distance mate points selected. Current distance: {current_distance:.4f} mm.")
        self._redraw_canvas()

    def _apply_distance_mate(self) -> None:
        if len(self.distance_mate_points) != 2:
            messagebox.showerror("Distance Mate", "Pick exactly two points for the distance mate.", parent=self.root)
            return
        try:
            target_distance_mm = float(self.distance_mate_distance_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Distance Mate", "Enter a valid target distance in mm.", parent=self.root)
            return
        if target_distance_mm <= 0.0:
            messagebox.showerror("Distance Mate", "Target distance must be greater than 0.", parent=self.root)
            return

        first_ref, second_ref = self.distance_mate_points
        first_point = self._distance_mate_point_mm(first_ref)
        second_point = self._distance_mate_point_mm(second_ref)
        first_contour = self._contour_by_id(first_ref[0])
        second_contour = self._contour_by_id(second_ref[0])
        if first_point is None or second_point is None or first_contour is None or second_contour is None:
            messagebox.showerror("Distance Mate", "The selected points are no longer available.", parent=self.root)
            return

        current_distance_mm = math.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1])
        if current_distance_mm <= 1e-9 and target_distance_mm > 0.0:
            messagebox.showerror("Distance Mate", "Cannot resolve distance from two identical points.", parent=self.root)
            return

        self._record_undo_state("Apply distance mate")
        if not self.distance_mate_has_scaled:
            if current_distance_mm <= 1e-9:
                messagebox.showerror("Distance Mate", "First distance mate needs two distinct points.", parent=self.root)
                return
            scale_factor = target_distance_mm / current_distance_mm
            for contour in self.contours:
                contour.points_mm = [
                    (x_mm * scale_factor, y_mm * scale_factor)
                    for x_mm, y_mm in contour.points_mm
                ]
            for part_name, part in list(self.parts.items()):
                self.parts[part_name] = PartDefinition(
                    name=part.name,
                    color=part.color,
                    height_mm=part.height_mm * scale_factor,
                )
            self.distance_mate_has_scaled = True
            self._refresh_part_tree()
            self._on_active_part_changed()
            self._redraw_canvas()
            self._push_payload()
            self.status_var.set(f"Scaled all geometry by {scale_factor:.6f} from first distance mate.")
            return

        if first_ref[0] == second_ref[0]:
            desired_second_point = (
                first_point[0] + ((second_point[0] - first_point[0]) / current_distance_mm * target_distance_mm),
                first_point[1] + ((second_point[1] - first_point[1]) / current_distance_mm * target_distance_mm),
            )
            second_contour.points_mm[second_ref[1]] = desired_second_point
            self.selected_contour_id = second_contour.contour_id
            self._redraw_canvas()
            self._push_payload()
            self.status_var.set(
                f"Set selected line distance to {target_distance_mm:.4f} mm."
            )
            return

        direction_x = second_point[0] - first_point[0]
        direction_y = second_point[1] - first_point[1]
        if current_distance_mm <= 1e-9:
            direction = (1.0, 0.0)
        else:
            direction = (direction_x / current_distance_mm, direction_y / current_distance_mm)
        desired_second_point = (
            first_point[0] + (direction[0] * target_distance_mm),
            first_point[1] + (direction[1] * target_distance_mm),
        )
        translate_x = desired_second_point[0] - second_point[0]
        translate_y = desired_second_point[1] - second_point[1]
        second_contour.points_mm = [
            (x_mm + translate_x, y_mm + translate_y)
            for x_mm, y_mm in second_contour.points_mm
        ]
        self.selected_contour_id = second_contour.contour_id
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(
            f"Moved contour to set distance mate to {target_distance_mm:.4f} mm."
        )

    def _apply_parallel_line_distance(self) -> None:
        try:
            target_distance_mm = float(self.parallel_line_distance_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Parallel Line Distance", "Enter a valid target distance in mm.", parent=self.root)
            return
        if target_distance_mm < 0.0:
            messagebox.showerror("Parallel Line Distance", "Target distance must be zero or greater.", parent=self.root)
            return

        try:
            first_segment, second_segment = self._parallel_line_segments_mm()
        except ValueError as exc:
            messagebox.showerror("Parallel Line Distance", str(exc), parent=self.root)
            return

        first_contour, _first_indices, first_line = first_segment
        second_contour, second_indices, second_line = second_segment
        first_dx = first_line[1][0] - first_line[0][0]
        first_dy = first_line[1][1] - first_line[0][1]
        second_dx = second_line[1][0] - second_line[0][0]
        second_dy = second_line[1][1] - second_line[0][1]
        second_length = math.hypot(second_dx, second_dy)
        first_length = math.hypot(first_dx, first_dy)
        if first_length <= 1e-9 or second_length <= 1e-9:
            messagebox.showerror("Parallel Line Distance", "Selected lines must have non-zero length.", parent=self.root)
            return
        cross_value = abs((first_dx * second_dy) - (first_dy * second_dx)) / (first_length * second_length)
        if cross_value > PARALLEL_LINE_TOLERANCE:
            messagebox.showerror("Parallel Line Distance", "The selected lines are not parallel.", parent=self.root)
            return

        self._record_undo_state("Apply parallel line distance")
        current_distance_mm, normal, signed_distance = self._parallel_line_distance_mm(first_line, second_line)
        direction_sign = 1.0 if signed_distance >= 0.0 else -1.0
        desired_signed_distance = direction_sign * target_distance_mm
        delta_distance = desired_signed_distance - signed_distance
        translate_x = normal[0] * delta_distance
        translate_y = normal[1] * delta_distance

        first_index, second_index = second_indices
        second_contour.points_mm[first_index] = (
            second_contour.points_mm[first_index][0] + translate_x,
            second_contour.points_mm[first_index][1] + translate_y,
        )
        second_contour.points_mm[second_index] = (
            second_contour.points_mm[second_index][0] + translate_x,
            second_contour.points_mm[second_index][1] + translate_y,
        )
        status_message = f"Shifted selected edge to {target_distance_mm:.4f} mm from the reference edge."

        self.selected_contour_id = second_contour.contour_id
        self.parallel_line_pick_active = False
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(
            status_message + f" Previous distance: {current_distance_mm:.4f} mm."
        )

    def _active_part(self) -> PartDefinition:
        part_name = self.active_part_var.get().strip()
        part = self.parts.get(part_name)
        if part is None:
            raise ValueError("Choose an Active Part from the dropdown, or create one first.")
        return part

    def _choose_part_color(self) -> None:
        selected = colorchooser.askcolor(color=self.part_color_var.get(), parent=self.root)
        if selected and selected[1]:
            self.part_color_var.set(selected[1])

    def _on_active_part_changed(self, _event=None, *, sync_tree: bool = True) -> None:
        part = self.parts.get(self.active_part_var.get())
        if part is None:
            if sync_tree:
                self._sync_part_tree_selection(None)
            self.part_name_var.set("")
            self.part_color_var.set(DEFAULT_PART_COLOR)
            self.part_height_var.set(DEFAULT_PART_HEIGHT_MM)
            return
        if sync_tree:
            self._sync_part_tree_selection(part.name)
        self.part_name_var.set(part.name)
        self.part_color_var.set(part.color)
        self.part_height_var.set(part.height_mm)

    def _on_part_tree_selected(self, _event=None) -> None:
        if self._syncing_part_selection:
            return
        selection = self.part_tree.selection()
        if not selection:
            return
        part_name = selection[0]
        self.active_part_var.set(part_name)
        self._on_active_part_changed(sync_tree=False)

    def _refresh_part_tree(self) -> None:
        previous_selection = tuple(self.part_tree.selection())
        self.active_part_combo["values"] = sorted(self.parts.keys())
        for item_id in self.part_tree.get_children():
            self.part_tree.delete(item_id)
        for part in sorted(self.parts.values(), key=lambda item: item.name.lower()):
            self.part_tree.insert("", "end", iid=part.name, values=(part.name, f"{part.height_mm:.3f}", part.color))
        existing_selection = [item_id for item_id in previous_selection if item_id in self.part_tree.get_children()]
        if existing_selection:
            self.part_tree.selection_set(existing_selection)

    def _next_new_part_name(self) -> str:
        base_name = "New Part"
        if base_name not in self.parts:
            return base_name
        index = 1
        while f"{base_name}({index})" in self.parts:
            index += 1
        return f"{base_name}({index})"

    def _create_new_part(self) -> None:
        self._record_undo_state("Create new part")
        name = self._next_new_part_name()
        self.parts[name] = PartDefinition(
            name=name,
            color=DEFAULT_PART_COLOR,
            height_mm=DEFAULT_PART_HEIGHT_MM,
        )
        self.active_part_var.set(name)
        self.part_name_var.set(name)
        self.part_color_var.set(DEFAULT_PART_COLOR)
        self.part_height_var.set(DEFAULT_PART_HEIGHT_MM)
        self._refresh_part_tree()
        self._on_active_part_changed()
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(f"Created part {name!r}.")

    def _upsert_part(self) -> None:
        name = self.part_name_var.get().strip()
        color = self.part_color_var.get().strip() or DEFAULT_PART_COLOR
        height_mm = float(self.part_height_var.get())
        if not name:
            messagebox.showerror("Missing name", "Please provide a part name.", parent=self.root)
            return
        if height_mm <= 0.0:
            messagebox.showerror("Invalid height", "Extrusion height must be greater than 0.", parent=self.root)
            return

        self._record_undo_state("Save part")
        existing = self.parts.get(self.active_part_var.get().strip())
        self.parts[name] = PartDefinition(name=name, color=color, height_mm=height_mm)
        if existing is not None and existing.name != name:
            for contour in self.contours:
                if contour.part_name == existing.name:
                    contour.part_name = name
            del self.parts[existing.name]

        self.active_part_var.set(name)
        self._refresh_part_tree()
        self._on_active_part_changed()
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(f"Part {name!r} saved.")

    def _delete_active_part(self) -> None:
        part_name = self.active_part_var.get().strip()
        if part_name not in self.parts:
            return
        if any(contour.part_name == part_name for contour in self.contours):
            messagebox.showerror(
                "Part In Use",
                "Delete or reassign contours using this part before removing it.",
                parent=self.root,
            )
            return
        self._record_undo_state("Delete part")
        del self.parts[part_name]
        if self.parts:
            first_part = sorted(self.parts.keys())[0]
            self.active_part_var.set(first_part)
            self._on_active_part_changed()
        else:
            self.active_part_var.set("")
            self._on_active_part_changed()
        self._refresh_part_tree()
        self._push_payload()
        self.status_var.set(f"Part {part_name!r} deleted.")

    def _on_canvas_left_click(self, event) -> None:
        clicked_items = self.canvas.find_overlapping(event.x - 2, event.y - 2, event.x + 2, event.y + 2)
        vertex_ref = next(
            (self.canvas_item_to_vertex[item_id] for item_id in reversed(clicked_items) if item_id in self.canvas_item_to_vertex),
            None,
        )
        if vertex_ref is not None:
            if self.parallel_line_pick_active:
                self._register_parallel_line_point(vertex_ref)
                if len(self.parallel_line_points) >= 4:
                    self.parallel_line_pick_active = False
                return
            if self.distance_mate_pick_active:
                self._register_distance_mate_point(vertex_ref)
                if len(self.distance_mate_points) >= 2:
                    self.distance_mate_pick_active = False
                return
            if vertex_ref not in self.selected_vertex_refs or len(self.selected_vertex_refs) <= 1:
                self.selected_vertex_refs = {vertex_ref}
            if vertex_ref in self.selected_vertex_refs and len(self.selected_vertex_refs) >= 2:
                self._begin_pending_drag_undo("Move selected points")
                self.group_drag_anchor_px = (float(event.x), float(event.y))
                self.selected_contour_id = vertex_ref[0]
                self._redraw_canvas()
                self.status_var.set(f"Dragging {len(self.selected_vertex_refs)} selected points.")
                return
            self._begin_pending_drag_undo("Move point")
            self.dragging_vertex_ref = vertex_ref
            contour_id, _point_index = vertex_ref
            self.selected_contour_id = contour_id
            contour = self._contour_by_id(contour_id)
            if contour is not None:
                self.active_part_var.set(contour.part_name)
                self._on_active_part_changed()
                self.status_var.set("Dragging vertex. Alignment snap is active.")
            self._redraw_canvas()
            return

        if self.point_lasso_enabled:
            self.lasso_start_px = (float(event.x), float(event.y))
            self.lasso_current_px = self.lasso_start_px
            self.group_drag_anchor_px = None
            self._redraw_canvas()
            self.status_var.set("Dragging lasso selection box.")
            return

        contour_id = next(
            (self.canvas_item_to_contour[item_id] for item_id in reversed(clicked_items) if item_id in self.canvas_item_to_contour),
            None,
        )
        if contour_id is not None:
            self.selected_vertex_refs.clear()
            self._handle_canvas_item_click(contour_id)
            return

        if self.draft_part_name is None:
            self.selected_vertex_refs.clear()
            try:
                self.draft_part_name = self._active_part().name
            except ValueError as exc:
                self.status_var.set(str(exc))
                return

        next_point = self._constrained_cursor_point((float(event.x), float(event.y)))
        self.current_points_px.append(next_point)
        self._redraw_canvas()
        point_count = len(self.current_points_px)
        self.status_var.set(
            f"Drafting {self.draft_part_name}. Point count: {point_count}."
            + (" Point mated." if self.mated_target_px is not None else "")
        )

    def _on_canvas_drag_motion(self, event) -> None:
        if self.group_drag_anchor_px is not None:
            last_x, last_y = self.group_drag_anchor_px
            delta_x_px = float(event.x) - last_x
            delta_y_px = float(event.y) - last_y
            self.group_drag_anchor_px = (float(event.x), float(event.y))
            self._move_selected_vertices_by_mm(
                delta_x_px / self.scale_px_per_mm,
                -delta_y_px / self.scale_px_per_mm,
            )
            self.status_var.set(f"Dragging {len(self.selected_vertex_refs)} selected points.")
            return
        if self.lasso_start_px is not None:
            self.lasso_current_px = (float(event.x), float(event.y))
            self._redraw_canvas()
            return
        if self.dragging_vertex_ref is None:
            return
        contour_id, point_index = self.dragging_vertex_ref
        contour = self._contour_by_id(contour_id)
        if contour is None or point_index < 0 or point_index >= len(contour.points_mm):
            return
        snapped_point_px = self._aligned_vertex_drag_point(contour_id, point_index, (float(event.x), float(event.y)))
        contour.points_mm[point_index] = self._canvas_to_world(snapped_point_px)
        self._redraw_canvas()
        self._push_payload()
        snap_summary = []
        if self.alignment_snap_x_px is not None:
            snap_summary.append("vertical alignment")
        if self.alignment_snap_y_px is not None:
            snap_summary.append("horizontal alignment")
        self.status_var.set("Dragging vertex." + (f" Snapped to {' and '.join(snap_summary)}." if snap_summary else ""))

    def _on_canvas_left_release(self, _event) -> None:
        if self.group_drag_anchor_px is not None:
            self.group_drag_anchor_px = None
            self._finish_pending_drag_undo()
            self._redraw_canvas()
            return
        if self.lasso_start_px is not None and self.lasso_current_px is not None:
            min_x = min(self.lasso_start_px[0], self.lasso_current_px[0])
            max_x = max(self.lasso_start_px[0], self.lasso_current_px[0])
            min_y = min(self.lasso_start_px[1], self.lasso_current_px[1])
            max_y = max(self.lasso_start_px[1], self.lasso_current_px[1])
            self.selected_vertex_refs = self._lasso_selected_vertex_refs(min_x, max_x, min_y, max_y)
            self.lasso_start_px = None
            self.lasso_current_px = None
            self._redraw_canvas()
            self.status_var.set(f"Selected {len(self.selected_vertex_refs)} point(s) with lasso.")
            return
        if self.dragging_vertex_ref is None:
            return
        self.dragging_vertex_ref = None
        self.alignment_snap_x_px = None
        self.alignment_snap_y_px = None
        self._finish_pending_drag_undo()
        self._redraw_canvas()

    def _on_canvas_pan_start(self, event) -> None:
        self.is_panning_canvas = True
        self.last_pan_anchor_px = (float(event.x), float(event.y))
        self.status_var.set("Panning sketch view.")

    def _on_canvas_pan_motion(self, event) -> None:
        if not self.is_panning_canvas or self.last_pan_anchor_px is None:
            return
        last_x, last_y = self.last_pan_anchor_px
        delta_x = float(event.x) - last_x
        delta_y = float(event.y) - last_y
        offset_x_px, offset_y_px = self.view_offset_px
        self.view_offset_px = (offset_x_px + delta_x, offset_y_px + delta_y)
        self.last_pan_anchor_px = (float(event.x), float(event.y))
        self._redraw_canvas()

    def _on_canvas_pan_end(self, _event) -> None:
        self.is_panning_canvas = False
        self.last_pan_anchor_px = None

    def _on_canvas_motion(self, event) -> None:
        if self.dragging_vertex_ref is not None:
            return
        if self.is_panning_canvas:
            return
        if not self.current_points_px:
            self.preview_cursor_px = None
            self.mated_target_px = None
            self.current_draft_axis_mode = None
            return
        self.preview_cursor_px = self._constrained_cursor_point((float(event.x), float(event.y)))
        self._redraw_canvas()

    def _on_canvas_zoom(self, event) -> None:
        if getattr(event, "delta", 0):
            zoom_in = event.delta > 0
        elif getattr(event, "num", None) == 4:
            zoom_in = True
        elif getattr(event, "num", None) == 5:
            zoom_in = False
        else:
            return
        factor = 1.1 if zoom_in else (1.0 / 1.1)
        self.scale_px_per_mm = min(300.0, max(5.0, self.scale_px_per_mm * factor))
        self._redraw_canvas()
        self.status_var.set(f"Sketch zoom: {self.scale_px_per_mm:.1f} px/mm")

    def _on_alt_press(self, _event=None) -> None:
        self.alt_pressed = True
        if self.dragging_vertex_ref is not None or self.current_points_px:
            self._redraw_canvas()

    def _on_alt_release(self, _event=None) -> None:
        self.alt_pressed = False
        if self.dragging_vertex_ref is not None or self.current_points_px:
            self._redraw_canvas()

    def _on_undo_shortcut(self, event) -> None:
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self._undo_point()

    def _undo_point(self) -> None:
        if not self.current_points_px:
            return
        self.current_points_px.pop()
        if not self.current_points_px:
            self.draft_part_name = None
            self.preview_cursor_px = None
            self.mated_target_px = None
            self.current_draft_axis_mode = None
        self._redraw_canvas()
        self.status_var.set("Last draft point removed.")

    def _cancel_draft(self) -> None:
        self.current_points_px.clear()
        self.preview_cursor_px = None
        self.mated_target_px = None
        self.current_draft_axis_mode = None
        self.draft_part_name = None
        self._redraw_canvas()
        self.status_var.set("Draft cancelled.")

    def _finish_closed_shape(self) -> None:
        try:
            self._commit_current_contour(closed=True)
            self.status_var.set("Closed contour added.")
        except Exception as exc:
            messagebox.showerror("Cannot Finish Shape", str(exc), parent=self.root)
            self.status_var.set(f"Closed contour failed: {exc}")

    def _finish_open_line(self) -> None:
        try:
            self._commit_current_contour(closed=False)
            self.status_var.set("Open line added. Preview/export will block until it is closed or removed.")
        except Exception as exc:
            messagebox.showerror("Cannot Finish Line", str(exc), parent=self.root)
            self.status_var.set(f"Open line failed: {exc}")

    def _commit_current_contour(self, *, closed: bool) -> None:
        if len(self.current_points_px) < 2:
            raise ValueError("Add at least 2 points before finishing a contour.")
        if closed and len(self.current_points_px) < 3:
            raise ValueError("Closed shapes need at least 3 points.")
        if self.draft_part_name is None:
            self.draft_part_name = self._active_part().name

        self._record_undo_state("Add contour")
        points_mm = [self._canvas_to_world(point_px) for point_px in self.current_points_px]
        contour = ContourRecord(
            contour_id=uuid.uuid4().hex,
            part_name=self.draft_part_name,
            points_mm=points_mm,
            closed=closed,
        )
        self.contours.append(contour)
        self.selected_contour_id = contour.contour_id
        self.current_points_px.clear()
        self.preview_cursor_px = None
        self.mated_target_px = None
        self.current_draft_axis_mode = None
        self.draft_part_name = None
        self._redraw_canvas()
        self._push_payload()

    def _assign_selected_to_active_part(self) -> None:
        contour = self._contour_by_id(self.selected_contour_id or "")
        if contour is None:
            self.status_var.set("Select a contour first.")
            return
        self._record_undo_state("Assign contour part")
        contour.part_name = self._active_part().name
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(f"Contour assigned to {contour.part_name}.")

    def _center_selected_contour_to_origin(self) -> None:
        contour = self._contour_by_id(self.selected_contour_id or "")
        if contour is None:
            self.status_var.set("Select a contour first.")
            return
        self._record_undo_state("Center contour to origin")
        centroid_x, centroid_y = self._contour_centroid_mm(contour)
        contour.points_mm = [
            (x_mm - centroid_x, y_mm - centroid_y)
            for x_mm, y_mm in contour.points_mm
        ]
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set("Selected contour centered to origin.")

    def _center_selected_parts_to_origin(self) -> None:
        try:
            min_x, max_x, min_y, max_y, _max_z = self._selected_parts_bbox_mm()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self._record_undo_state("Center selected parts to origin")
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        selected_names = set(self._selected_part_names())
        for contour in self.contours:
            if contour.part_name not in selected_names:
                continue
            contour.points_mm = [
                (x_mm - center_x, y_mm - center_y)
                for x_mm, y_mm in contour.points_mm
            ]
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(f"Centered {len(selected_names)} selected part(s) to origin.")

    def _scale_selected_parts_to_bbox(self) -> None:
        selected_names = set(self._selected_part_names())
        if not selected_names:
            self.status_var.set("Select one or more parts from the parts list first.")
            return
        try:
            target_x = float(self.scale_bbox_x_var.get())
            target_y = float(self.scale_bbox_y_var.get())
            target_z = float(self.scale_bbox_z_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Scale Selected Parts", "Enter valid X, Y, and Z bounding-box sizes.", parent=self.root)
            return
        if target_x <= 0.0 or target_y <= 0.0 or target_z <= 0.0:
            messagebox.showerror("Scale Selected Parts", "Target X, Y, and Z sizes must all be greater than 0.", parent=self.root)
            return
        try:
            min_x, max_x, min_y, max_y, max_z = self._selected_parts_bbox_mm()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        current_x = max_x - min_x
        current_y = max_y - min_y
        if current_x <= 1e-9 or current_y <= 1e-9 or max_z <= 1e-9:
            messagebox.showerror(
                "Scale Selected Parts",
                "Selected parts must have non-zero X, Y, and Z bounding-box sizes before scaling.",
                parent=self.root,
            )
            return
        self._record_undo_state("Scale selected parts to bounding box")
        scale_x = target_x / current_x
        scale_y = target_y / current_y
        scale_z = target_z / max_z
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        for contour in self.contours:
            if contour.part_name not in selected_names:
                continue
            contour.points_mm = [
                (
                    center_x + ((x_mm - center_x) * scale_x),
                    center_y + ((y_mm - center_y) * scale_y),
                )
                for x_mm, y_mm in contour.points_mm
            ]
        for part_name in selected_names:
            part = self.parts.get(part_name)
            if part is None:
                continue
            self.parts[part_name] = PartDefinition(
                name=part.name,
                color=part.color,
                height_mm=part.height_mm * scale_z,
            )
        self._refresh_part_tree()
        self._on_active_part_changed()
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(
            f"Scaled selected parts to bounding box X={target_x:.3f} Y={target_y:.3f} Z={target_z:.3f} mm."
        )

    def _mirror_selected_contour(self, axis_name: str) -> None:
        contour = self._contour_by_id(self.selected_contour_id or "")
        if contour is None:
            self.status_var.set("Select a contour first.")
            return
        self._record_undo_state(f"Mirror contour across {axis_name.upper()}")
        mirrored_contour = ContourRecord(
            contour_id=uuid.uuid4().hex,
            part_name=contour.part_name,
            points_mm=[self._reflect_world_point(point_mm, axis_name) for point_mm in contour.points_mm],
            closed=contour.closed,
        )
        self.contours.append(mirrored_contour)
        self.selected_contour_id = mirrored_contour.contour_id
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set(f"Selected contour mirrored across {axis_name.upper()} axis.")

    def _delete_selected_contour(self) -> None:
        if self.selected_contour_id is None:
            self.status_var.set("Select a contour to delete.")
            return
        self._record_undo_state("Delete contour")
        self.selected_vertex_refs = {
            vertex_ref for vertex_ref in self.selected_vertex_refs
            if vertex_ref[0] != self.selected_contour_id
        }
        self.distance_mate_points = [
            vertex_ref for vertex_ref in self.distance_mate_points
            if vertex_ref[0] != self.selected_contour_id
        ]
        self.parallel_line_points = [
            vertex_ref for vertex_ref in self.parallel_line_points
            if vertex_ref[0] != self.selected_contour_id
        ]
        self.contours = [contour for contour in self.contours if contour.contour_id != self.selected_contour_id]
        self.selected_contour_id = None
        self.dragging_vertex_ref = None
        self.group_drag_anchor_px = None
        self.alignment_snap_x_px = None
        self.alignment_snap_y_px = None
        self._redraw_canvas()
        self._push_payload()
        self.status_var.set("Selected contour deleted.")

    def _handle_canvas_item_click(self, contour_id: str) -> None:
        self.selected_contour_id = contour_id
        self.dragging_vertex_ref = None
        self.alignment_snap_x_px = None
        self.alignment_snap_y_px = None
        contour = self._contour_by_id(contour_id)
        if contour is not None:
            self.active_part_var.set(contour.part_name)
            self._on_active_part_changed()
            shape_type = "closed shape" if contour.closed else "open line"
            self.status_var.set(f"Selected {shape_type} for part {contour.part_name}.")
        self._redraw_canvas()

    def _aligned_vertex_drag_point(
        self,
        contour_id: str,
        point_index: int,
        raw_point_px: tuple[float, float],
    ) -> tuple[float, float]:
        aligned_x = raw_point_px[0]
        aligned_y = raw_point_px[1]
        self.alignment_snap_x_px = None
        self.alignment_snap_y_px = None
        if self.alt_pressed:
            return (aligned_x, aligned_y)

        candidate_xs: list[float] = []
        candidate_ys: list[float] = []
        for contour in self.contours:
            for candidate_index, point_mm in enumerate(contour.points_mm):
                if contour.contour_id == contour_id and candidate_index == point_index:
                    continue
                candidate_x, candidate_y = self._world_to_canvas(point_mm)
                candidate_xs.append(candidate_x)
                candidate_ys.append(candidate_y)

        best_x = next(
            (
                candidate_x
                for candidate_x in sorted(candidate_xs, key=lambda value: abs(value - raw_point_px[0]))
                if abs(candidate_x - raw_point_px[0]) <= DEFAULT_ALIGNMENT_SNAP_DISTANCE_PX
            ),
            None,
        )
        best_y = next(
            (
                candidate_y
                for candidate_y in sorted(candidate_ys, key=lambda value: abs(value - raw_point_px[1]))
                if abs(candidate_y - raw_point_px[1]) <= DEFAULT_ALIGNMENT_SNAP_DISTANCE_PX
            ),
            None,
        )
        if best_x is not None:
            aligned_x = best_x
            self.alignment_snap_x_px = best_x
        if best_y is not None:
            aligned_y = best_y
            self.alignment_snap_y_px = best_y
        return (aligned_x, aligned_y)

    def _snap_to_mated_point(self, point_px: tuple[float, float]) -> tuple[float, float]:
        self.mated_target_px = None
        if self.alt_pressed or not self.point_mate_enabled_var.get():
            return point_px

        candidate_points: list[tuple[float, float]] = []
        if self.current_points_px:
            candidate_points.extend(self.current_points_px[:-1])
        for contour in self.contours:
            for point_mm in contour.points_mm:
                candidate_points.append(self._world_to_canvas(point_mm))

        if not candidate_points:
            return point_px

        best_point: tuple[float, float] | None = None
        best_distance = float("inf")
        for candidate_x, candidate_y in candidate_points:
            distance = math.hypot(point_px[0] - candidate_x, point_px[1] - candidate_y)
            if distance < best_distance:
                best_distance = distance
                best_point = (candidate_x, candidate_y)

        if best_point is None or best_distance > DEFAULT_MATE_SNAP_DISTANCE_PX:
            return point_px

        self.mated_target_px = best_point
        return best_point

    def _constrained_cursor_point(self, raw_point_px: tuple[float, float]) -> tuple[float, float]:
        if not self.current_points_px:
            self.mated_target_px = None
            self.current_draft_axis_mode = None
            return raw_point_px
        if not self.angle_constraint_enabled_var.get():
            if self.axis_snap_enabled_var.get():
                last_point = self.current_points_px[-1]
                dx = raw_point_px[0] - last_point[0]
                dy = raw_point_px[1] - last_point[1]
                if abs(dx) >= abs(dy):
                    self.current_draft_axis_mode = "horizontal"
                    return self._snap_to_mated_point((raw_point_px[0], last_point[1]))
                self.current_draft_axis_mode = "vertical"
                return self._snap_to_mated_point((last_point[0], raw_point_px[1]))
            self.current_draft_axis_mode = None
            return self._snap_to_mated_point(raw_point_px)
        if len(self.current_points_px) < 2:
            self.current_draft_axis_mode = None
            return self._snap_to_mated_point(raw_point_px)

        last_point = self.current_points_px[-1]
        prev_point = self.current_points_px[-2]
        base_dx = last_point[0] - prev_point[0]
        base_dy = last_point[1] - prev_point[1]
        base_length = math.hypot(base_dx, base_dy)
        if base_length <= 1e-9:
            self.current_draft_axis_mode = None
            return self._snap_to_mated_point(raw_point_px)

        raw_dx = raw_point_px[0] - last_point[0]
        raw_dy = raw_point_px[1] - last_point[1]
        raw_length = math.hypot(raw_dx, raw_dy)
        if raw_length <= 1e-9:
            self.current_draft_axis_mode = None
            return self._snap_to_mated_point(raw_point_px)

        base_angle = math.atan2(base_dy, base_dx)
        try:
            delta_angle = math.radians(float(self.angle_constraint_deg_var.get()))
        except (tk.TclError, ValueError):
            self.current_draft_axis_mode = None
            return self._snap_to_mated_point(raw_point_px)
        candidate_angles = (base_angle + delta_angle, base_angle - delta_angle)
        raw_angle = math.atan2(raw_dy, raw_dx)
        chosen_angle = min(
            candidate_angles,
            key=lambda candidate: abs(math.atan2(math.sin(raw_angle - candidate), math.cos(raw_angle - candidate))),
        )
        self.current_draft_axis_mode = None
        return self._snap_to_mated_point((
            last_point[0] + (raw_length * math.cos(chosen_angle)),
            last_point[1] + (raw_length * math.sin(chosen_angle)),
        ))

    def _draw_saved_contours(self) -> None:
        self.canvas_item_to_contour.clear()
        self.canvas_item_to_vertex.clear()
        highlighted_part_name = self._part_tree_highlighted_name()
        distance_mate_lookup = {
            vertex_ref: index
            for index, vertex_ref in enumerate(self.distance_mate_points, start=1)
        }
        parallel_line_lookup = {
            vertex_ref: label
            for vertex_ref, label in zip(self.parallel_line_points, ("A", "B", "C", "D"))
        }
        for contour in self.contours:
            part = self.parts.get(contour.part_name, PartDefinition(contour.part_name, DEFAULT_PART_COLOR, DEFAULT_PART_HEIGHT_MM))
            points_px: list[float] = []
            point_pairs_px: list[tuple[float, float]] = []
            for point_mm in contour.points_mm:
                x_px, y_px = self._world_to_canvas(point_mm)
                points_px.extend([x_px, y_px])
                point_pairs_px.append((x_px, y_px))
            is_emphasized = contour.contour_id == self.selected_contour_id or highlighted_part_name == contour.part_name
            outline_color = "#0e0e0e" if contour.contour_id == self.selected_contour_id else (part.color if is_emphasized else "#9a9388")
            line_color = part.color if is_emphasized else "#a7a093"
            width = 3 if contour.contour_id == self.selected_contour_id else 2
            if contour.closed:
                item_id = self.canvas.create_polygon(
                    *points_px,
                    fill=part.color if is_emphasized else "#d8d1c4",
                    stipple="gray25",
                    outline=outline_color,
                    width=width,
                )
            else:
                item_id = self.canvas.create_line(
                    *points_px,
                    fill=line_color,
                    width=width,
                    smooth=False,
                )
            self.canvas_item_to_contour[item_id] = contour.contour_id
            show_handles = highlighted_part_name == contour.part_name or any(
                contour.contour_id == selected_contour_id for selected_contour_id, _point_index in self.selected_vertex_refs
            )
            if not show_handles:
                continue
            for point_index, (x_px, y_px) in enumerate(point_pairs_px):
                is_selected_vertex = (contour.contour_id, point_index) in self.selected_vertex_refs
                radius = 5 if contour.contour_id == self.selected_contour_id else 4
                if is_selected_vertex:
                    radius = max(radius, 6)
                fill_color = "#8ec5ff" if is_selected_vertex else ("#ffffff" if contour.closed else "#f3dfc4")
                handle_id = self.canvas.create_oval(
                    x_px - radius,
                    y_px - radius,
                    x_px + radius,
                    y_px + radius,
                    fill=fill_color,
                    outline="#124c8a" if is_selected_vertex else outline_color,
                    width=2 if contour.contour_id == self.selected_contour_id or is_selected_vertex else 1,
                )
                if contour.closed:
                    self.canvas_item_to_vertex[handle_id] = (contour.contour_id, point_index)
                mate_label = distance_mate_lookup.get((contour.contour_id, point_index))
                if mate_label is not None:
                    self.canvas.create_oval(
                        x_px - 9,
                        y_px - 9,
                        x_px + 9,
                        y_px + 9,
                        outline="#111111",
                        width=2,
                    )
                    self.canvas.create_text(
                        x_px + 10,
                        y_px - 10,
                        text=str(mate_label),
                        fill="#111111",
                        anchor="sw",
                    )
                parallel_label = parallel_line_lookup.get((contour.contour_id, point_index))
                if parallel_label is not None:
                    self.canvas.create_rectangle(
                        x_px - 10,
                        y_px - 10,
                        x_px + 10,
                        y_px + 10,
                        outline="#8b1e1e",
                        width=2,
                    )
                    self.canvas.create_text(
                        x_px + 12,
                        y_px + 12,
                        text=parallel_label,
                        fill="#8b1e1e",
                        anchor="nw",
                    )
        if len(self.distance_mate_points) == 2:
            first_point = self._distance_mate_point_mm(self.distance_mate_points[0])
            second_point = self._distance_mate_point_mm(self.distance_mate_points[1])
            if first_point is not None and second_point is not None:
                first_x, first_y = self._world_to_canvas(first_point)
                second_x, second_y = self._world_to_canvas(second_point)
                self.canvas.create_line(
                    first_x,
                    first_y,
                    second_x,
                    second_y,
                    fill="#111111",
                    dash=(5, 4),
                    width=2,
                )
                midpoint_x = (first_x + second_x) / 2.0
                midpoint_y = (first_y + second_y) / 2.0
                current_distance = math.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1])
                self.canvas.create_text(
                    midpoint_x + 10,
                    midpoint_y - 10,
                    text=f"{current_distance:.4f} mm",
                    fill="#111111",
                    anchor="sw",
                )
        if len(self.parallel_line_points) >= 2:
            self._draw_parallel_line_preview()
        if self.lasso_start_px is not None and self.lasso_current_px is not None:
            self.canvas.create_rectangle(
                self.lasso_start_px[0],
                self.lasso_start_px[1],
                self.lasso_current_px[0],
                self.lasso_current_px[1],
                outline="#124c8a",
                dash=(4, 4),
                width=2,
            )

    def _draw_parallel_line_preview(self) -> None:
        colors = ("#8b1e1e", "#8b1e1e")
        for start_index, color in zip((0, 2), colors):
            if len(self.parallel_line_points) <= start_index + 1:
                continue
            first_point = self._distance_mate_point_mm(self.parallel_line_points[start_index])
            second_point = self._distance_mate_point_mm(self.parallel_line_points[start_index + 1])
            if first_point is None or second_point is None:
                continue
            first_x, first_y = self._world_to_canvas(first_point)
            second_x, second_y = self._world_to_canvas(second_point)
            self.canvas.create_line(
                first_x,
                first_y,
                second_x,
                second_y,
                fill=color,
                dash=(3, 3),
                width=2,
            )
        if len(self.parallel_line_points) != 4:
            return
        try:
            first_segment, second_segment = self._parallel_line_segments_mm()
            current_distance, _normal, _signed_distance = self._parallel_line_distance_mm(first_segment[2], second_segment[2])
            second_anchor = second_segment[2][0]
            projected_anchor = self._project_point_to_line(second_anchor, first_segment[2])
        except ValueError:
            return
        first_x, first_y = self._world_to_canvas(projected_anchor)
        second_x, second_y = self._world_to_canvas(second_anchor)
        self.canvas.create_line(
            first_x,
            first_y,
            second_x,
            second_y,
            fill="#8b1e1e",
            dash=(6, 4),
            width=1,
        )
        self.canvas.create_text(
            ((first_x + second_x) / 2.0) + 10,
            ((first_y + second_y) / 2.0) - 10,
            text=f"{current_distance:.4f} mm",
            fill="#8b1e1e",
            anchor="sw",
        )

    def _draw_draft(self) -> None:
        if not self.current_points_px:
            return
        part_name = self.draft_part_name
        if not part_name:
            return
        part = self.parts.get(part_name, PartDefinition(part_name, DEFAULT_PART_COLOR, DEFAULT_PART_HEIGHT_MM))
        dash_pattern = self._part_line_style(part_name)

        flat_points = [coordinate for point in self.current_points_px for coordinate in point]
        if len(flat_points) >= 4:
            self.canvas.create_line(*flat_points, fill=part.color, width=2, dash=dash_pattern or (6, 4))
        for x_px, y_px in self.current_points_px:
            self.canvas.create_oval(x_px - 3, y_px - 3, x_px + 3, y_px + 3, fill="#1d1d1d", outline="")
        if self.mated_target_px is not None:
            mate_x, mate_y = self.mated_target_px
            self.canvas.create_oval(
                mate_x - 6,
                mate_y - 6,
                mate_x + 6,
                mate_y + 6,
                outline="#0e0e0e",
                width=2,
            )
        if self.preview_cursor_px is not None:
            last_x, last_y = self.current_points_px[-1]
            cursor_x, cursor_y = self.preview_cursor_px
            self.preview_line_id = self.canvas.create_line(
                last_x,
                last_y,
                cursor_x,
                cursor_y,
                fill=part.color,
                width=2,
                dash=dash_pattern or (2, 3),
            )
        axis_mode = self.current_draft_axis_mode
        if axis_mode in {"vertical", "horizontal"} and self.current_points_px:
            last_x, last_y = self.current_points_px[-1]
            self.canvas.create_text(
                last_x + 12,
                last_y + 12,
                text=f"{axis_mode} snap",
                fill="#2d241f",
                anchor="nw",
            )
        if self.angle_constraint_enabled_var.get() and len(self.current_points_px) >= 2:
            last_x, last_y = self.current_points_px[-1]
            self.canvas.create_text(
                last_x + 12,
                last_y - 12,
                text=f"{self.angle_constraint_deg_var.get():.1f} deg",
                fill="#2d241f",
                anchor="sw",
            )
        if self.mated_target_px is not None:
            mate_x, mate_y = self.mated_target_px
            self.canvas.create_text(
                mate_x + 10,
                mate_y - 10,
                text="mate",
                fill="#2d241f",
                anchor="sw",
            )

    def _draw_alignment_guides(self) -> None:
        if self.alignment_snap_x_px is not None:
            self.canvas.create_line(
                self.alignment_snap_x_px,
                0,
                self.alignment_snap_x_px,
                DEFAULT_CANVAS_HEIGHT,
                fill="#7a6a58",
                dash=(4, 4),
                width=1,
            )
        if self.alignment_snap_y_px is not None:
            self.canvas.create_line(
                0,
                self.alignment_snap_y_px,
                DEFAULT_CANVAS_WIDTH,
                self.alignment_snap_y_px,
                fill="#7a6a58",
                dash=(4, 4),
                width=1,
            )

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_alignment_guides()
        self._draw_saved_contours()
        self._draw_draft()

    def _project_payload(self, status_message: str = "") -> dict:
        return build_payload(
            self.parts,
            self.contours,
            status_message=status_message,
            history=self.undo_history,
            component_export={
                "family": self.component_family_var.get().strip(),
                "name": self.component_name_var.get().strip(),
                "rotation_x_deg": float(self.component_rotation_x_var.get()),
                "rotation_y_deg": float(self.component_rotation_y_var.get()),
                "rotation_z_deg": float(self.component_rotation_z_var.get()),
            },
        )

    def _push_payload(self) -> None:
        try:
            write_bridge_payload(self.bridge_path, self._project_payload(self.status_var.get()))
        except Exception as exc:
            self.status_var.set(f"Viewer sync failed: {exc}")

    def _save_project(self) -> None:
        project_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save part generator project",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            initialfile=DEFAULT_PROJECT_PATH.name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not project_path:
            return
        payload = self._project_payload("Project saved.")
        Path(project_path).parent.mkdir(parents=True, exist_ok=True)
        Path(project_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.current_project_path = Path(project_path).expanduser().resolve()
        self.status_var.set(f"Project saved to {project_path}.")
        self._push_payload()

    def _load_project(self, project_path: Path) -> None:
        payload = json.loads(project_path.expanduser().resolve().read_text(encoding="utf-8"))
        parts = _deserialize_parts(payload.get("parts", []))
        contours = _deserialize_contours(payload.get("contours", []))
        if not parts:
            raise ValueError("Project file does not contain any parts.")
        self.parts = parts
        self.contours = contours
        self.current_project_path = project_path.expanduser().resolve()
        raw_history = payload.get("history", [])
        self.undo_history = [item for item in raw_history if isinstance(item, dict)][-MAX_UNDO_HISTORY:]
        component_export = payload.get("component_export", {})
        if isinstance(component_export, dict):
            self.component_family_var.set(str(component_export.get("family", self.component_family_var.get())).strip() or "generic")
            self.component_name_var.set(str(component_export.get("name", self.component_name_var.get())).strip() or "generic_component")
            self.component_rotation_x_var.set(float(component_export.get("rotation_x_deg", self.component_rotation_x_var.get())))
            self.component_rotation_y_var.set(float(component_export.get("rotation_y_deg", self.component_rotation_y_var.get())))
            self.component_rotation_z_var.set(float(component_export.get("rotation_z_deg", self.component_rotation_z_var.get())))
        self.selected_contour_id = None
        self.selected_vertex_refs.clear()
        self.distance_mate_points.clear()
        self.distance_mate_pick_active = False
        self.distance_mate_has_scaled = False
        self.parallel_line_points.clear()
        self.parallel_line_pick_active = False
        self.point_lasso_enabled = False
        self.lasso_start_px = None
        self.lasso_current_px = None
        self.group_drag_anchor_px = None
        self.pending_drag_undo_snapshot = None
        self.pending_drag_description = ""
        self.current_points_px.clear()
        first_part = sorted(self.parts.keys())[0]
        self.active_part_var.set(first_part)
        self._on_active_part_changed()
        self._refresh_part_tree()
        self._redraw_canvas()
        self.status_var.set(f"Loaded project from {project_path}.")
        self._push_payload()

    def _prompt_load_project(self) -> None:
        project_path = filedialog.askopenfilename(
            parent=self.root,
            title="Load part generator project",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not project_path:
            return
        try:
            self._load_project(Path(project_path))
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc), parent=self.root)
            self.status_var.set(f"Load failed: {exc}")

    def _start_viewer(self) -> None:
        try:
            write_bridge_payload(self.bridge_path, self._project_payload("Launching viewer..."))
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
            self.status_var.set("Viewer launch failed.")

    def _export_stls(self) -> None:
        try:
            part_meshes = build_part_meshes(self.parts, self.contours)
        except Exception as exc:
            messagebox.showerror("Export Blocked", str(exc), parent=self.root)
            self.status_var.set(f"Export blocked: {exc}")
            return

        DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_COMPONENT_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        exported_paths: dict[str, Path] = {}
        for part_name, mesh in part_meshes.items():
            safe_name = "".join(char if char.isalnum() else "_" for char in part_name).strip("_").lower() or "part"
            output_path = DEFAULT_EXPORT_DIR / f"{safe_name}.stl"
            mesh.export(output_path)
            exported_paths[part_name] = output_path

        definition_payload, definition_path = self._component_definition_payload(part_meshes, exported_paths)
        definition_path.write_text(json.dumps(definition_payload, indent=2), encoding="utf-8")

        self.status_var.set(
            "Exported component assets: "
            + ", ".join(path.name for path in exported_paths.values())
            + f" | Library: {definition_path.name}"
        )
        self._push_payload()

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
    parser = argparse.ArgumentParser(description="2D PCB part sketcher with simple 3D extrusion preview.")
    parser.add_argument("--project", type=Path, default=None, help="Optional project JSON file to load on startup.")
    parser.add_argument("--viewer-bridge", type=Path, default=None, help="Internal: viewer bridge JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.viewer_bridge is not None:
        viewer = PartExtrusionViewer(args.viewer_bridge.expanduser().resolve())
        viewer.run()
        return 0

    panel = PartGeneratorControlPanel(args)
    panel.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
