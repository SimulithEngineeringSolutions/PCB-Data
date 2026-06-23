from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import trimesh
from vedo import Mesh, Plotter, Text2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIN_PATH = REPO_ROOT / "output" / "component_maker" / "ConnectorPin" / "connector Pin.stl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "component_maker" / "FemalePinConnector1x1"
BRIDGE_DIR = DEFAULT_OUTPUT_DIR / "viewer_bridge"
PIN_ENLARGE_MM = 0.001


@dataclass(slots=True)
class HolderSpec:
    outer_square_side_length_mm: float = 3.3
    inner_square_tube_thickness_mm: float = 0.4
    z_start_mm: float = -1.27
    z_end_mm: float = 1.27


DEFAULT_SPEC = HolderSpec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 1x1 female pin connector housing by subtracting an enlarged pin from a square plastic solid."
    )
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN_PATH, help="Path to the internal copper pin STL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the plastic holder, aligned pin, and assembly STLs are exported.",
    )
    parser.add_argument("--outer-square-side-length-mm", type=float, default=DEFAULT_SPEC.outer_square_side_length_mm)
    parser.add_argument("--inner-square-tube-thickness-mm", type=float, default=DEFAULT_SPEC.inner_square_tube_thickness_mm)
    parser.add_argument("--z-start-mm", type=float, default=DEFAULT_SPEC.z_start_mm)
    parser.add_argument("--z-end-mm", type=float, default=DEFAULT_SPEC.z_end_mm)
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run without the UI and export immediately using the provided arguments.",
    )
    parser.add_argument("--viewer-bridge", type=Path, default=None, help="Internal: viewer bridge JSON file.")
    return parser.parse_args()


def load_pin_mesh(pin_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(pin_path.expanduser().resolve(), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh STL at {pin_path}, got {type(mesh)!r}.")
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)

    if float(max(mesh.extents)) < 1.0:
        mesh.apply_scale(1000.0)

    return mesh


def _rotation_matrix_for_axis_alignment(from_axis: int, to_axis: int) -> list[list[float]] | None:
    if from_axis == to_axis:
        return None
    angle = math.pi / 2.0
    if from_axis == 0 and to_axis == 2:
        return trimesh.transformations.rotation_matrix(-angle, [0.0, 1.0, 0.0])
    if from_axis == 2 and to_axis == 0:
        return trimesh.transformations.rotation_matrix(angle, [0.0, 1.0, 0.0])
    if from_axis == 1 and to_axis == 2:
        return trimesh.transformations.rotation_matrix(angle, [1.0, 0.0, 0.0])
    if from_axis == 2 and to_axis == 1:
        return trimesh.transformations.rotation_matrix(-angle, [1.0, 0.0, 0.0])
    if from_axis == 0 and to_axis == 1:
        return trimesh.transformations.rotation_matrix(angle, [0.0, 0.0, 1.0])
    if from_axis == 1 and to_axis == 0:
        return trimesh.transformations.rotation_matrix(-angle, [0.0, 0.0, 1.0])
    return None


def normalize_pin_mesh(pin_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    normalized = trimesh.Trimesh(vertices=pin_mesh.vertices.copy(), faces=pin_mesh.faces.copy(), process=False)
    normalized.apply_translation(-normalized.centroid)
    longest_axis = int(normalized.extents.argmax())
    rotation = _rotation_matrix_for_axis_alignment(longest_axis, 2)
    if rotation is not None:
        normalized.apply_transform(rotation)
    return normalized


def validate_spec(spec: HolderSpec) -> None:
    if spec.outer_square_side_length_mm <= 0.0:
        raise ValueError("outer_square_side_length_mm must be positive.")
    if spec.inner_square_tube_thickness_mm <= 0.0:
        raise ValueError("inner_square_tube_thickness_mm must be positive.")
    if spec.z_end_mm <= spec.z_start_mm:
        raise ValueError("z_end_mm must be greater than z_start_mm.")
    inner_side_mm = spec.outer_square_side_length_mm - (2.0 * spec.inner_square_tube_thickness_mm)
    if inner_side_mm <= 0.0:
        raise ValueError("outer_square_side_length_mm must be greater than twice the inner square tube thickness.")


def build_outer_plastic_block(
    outer_square_side_length_mm: float,
    z_start_mm: float,
    z_end_mm: float,
) -> trimesh.Trimesh:
    z_length_mm = z_end_mm - z_start_mm
    z_center_mm = (z_start_mm + z_end_mm) / 2.0
    block = trimesh.creation.box(
        extents=(outer_square_side_length_mm, outer_square_side_length_mm, z_length_mm)
    )
    block.apply_translation((0.0, 0.0, z_center_mm))
    return trimesh.Trimesh(vertices=block.vertices.copy(), faces=block.faces.copy(), process=False)


def build_inner_void_block(
    outer_square_side_length_mm: float,
    inner_square_tube_thickness_mm: float,
    z_start_mm: float,
    z_end_mm: float,
) -> trimesh.Trimesh:
    z_length_mm = z_end_mm - z_start_mm
    z_center_mm = (z_start_mm + z_end_mm) / 2.0
    inner_side_mm = outer_square_side_length_mm - (2.0 * inner_square_tube_thickness_mm)
    inner_block = trimesh.creation.box(
        extents=(inner_side_mm, inner_side_mm, z_length_mm)
    )
    inner_block.apply_translation((0.0, 0.0, z_center_mm))
    return trimesh.Trimesh(vertices=inner_block.vertices.copy(), faces=inner_block.faces.copy(), process=False)


def enlarge_pin_uniformly(pin_mesh: trimesh.Trimesh, offset_mm: float) -> trimesh.Trimesh:
    enlarged = trimesh.Trimesh(vertices=pin_mesh.vertices.copy(), faces=pin_mesh.faces.copy(), process=False)
    scale = 1.0 + (2.0 * offset_mm / max(float(enlarged.extents.max()), 1e-9))
    enlarged.apply_scale(scale)
    return enlarged


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    pieces = mesh.split(only_watertight=False)
    if not pieces:
        raise ValueError("Boolean subtraction produced no plastic mesh.")
    largest = max(pieces, key=lambda piece: float(piece.volume) if piece.is_volume else piece.area)
    return trimesh.Trimesh(vertices=largest.vertices.copy(), faces=largest.faces.copy(), process=False)


def build_plastic_holder(spec: HolderSpec, normalized_pin: trimesh.Trimesh) -> trimesh.Trimesh:
    outer_block = build_outer_plastic_block(
        outer_square_side_length_mm=spec.outer_square_side_length_mm,
        z_start_mm=spec.z_start_mm,
        z_end_mm=spec.z_end_mm,
    )
    inner_void = build_inner_void_block(
        outer_square_side_length_mm=spec.outer_square_side_length_mm,
        inner_square_tube_thickness_mm=spec.inner_square_tube_thickness_mm,
        z_start_mm=spec.z_start_mm,
        z_end_mm=spec.z_end_mm,
    )
    tube = trimesh.boolean.difference([outer_block, inner_void], engine="manifold")
    if tube is None:
        raise ValueError("Boolean subtraction failed while creating the square tube.")
    if not isinstance(tube, trimesh.Trimesh):
        if hasattr(tube, "dump"):
            dumped = tube.dump(concatenate=True)
            tube = dumped if isinstance(dumped, trimesh.Trimesh) else trimesh.util.concatenate(dumped)
        else:
            tube = trimesh.util.concatenate(list(tube.geometry.values()))
    tube = trimesh.Trimesh(vertices=tube.vertices.copy(), faces=tube.faces.copy(), process=False)

    enlarged_pin = enlarge_pin_uniformly(normalized_pin, PIN_ENLARGE_MM)
    plastic = trimesh.boolean.difference([tube, enlarged_pin], engine="manifold")
    if plastic is None:
        raise ValueError("Boolean subtraction failed while creating the plastic holder.")
    if not isinstance(plastic, trimesh.Trimesh):
        if hasattr(plastic, "dump"):
            dumped = plastic.dump(concatenate=True)
            plastic = dumped if isinstance(dumped, trimesh.Trimesh) else trimesh.util.concatenate(dumped)
        else:
            plastic = trimesh.util.concatenate(list(plastic.geometry.values()))
    return keep_largest_component(
        trimesh.Trimesh(vertices=plastic.vertices.copy(), faces=plastic.faces.copy(), process=False)
    )


def export_connector(spec: HolderSpec, pin_path: Path, output_dir: Path) -> dict[str, Path]:
    meshes = build_connector_meshes(spec, pin_path)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    holder_path = output_dir / "female_pin_holder.stl"
    pin_out_path = output_dir / "female_pin_internal_contact.stl"
    assembly_path = output_dir / "female_pin_connector_1x1_assembly.stl"

    meshes["holder"].export(holder_path)
    meshes["pin"].export(pin_out_path)
    meshes["assembly"].export(assembly_path)

    return {
        "holder": holder_path,
        "pin": pin_out_path,
        "assembly": assembly_path,
    }


def build_connector_meshes(spec: HolderSpec, pin_path: Path) -> dict[str, trimesh.Trimesh]:
    pin_mesh = load_pin_mesh(pin_path)
    normalized_pin = normalize_pin_mesh(pin_mesh)
    validate_spec(spec)
    holder = build_plastic_holder(spec, normalized_pin)
    assembly = trimesh.util.concatenate([holder, normalized_pin])
    return {
        "holder": trimesh.Trimesh(vertices=holder.vertices.copy(), faces=holder.faces.copy(), process=False),
        "pin": trimesh.Trimesh(vertices=normalized_pin.vertices.copy(), faces=normalized_pin.faces.copy(), process=False),
        "assembly": trimesh.Trimesh(vertices=assembly.vertices.copy(), faces=assembly.faces.copy(), process=False),
    }


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
    spec: HolderSpec,
    pin_path: Path,
    status_message: str,
    summary_message: str,
) -> dict:
    return {
        "pin_path": str(pin_path),
        "spec": {
            "outer_square_side_length_mm": spec.outer_square_side_length_mm,
            "inner_square_tube_thickness_mm": spec.inner_square_tube_thickness_mm,
            "z_start_mm": spec.z_start_mm,
            "z_end_mm": spec.z_end_mm,
        },
        "status_message": status_message,
        "summary_message": summary_message,
    }


class FemalePinConnectorViewer:
    def __init__(self, bridge_path: Path) -> None:
        self.bridge_path = bridge_path
        self.plotter = Plotter(
            title="Female Pin Connector 1x1 Preview",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1100, 760),
        )
        self.info = Text2D("", pos="top-left", s=0.75, c="#2d241f", bg=None, font="Courier")
        self.actors: list = []
        self.last_signature: tuple | None = None

    def _payload_signature(self, payload: dict) -> tuple:
        spec = payload.get("spec", {})
        return (
            payload.get("pin_path", ""),
            tuple(sorted(spec.items())) if isinstance(spec, dict) else (),
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
            spec = HolderSpec(**payload["spec"])
            pin_path = Path(str(payload["pin_path"])).expanduser().resolve()
            meshes = build_connector_meshes(spec, pin_path)
            holder_actor = self._mesh_actor(meshes["holder"], "#202020", 0.55)
            pin_actor = self._mesh_actor(meshes["pin"], "#d08a35", 1.0)
            self.actors.extend([holder_actor, pin_actor])
            self.plotter += holder_actor
            self.plotter += pin_actor
            preview_line = "Preview: boolean-subtracted plastic + internal contact"
        except Exception as exc:
            preview_line = f"Preview blocked: {exc}"

        self.info.text(
            "Female Pin Connector 1x1\n"
            f"{preview_line}\n"
            "Origin: connector pin centroid\n"
            f"{status_message or 'Edit values in the control panel.'}\n"
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


def build_spec_from_args(args: argparse.Namespace) -> HolderSpec:
    return HolderSpec(
        outer_square_side_length_mm=args.outer_square_side_length_mm,
        inner_square_tube_thickness_mm=args.inner_square_tube_thickness_mm,
        z_start_mm=args.z_start_mm,
        z_end_mm=args.z_end_mm,
    )


class FemalePinConnectorControlPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pin_path = args.pin.expanduser().resolve()
        self.output_dir = args.output_dir.expanduser().resolve()
        self.pin_mesh = load_pin_mesh(self.pin_path)

        self.root = tk.Tk()
        self.root.title("Female Pin Connector 1x1")
        self.root.geometry("640x530")
        self.root.minsize(600, 500)

        self.outer_square_side_length_var = tk.DoubleVar(value=args.outer_square_side_length_mm)
        self.inner_square_tube_thickness_var = tk.DoubleVar(value=args.inner_square_tube_thickness_mm)
        self.z_start_var = tk.DoubleVar(value=args.z_start_mm)
        self.z_end_var = tk.DoubleVar(value=args.z_end_mm)
        self.pin_path_var = tk.StringVar(value=str(self.pin_path))
        self.output_dir_var = tk.StringVar(value=str(self.output_dir))
        self.status_var = tk.StringVar(value="Adjust the plastic block around the connector pin.")
        self.summary_var = tk.StringVar(value="")
        self.bridge_path = BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
        self.viewer_process: subprocess.Popen | None = None
        self.preview_after_id: str | None = None

        self._build_ui()
        self._attach_live_updates()
        self._start_viewer()
        self._refresh_summary()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Female Pin Connector 1x1", font=("Georgia", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                "Plastic is built as a square tube from outer minus inner square blocks, then the "
                "connector pin enlarged by 0.001 mm is boolean-subtracted. Only the largest remaining "
                "plastic body is kept."
            ),
            wraplength=560,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        files_box = ttk.LabelFrame(outer, text="Paths", padding=10)
        files_box.grid(row=2, column=0, sticky="ew")
        files_box.columnconfigure(1, weight=1)

        ttk.Label(files_box, text="Pin STL").grid(row=0, column=0, sticky="w")
        ttk.Entry(files_box, textvariable=self.pin_path_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(files_box, text="Browse", command=self._browse_pin).grid(row=0, column=2, sticky="ew")

        ttk.Label(files_box, text="Output Folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(files_box, textvariable=self.output_dir_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(files_box, text="Browse", command=self._browse_output_dir).grid(row=1, column=2, sticky="ew", pady=(8, 0))

        sizing_box = ttk.LabelFrame(outer, text="Plastic Tube", padding=10)
        sizing_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        sizing_box.columnconfigure(1, weight=1)

        self._add_number_field(sizing_box, 0, "External Square Side Length (mm)", self.outer_square_side_length_var)
        self._add_number_field(sizing_box, 1, "Inner Square Tube Thickness (mm)", self.inner_square_tube_thickness_var)
        self._add_number_field(sizing_box, 2, "Z Start (mm)", self.z_start_var)
        self._add_number_field(sizing_box, 3, "Z End (mm)", self.z_end_var)

        summary_box = ttk.LabelFrame(outer, text="Build Summary", padding=10)
        summary_box.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(summary_box, textvariable=self.summary_var, wraplength=560, justify="left").pack(anchor="w")

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        ttk.Button(actions, text="Refresh Summary", command=self._refresh_summary).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Reset Defaults", command=self._reset_defaults).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(actions, text="Export STLs", command=self._export).grid(row=0, column=2, sticky="ew")

        ttk.Label(outer, textvariable=self.status_var, wraplength=560, foreground="#7a2f20").grid(
            row=6, column=0, sticky="w", pady=(12, 0)
        )

    def _add_number_field(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.DoubleVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 8, 0))
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=(0 if row == 0 else 8, 0),
            padx=(10, 0),
        )

    def _attach_live_updates(self) -> None:
        for variable in (
            self.outer_square_side_length_var,
            self.inner_square_tube_thickness_var,
            self.z_start_var,
            self.z_end_var,
        ):
            variable.trace_add("write", self._schedule_preview_refresh)

    def _schedule_preview_refresh(self, *_args) -> None:
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(120, self._refresh_summary)

    def _browse_pin(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Select internal connector pin STL",
            initialdir=str(self.pin_path.parent),
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if not chosen:
            return
        self.pin_path = Path(chosen).expanduser().resolve()
        self.pin_path_var.set(str(self.pin_path))
        try:
            self.pin_mesh = load_pin_mesh(self.pin_path)
            self.status_var.set("Loaded a new copper pin STL.")
        except Exception as exc:
            messagebox.showerror("Pin Load Failed", str(exc), parent=self.root)
            self.status_var.set(f"Pin load failed: {exc}")
        self._refresh_summary()

    def _browse_output_dir(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self.root,
            title="Select export folder",
            initialdir=str(self.output_dir),
            mustexist=False,
        )
        if not chosen:
            return
        self.output_dir = Path(chosen).expanduser().resolve()
        self.output_dir_var.set(str(self.output_dir))
        self.status_var.set("Updated the export folder.")

    def _current_spec(self) -> HolderSpec:
        return HolderSpec(
            outer_square_side_length_mm=float(self.outer_square_side_length_var.get()),
            inner_square_tube_thickness_mm=float(self.inner_square_tube_thickness_var.get()),
            z_start_mm=float(self.z_start_var.get()),
            z_end_mm=float(self.z_end_var.get()),
        )

    def _refresh_summary(self) -> None:
        self.preview_after_id = None
        try:
            spec = self._current_spec()
            normalized_pin = normalize_pin_mesh(self.pin_mesh)
            validate_spec(spec)
            pin_width_mm = float(normalized_pin.extents[0])
            pin_depth_mm = float(normalized_pin.extents[1])
            pin_length_mm = float(normalized_pin.extents[2])
            inner_side_mm = spec.outer_square_side_length_mm - (2.0 * spec.inner_square_tube_thickness_mm)
            self.summary_var.set(
                "Pin normalized to origin at centroid.\n"
                "Pin size: "
                f"{pin_width_mm:.3f} X x {pin_depth_mm:.3f} Y x {pin_length_mm:.3f} Z mm\n"
                "Plastic tube: "
                f"{spec.outer_square_side_length_mm:.3f} outer side, {inner_side_mm:.3f} inner side, "
                f"Z [{spec.z_start_mm:.3f}, {spec.z_end_mm:.3f}] mm\n"
                f"Pin enlargement before subtraction: {PIN_ENLARGE_MM:.3f} mm\n"
                "Largest remaining plastic body will be kept after boolean subtraction."
            )
            self.status_var.set("Plastic subtraction settings look valid.")
            self._push_preview_payload(spec)
        except Exception as exc:
            self.summary_var.set("Status: current dimensions are not valid yet.")
            self.status_var.set(str(exc))
            self._push_preview_payload(self._safe_preview_spec())

    def _reset_defaults(self) -> None:
        self.outer_square_side_length_var.set(DEFAULT_SPEC.outer_square_side_length_mm)
        self.inner_square_tube_thickness_var.set(DEFAULT_SPEC.inner_square_tube_thickness_mm)
        self.z_start_var.set(DEFAULT_SPEC.z_start_mm)
        self.z_end_var.set(DEFAULT_SPEC.z_end_mm)
        self.status_var.set("Restored the default plastic tube dimensions.")
        self._refresh_summary()

    def _export(self) -> None:
        try:
            self.output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
            spec = self._current_spec()
            outputs = export_connector(spec=spec, pin_path=self.pin_path, output_dir=self.output_dir)
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc), parent=self.root)
            self.status_var.set(f"Export failed: {exc}")
            return

        self.status_var.set("Exported: " + ", ".join(path.name for path in outputs.values()))
        self.summary_var.set(self.summary_var.get() + "\nExport folder: " + str(self.output_dir))

    def _safe_preview_spec(self) -> HolderSpec:
        def current_or_default(variable: tk.DoubleVar, fallback: float) -> float:
            try:
                return float(variable.get())
            except Exception:
                return fallback

        return HolderSpec(
            outer_square_side_length_mm=current_or_default(
                self.outer_square_side_length_var,
                DEFAULT_SPEC.outer_square_side_length_mm,
            ),
            inner_square_tube_thickness_mm=current_or_default(
                self.inner_square_tube_thickness_var,
                DEFAULT_SPEC.inner_square_tube_thickness_mm,
            ),
            z_start_mm=current_or_default(self.z_start_var, DEFAULT_SPEC.z_start_mm),
            z_end_mm=current_or_default(self.z_end_var, DEFAULT_SPEC.z_end_mm),
        )

    def _push_preview_payload(self, spec: HolderSpec) -> None:
        try:
            write_bridge_payload(
                self.bridge_path,
                build_viewer_payload(
                    spec=spec,
                    pin_path=self.pin_path,
                    status_message=self.status_var.get(),
                    summary_message=self.summary_var.get(),
                ),
            )
        except Exception as exc:
            self.status_var.set(f"Viewer sync failed: {exc}")

    def _start_viewer(self) -> None:
        try:
            write_bridge_payload(
                self.bridge_path,
                build_viewer_payload(
                    spec=self._safe_preview_spec(),
                    pin_path=self.pin_path,
                    status_message="Launching preview...",
                    summary_message=self.summary_var.get(),
                ),
            )
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
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
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


def run_cli(args: argparse.Namespace) -> int:
    spec = build_spec_from_args(args)
    outputs = export_connector(spec=spec, pin_path=args.pin, output_dir=args.output_dir)

    print("Created female 1x1 connector assets:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.viewer_bridge is not None:
        viewer = FemalePinConnectorViewer(args.viewer_bridge.expanduser().resolve())
        viewer.run()
        return 0
    if args.cli:
        return run_cli(args)
    panel = FemalePinConnectorControlPanel(args)
    return panel.run()


if __name__ == "__main__":
    raise SystemExit(main())
