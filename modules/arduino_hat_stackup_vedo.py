from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tkinter as tk
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import shapely.affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
import numpy as np
import trimesh
from vedo import Cylinder, Line, Mesh, Plotter, Text2D

COPPER_THICKNESS_MM = 0.035
SOLDER_MASK_THICKNESS_MM = 0.02
BARREL_LAYER_OVERLAP_MM = 0.005
VIA_SIDE_COUNT = 10
VIA_PLATING_THICKNESS_MM = COPPER_THICKNESS_MM
PAD_CIRCLE_SIDE_COUNT = 10
PAD_CURVE_POINT_COUNT = 5
LAYER_COLORS = {
    "F.Cu": "#d6721d",
    "In1.Cu": "#2a9d8f",
    "In2.Cu": "#277da1",
    "B.Cu": "#8d5fd3",
}
DEFECT_COLORS = {
    "overetch": "#d62828",
    "underetch": "#f4a261",
    "mousebite": "#7b2cbf",
    "open_circuit": "#111111",
    "short_circuit": "#2a9d8f",
}
DIELECTRIC_COLOR = "#315f52"
SOLDER_MASK_COLOR = "#2f6f62"
OUTLINE_COLOR = "#232323"
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTER_SCRIPT = REPO_ROOT / "modules" / "export_kicad_copper_paths.py"
SOURCE_PCB = REPO_ROOT / "DataSet" / "KICAD" / "Arduino hat" / "Arduino_hat.kicad_pcb"
DEFAULT_INPUT = REPO_ROOT / "output" / "arduino_hat" / "copper_paths.json"
COMMON_KICAD_PYTHON_PATHS = (
    Path(r"C:\Program Files\KiCad\10.0\bin\python.exe"),
    Path(r"C:\Program Files\KiCad\9.0\bin\python.exe"),
    Path(r"C:\Program Files\KiCad\8.0\bin\python.exe"),
)
DEFAULT_EXPORT_DIR = REPO_ROOT / "output" / "arduino_hat" / "material_partition_live"
EXPORT_OPTION_LABELS = [
    ("copper_layers", "Copper layer STLs"),
    ("trace_layers", "Trace layer STLs"),
    ("via_barrels", "Via barrel STL"),
    ("pad_barrels", "Pad barrel STL"),
    ("solder_mask_layers", "Solder mask STLs"),
    ("copper_all", "Combined copper STL"),
    ("trace_all", "Combined trace STL"),
    ("via_air", "Via air STL"),
    ("pad_air", "Pad air STL"),
    ("dielectric_layers", "Dielectric layer STLs"),
    ("pcb_parts_all", "Combined PCB parts STL"),
]
EXPORT_OPTION_GROUPS = [
    ("Copper", ["copper_layers", "trace_layers", "via_barrels", "pad_barrels", "copper_all", "trace_all"]),
    ("FR-4 / Dielectric", ["dielectric_layers"]),
    ("Solder Mask", ["solder_mask_layers"]),
    ("Air / Drill Voids", ["via_air", "pad_air"]),
    ("Combined", ["pcb_parts_all"]),
]


@dataclass(slots=True)
class PointMM:
    x_mm: float
    y_mm: float


@dataclass(slots=True)
class TrackData:
    layer: str
    net: str
    start_mm: PointMM
    end_mm: PointMM
    width_mm: float


@dataclass(slots=True)
class ViaData:
    net: str
    position_mm: PointMM
    drill_mm: float
    diameter_by_layer_mm: dict[str, float]


@dataclass(slots=True)
class PadData:
    reference: str
    pad_number: str
    layer: str
    net: str
    center_mm: PointMM
    size_x_mm: float
    size_y_mm: float
    rotation_deg: float
    shape: str
    roundrect_radius_mm: float | None = None
    drill_x_mm: float = 0.0
    drill_y_mm: float = 0.0


@dataclass(slots=True)
class OutlineData:
    kind: str
    start_mm: PointMM
    end_mm: PointMM
    mid_mm: PointMM | None = None


@dataclass(slots=True)
class ZoneContourData:
    exterior_mm: list[PointMM]
    holes_mm: list[list[PointMM]]


@dataclass(slots=True)
class ZoneData:
    layer: str
    net: str
    contours: list[ZoneContourData]


@dataclass(slots=True)
class StackupCopperLayer:
    name: str
    thickness_mm: float


@dataclass(slots=True)
class StackupDielectricLayer:
    name: str
    upper_layer: str
    lower_layer: str
    thickness_mm: float


@dataclass(slots=True)
class StackupDefinition:
    copper_layers: list[StackupCopperLayer]
    dielectric_layers: list[StackupDielectricLayer]


@dataclass(slots=True)
class BoardViewModel:
    board_path: Path
    board_thickness_mm: float
    left_mm: float
    top_mm: float
    width_mm: float
    height_mm: float
    tracks: list[TrackData]
    vias: list[ViaData]
    pads: list[PadData]
    zones: list[ZoneData]
    outline: list[OutlineData]
    active_layers: list[str]
    nets: list[str]
    stackup: StackupDefinition


@dataclass(slots=True)
class OverEtchSettings:
    enabled: bool = False
    count: int = 3
    severity: float = 0.55
    recovery_mm: float = 1.2
    falloff_mode: str = "gaussian"
    noise_amount: float = 0.12
    seed: int = 1


@dataclass(slots=True)
class MouseBiteSettings:
    enabled: bool = False
    count: int = 3
    recovery_mm: float = 1.0
    noise_amount: float = 0.15
    blob_count: int = 4
    blob_size_mm: float = 0.18
    seed: int = 101


@dataclass(slots=True)
class UnderEtchSettings:
    enabled: bool = False
    count: int = 3
    severity: float = 0.45
    recovery_mm: float = 1.4
    falloff_mode: str = "gaussian"
    noise_amount: float = 0.15
    blob_count: int = 4
    blob_size_mm: float = 0.16
    seed: int = 201


@dataclass(slots=True)
class OpenCircuitSettings:
    enabled: bool = False
    count: int = 2
    gap_mm: float = 0.35
    recovery_mm: float = 0.7
    noise_amount: float = 0.05
    seed: int = 301


@dataclass(slots=True)
class ShortCircuitSettings:
    enabled: bool = False
    count: int = 2
    max_gap_mm: float = 0.5
    bridge_width_mm: float = 0.18
    noise_amount: float = 0.05
    seed: int = 401


@dataclass(slots=True)
class TraceDefect:
    defect_type: str
    layer_name: str
    center_x: float
    center_y: float
    tangent_x: float
    tangent_y: float
    track_width_mm: float
    recovery_mm: float
    severity: float
    falloff_mode: str
    noise_amount: float
    noise_seed: int
    blob_count: int = 0
    blob_size_mm: float = 0.0
    secondary_center_x: float | None = None
    secondary_center_y: float | None = None
    secondary_track_width_mm: float | None = None
    bridge_width_mm: float | None = None
    path_polyline_xy: list[tuple[float, float]] | None = None
    path_distance_mm: float | None = None


@dataclass(slots=True)
class DefectListItem:
    index: int
    layer_name: str
    defect: TraceDefect


@dataclass(slots=True)
class MeshRecord:
    name: str
    path: str
    triangle_count: int


@dataclass(slots=True)
class PartitionManifest:
    source: str
    export_dir: str
    defects_enabled: bool
    meshes: list[MeshRecord]


class ExportRefreshError(RuntimeError):
    pass


def export_material_partition_with_defects(
    model: BoardViewModel,
    output_dir: Path = DEFAULT_EXPORT_DIR,
    *,
    overetch: OverEtchSettings | None = None,
    mousebite: MouseBiteSettings | None = None,
    underetch: UnderEtchSettings | None = None,
    opencircuit: OpenCircuitSettings | None = None,
    shortcircuit: ShortCircuitSettings | None = None,
) -> Path:
    track_paths_by_layer = {
        layer: build_connected_track_paths(
            model,
            [track for track in model.tracks if track.layer == layer],
        )
        for layer in model.active_layers
    }
    defects_by_layer = build_all_defects_by_layer(
        track_paths_by_layer,
        overetch or OverEtchSettings(),
        mousebite or MouseBiteSettings(),
        underetch or UnderEtchSettings(),
        opencircuit or OpenCircuitSettings(),
        shortcircuit or ShortCircuitSettings(),
    )
    z_map = build_z_map(
        board_thickness_mm=model.board_thickness_mm,
        stackup=model.stackup,
        copper_layers=model.active_layers,
        explode_scale=1.0,
    )
    return export_partition_from_state(
        model=model,
        output_dir=output_dir,
        track_paths_by_layer=track_paths_by_layer,
        defects_by_layer=defects_by_layer,
        z_map=z_map,
    )


def export_partition_from_state(
    *,
    model: BoardViewModel,
    output_dir: Path,
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]],
    defects_by_layer: dict[str, list[TraceDefect]],
    z_map: dict[str, float],
    selected_outputs: set[str] | None = None,
) -> Path:
    selected_outputs = set(selected_outputs or {name for name, _label in EXPORT_OPTION_LABELS})
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    board_polygon = build_board_polygon(model)
    mesh_records: list[MeshRecord] = []
    copper_meshes: list[trimesh.Trimesh] = []
    trace_meshes: list[trimesh.Trimesh] = []
    layer_mesh_by_name: dict[str, trimesh.Trimesh] = {}
    partition_meshes: list[trimesh.Trimesh] = []

    for layer_name in model.active_layers:
        copper_thickness_mm = copper_thickness_for_layer(model, layer_name)
        trace_geometry = create_layer_copper_polygon(
            model=model,
            layer_name=layer_name,
            track_paths=track_paths_by_layer.get(layer_name, []),
            defects=defects_by_layer.get(layer_name, []),
            include_zones=False,
            include_tracks=True,
            include_pads=False,
        )
        trace_mesh = extrude_geometry(
            trace_geometry,
            height_mm=copper_thickness_mm,
            z_bottom_mm=z_map[layer_name] - (copper_thickness_mm / 2.0),
        )
        if trace_mesh is not None:
            trace_meshes.append(trace_mesh)
            if "trace_layers" in selected_outputs:
                trace_path = output_dir / f"trace_{layer_name.replace('.', '_')}.stl"
                trace_mesh.export(trace_path)
                mesh_records.append(MeshRecord(name=f"trace_{layer_name}", path=str(trace_path), triangle_count=len(trace_mesh.faces)))

        copper_geometry = create_layer_copper_polygon(
            model=model,
            layer_name=layer_name,
            track_paths=track_paths_by_layer.get(layer_name, []),
            defects=defects_by_layer.get(layer_name, []),
            include_zones=True,
            include_tracks=True,
            include_pads=True,
        ).intersection(board_polygon)
        mesh = extrude_geometry(
            copper_geometry,
            height_mm=copper_thickness_mm,
            z_bottom_mm=z_map[layer_name] - (copper_thickness_mm / 2.0),
        )
        if mesh is None:
            continue
        copper_meshes.append(mesh)
        layer_mesh_by_name[layer_name] = mesh
        if "copper_layers" in selected_outputs:
            path = output_dir / f"copper_{layer_name.replace('.', '_')}.stl"
            mesh.export(path)
            mesh_records.append(MeshRecord(name=f"copper_{layer_name}", path=str(path), triangle_count=len(mesh.faces)))

    via_mesh = create_via_barrel_mesh(model, z_map)
    if via_mesh is not None:
        copper_meshes.append(via_mesh)
        if "via_barrels" in selected_outputs:
            via_path = output_dir / "via_barrels.stl"
            via_mesh.export(via_path)
            mesh_records.append(MeshRecord(name="via_barrels", path=str(via_path), triangle_count=len(via_mesh.faces)))

    pad_barrel_mesh = create_pad_barrel_mesh(model, z_map)
    if pad_barrel_mesh is not None:
        copper_meshes.append(pad_barrel_mesh)
        if "pad_barrels" in selected_outputs:
            pad_barrel_path = output_dir / "pad_barrels.stl"
            pad_barrel_mesh.export(pad_barrel_path)
            mesh_records.append(MeshRecord(name="pad_barrels", path=str(pad_barrel_path), triangle_count=len(pad_barrel_mesh.faces)))

    if copper_meshes:
        ordered_union_inputs: list[trimesh.Trimesh] = []
        for layer_name in model.active_layers:
            layer_mesh = layer_mesh_by_name.get(layer_name)
            if layer_mesh is None:
                continue
            ordered_union_inputs.append(layer_mesh)
        via_union_mesh = create_via_barrel_mesh(model, z_map, overlap_mm=0.0)
        pad_union_mesh = create_pad_barrel_mesh(model, z_map, overlap_mm=0.0)
        if via_union_mesh is not None:
            ordered_union_inputs.append(via_union_mesh)
        if pad_union_mesh is not None:
            ordered_union_inputs.append(pad_union_mesh)
        copper_all_mesh = sequential_union(ordered_union_inputs)
        if copper_all_mesh is not None:
            air_cutters: list[trimesh.Trimesh] = []
            via_air_mesh = create_via_air_mesh(model, z_map)
            if via_air_mesh is not None:
                air_cutters.append(via_air_mesh)
            pad_air_mesh = create_pad_air_mesh(model, z_map)
            if pad_air_mesh is not None:
                air_cutters.append(pad_air_mesh)
            if air_cutters:
                air_mesh = trimesh.util.concatenate(air_cutters)
                cut_mesh = boolean_difference(copper_all_mesh, air_mesh)
                if cut_mesh is not None:
                    copper_all_mesh = cut_mesh
            partition_meshes.append(copper_all_mesh)
            if "copper_all" in selected_outputs:
                copper_all_path = output_dir / "copper_all.stl"
                copper_all_mesh.export(copper_all_path)
                mesh_records.append(MeshRecord(name="copper_all", path=str(copper_all_path), triangle_count=len(copper_all_mesh.faces)))

    if trace_meshes:
        trace_all_mesh = trimesh.util.concatenate(trace_meshes)
        if "trace_all" in selected_outputs:
            trace_all_path = output_dir / "trace_all.stl"
            trace_all_mesh.export(trace_all_path)
            mesh_records.append(MeshRecord(name="trace_all", path=str(trace_all_path), triangle_count=len(trace_all_mesh.faces)))

    via_air_mesh = create_via_air_mesh(model, z_map)
    if via_air_mesh is not None:
        if "via_air" in selected_outputs:
            via_air_path = output_dir / "via_air.stl"
            via_air_mesh.export(via_air_path)
            mesh_records.append(MeshRecord(name="via_air", path=str(via_air_path), triangle_count=len(via_air_mesh.faces)))

    pad_air_mesh = create_pad_air_mesh(model, z_map)
    if pad_air_mesh is not None:
        if "pad_air" in selected_outputs:
            pad_air_path = output_dir / "pad_air.stl"
            pad_air_mesh.export(pad_air_path)
            mesh_records.append(MeshRecord(name="pad_air", path=str(pad_air_path), triangle_count=len(pad_air_mesh.faces)))

    for side_name, side_label in (("top", "F.Mask"), ("bottom", "B.Mask")):
        mask_geometry = create_solder_mask_geometry(
            model,
            side_name,
            board_polygon,
            track_paths_by_layer=track_paths_by_layer,
            defects_by_layer=defects_by_layer,
        )
        layer_name = mask_layer_order(model, side_name)
        if layer_name is None:
            continue
        if side_name == "top":
            z_bottom = copper_bounds_for_layer(model, layer_name, z_map)[1]
        else:
            z_bottom = copper_bounds_for_layer(model, layer_name, z_map)[0] - SOLDER_MASK_THICKNESS_MM
        mask_mesh = extrude_geometry(mask_geometry, height_mm=SOLDER_MASK_THICKNESS_MM, z_bottom_mm=z_bottom)
        if mask_mesh is None:
            continue
        partition_meshes.append(mask_mesh)
        if "solder_mask_layers" in selected_outputs:
            mask_path = output_dir / f"solder_mask_{side_label.replace('.', '_')}.stl"
            mask_mesh.export(mask_path)
            mesh_records.append(MeshRecord(name=f"solder_mask_{side_label}", path=str(mask_path), triangle_count=len(mask_mesh.faces)))

    for index in range(len(model.active_layers) - 1):
        upper_layer = model.active_layers[index]
        lower_layer = model.active_layers[index + 1]
        upper_z = copper_bounds_for_layer(model, upper_layer, z_map)[0]
        lower_z = copper_bounds_for_layer(model, lower_layer, z_map)[1]
        height_mm = upper_z - lower_z
        if height_mm <= 0:
            continue
        slab_geometry = create_dielectric_slab_polygon(board_polygon, model, index, index + 1, 0.0)
        mesh = extrude_geometry(slab_geometry, height_mm=height_mm, z_bottom_mm=lower_z)
        if mesh is None:
            continue
        slab_name = f"fr4_{upper_layer.replace('.', '_')}_to_{lower_layer.replace('.', '_')}"
        partition_meshes.append(mesh)
        if "dielectric_layers" in selected_outputs:
            path = output_dir / f"{slab_name}.stl"
            mesh.export(path)
            mesh_records.append(MeshRecord(name=slab_name, path=str(path), triangle_count=len(mesh.faces)))

    if partition_meshes:
        pcb_parts_all_mesh = trimesh.util.concatenate(partition_meshes)
        if "pcb_parts_all" in selected_outputs:
            pcb_parts_all_path = output_dir / "pcb_parts_all.stl"
            pcb_parts_all_mesh.export(pcb_parts_all_path)
            mesh_records.append(MeshRecord(name="pcb_parts_all", path=str(pcb_parts_all_path), triangle_count=len(pcb_parts_all_mesh.faces)))

    manifest = PartitionManifest(
        source=str(model.board_path),
        export_dir=str(output_dir),
        defects_enabled=any(len(items) > 0 for items in defects_by_layer.values()),
        meshes=mesh_records,
    )
    manifest_path = output_dir / "material_partition_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return output_dir


class VedoStackupViewer:
    def __init__(self, model: BoardViewModel, export_output_dir: Path | None = None) -> None:
        self.model = model
        self.export_output_dir = export_output_dir.expanduser().resolve() if export_output_dir is not None else DEFAULT_EXPORT_DIR
        self.explode_scale = 1.0
        self.visible_layers = {layer: True for layer in model.active_layers}
        self.show_dielectric = True
        self.show_solder_mask = True
        self.show_tracks = True
        self.show_pads = True
        self.show_vias = True
        self.show_outline = True
        self.show_defect_regions = False
        self.selected_net = "All nets"
        self.overetch_settings = OverEtchSettings()
        self.mousebite_settings = MouseBiteSettings()
        self.underetch_settings = UnderEtchSettings()
        self.opencircuit_settings = OpenCircuitSettings()
        self.shortcircuit_settings = ShortCircuitSettings()
        self.track_paths_by_layer = {
            layer: build_connected_track_paths(
                model,
                [track for track in model.tracks if track.layer == layer],
            )
            for layer in model.active_layers
        }
        self.defects_by_layer: dict[str, list[TraceDefect]] = {}

        self.plotter = Plotter(
            title=f"{self.model.board_path.stem} 3D Stackup Viewer",
            bg="#efe7d2",
            bg2="#f6f0e2",
            axes=1,
            size=(1500, 950),
        )
        self.info = Text2D("", pos="bottom-left", s=0.75, c="#332920", bg=None, font="Courier")
        self.legend = Text2D("", pos="top-left", s=0.7, c="#332920", bg=None, font="Courier")

        self.track_actors: dict[str, list] = {}
        self.pad_actors: dict[str, list] = {}
        self.via_actors: list = []
        self.layer_actors: dict[str, list] = {}
        self.outline_actors: list = []
        self.dielectric_actors: list = []
        self.solder_mask_actors: list = []
        self.defect_marker_actors: list = []
        self.defect_region_actors: list = []
        self.selected_defect_actor = None
        self.layer_z_map: dict[str, float] = {}
        self.defect_list_items: list[DefectListItem] = []

        self._build_scene()

    def replace_model(self, model: BoardViewModel, export_output_dir: Path | None = None) -> None:
        self.model = model
        if export_output_dir is not None:
            self.export_output_dir = export_output_dir.expanduser().resolve()
        self.visible_layers = {layer: True for layer in model.active_layers}
        self.selected_net = "All nets"
        self.track_paths_by_layer = {
            layer: build_connected_track_paths(
                model,
                [track for track in model.tracks if track.layer == layer],
            )
            for layer in model.active_layers
        }
        self.defects_by_layer = {}
        self.layer_z_map = {}
        self.defect_list_items = []
        self._rebuild()

    def _build_scene(self) -> None:
        self.plotter.clear()
        self.track_actors.clear()
        self.pad_actors.clear()
        self.via_actors.clear()
        self.layer_actors.clear()
        self.outline_actors.clear()
        self.dielectric_actors.clear()
        self.solder_mask_actors.clear()
        self.defect_marker_actors.clear()
        self.defect_region_actors.clear()
        self.selected_defect_actor = None
        visible_layers = {layer for layer, is_visible in self.visible_layers.items() if is_visible}

        self.layer_z_map = build_z_map(
            board_thickness_mm=self.model.board_thickness_mm,
            stackup=self.model.stackup,
            copper_layers=self.model.active_layers,
            explode_scale=self.explode_scale,
        )
        self.defects_by_layer = build_all_defects_by_layer(
            self.track_paths_by_layer,
            self.overetch_settings,
            self.mousebite_settings,
            self.underetch_settings,
            self.opencircuit_settings,
            self.shortcircuit_settings,
        )
        self.defect_list_items = self._build_defect_list_items()

        if self.show_dielectric:
            self.dielectric_actors = self._create_dielectric_actors(visible_layers)
            self.plotter += self.dielectric_actors

        if self.show_solder_mask:
            self.solder_mask_actors = self._create_solder_mask_actors()
            self.plotter += self.solder_mask_actors

        if self.show_outline:
            self.outline_actors = self._create_outline_actors()
            self.plotter += self.outline_actors

        for layer_name in self.model.active_layers:
            actors: list = []
            if self.visible_layers[layer_name]:
                if self.show_tracks or self.show_pads:
                    copper_regions = self._create_layer_copper_region_actors(layer_name)
                    self.track_actors[layer_name] = copper_regions
                    actors.extend(copper_regions)
            self.layer_actors[layer_name] = actors
            self.plotter += actors

        if self.show_vias:
            self.via_actors = self._create_via_actors()
            self.plotter += self.via_actors

        if any(
            settings.enabled
            for settings in (
                self.overetch_settings,
                self.mousebite_settings,
                self.underetch_settings,
                self.opencircuit_settings,
                self.shortcircuit_settings,
            )
        ):
            if self.show_defect_regions:
                self.defect_region_actors = self._create_defect_region_actors(visible_layers)
                self.plotter += self.defect_region_actors
            self.defect_marker_actors = self._create_defect_marker_actors(visible_layers)
            self.plotter += self.defect_marker_actors

        self._update_text()
        self.plotter += self.info
        self.plotter += self.legend
        self.plotter.add_callback("KeyPress", self._on_keypress)

    def _on_keypress(self, event) -> None:
        key = str(event.keypress).lower()
        if key == "r":
            self.explode_scale = 1.0
            self._rebuild()

    def _rebuild(self) -> None:
        camera = self.plotter.camera
        self._build_scene()
        if camera is not None:
            self.plotter.camera = camera
        self.plotter.render()

    def _create_dielectric_actors(self, visible_layers: set[str]) -> list:
        actors: list = []
        ordered_layers = self.model.active_layers
        gap_indices: set[int] = set()
        last_index = len(ordered_layers) - 1
        board_polygon = build_board_polygon(self.model)

        for index, layer_name in enumerate(ordered_layers):
            if layer_name not in visible_layers:
                continue
            if index == last_index:
                if last_index > 0:
                    gap_indices.add(last_index - 1)
            else:
                gap_indices.add(index)

        for index in sorted(gap_indices):
            upper_layer = ordered_layers[index]
            lower_layer = ordered_layers[index + 1]
            upper_z = copper_bounds_for_layer(self.model, upper_layer, self.layer_z_map)[0]
            lower_z = copper_bounds_for_layer(self.model, lower_layer, self.layer_z_map)[1]
            height = upper_z - lower_z
            if height <= 0:
                continue
            slab_geometry = create_dielectric_slab_polygon(board_polygon, self.model, index, index + 1, 0.0)
            for polygon in iter_polygons(slab_geometry):
                actor = extrude_polygon_to_vedo_mesh(
                    polygon=polygon,
                    z_bottom=lower_z,
                    height=height,
                    color=DIELECTRIC_COLOR,
                )
                if actor is not None:
                    actor.alpha(0.28)
                    actors.append(actor)
        return actors

    def _create_solder_mask_actors(self) -> list:
        actors: list = []
        board_polygon = build_board_polygon(self.model)
        for side_name, layer_name in (("top", mask_layer_order(self.model, "top")), ("bottom", mask_layer_order(self.model, "bottom"))):
            if layer_name is None or not self.visible_layers.get(layer_name, False):
                continue
            mask_geometry = create_solder_mask_geometry(
                self.model,
                side_name,
                board_polygon,
                track_paths_by_layer=self.track_paths_by_layer,
                defects_by_layer=self.defects_by_layer,
            )
            if side_name == "top":
                z_bottom = copper_bounds_for_layer(self.model, layer_name, self.layer_z_map)[1]
            else:
                z_bottom = copper_bounds_for_layer(self.model, layer_name, self.layer_z_map)[0] - SOLDER_MASK_THICKNESS_MM
            for polygon in iter_polygons(mask_geometry):
                actor = extrude_polygon_to_vedo_mesh(
                    polygon=polygon,
                    z_bottom=z_bottom,
                    height=SOLDER_MASK_THICKNESS_MM,
                    color=SOLDER_MASK_COLOR,
                )
                if actor is not None:
                    actor.alpha(0.5)
                    actors.append(actor)
        return actors

    def _create_outline_actors(self) -> list:
        actors: list = []
        top_z = max(copper_bounds_for_layer(self.model, layer_name, self.layer_z_map)[1] for layer_name in self.model.active_layers) + 0.04
        for item in self.model.outline:
            if item.kind == "segment":
                x1, y1 = board_to_centered(self.model, item.start_mm)
                x2, y2 = board_to_centered(self.model, item.end_mm)
                actors.append(Line((x1, y1, top_z), (x2, y2, top_z)).c(OUTLINE_COLOR).lw(2))
            elif item.kind == "arc" and item.mid_mm is not None:
                points = sample_quadratic_arc(item.start_mm, item.mid_mm, item.end_mm, 10)
                xyz = []
                for point in points:
                    x, y = board_to_centered(self.model, point)
                    xyz.append((x, y, top_z))
                actors.append(Line(xyz).c(OUTLINE_COLOR).lw(2))
        return actors

    def _create_layer_copper_region_actors(self, layer_name: str) -> list:
        actors: list = []
        copper_geometry = create_layer_copper_polygon(
            model=self.model,
            layer_name=layer_name,
            track_paths=self.track_paths_by_layer.get(layer_name, []),
            defects=self.defects_by_layer.get(layer_name, []),
            include_zones=self.show_tracks or self.show_pads,
            include_tracks=self.show_tracks,
            include_pads=self.show_pads,
        )
        copper_thickness_mm = copper_thickness_for_layer(self.model, layer_name)
        z_bottom = self.layer_z_map[layer_name] - (copper_thickness_mm / 2.0)
        for polygon in iter_polygons(copper_geometry):
            actor = extrude_polygon_to_vedo_mesh(
                polygon=polygon,
                z_bottom=z_bottom,
                height=copper_thickness_mm,
                color=LAYER_COLORS.get(layer_name, "#666666"),
            )
            if actor is not None:
                actors.append(actor)
        return actors

    def _create_via_actors(self) -> list:
        actors: list = []
        for via in self.model.vias:
            layer_names = [layer for layer in self.model.active_layers if layer in via.diameter_by_layer_mm]
            if not layer_names:
                continue
            center_x, center_y = board_to_centered(self.model, via.position_mm)
            bounds = [copper_bounds_for_layer(self.model, layer, self.layer_z_map) for layer in layer_names]
            z_min = min(bound[0] for bound in bounds)
            z_max = max(bound[1] for bound in bounds)
            actor = make_hollow_via_mesh(
                center_x=center_x,
                center_y=center_y,
                z_min=z_min,
                z_max=z_max,
                outer_radius=via_barrel_outer_radius(via),
                inner_radius=via_drill_radius(via),
                color="#d3bb8d",
            )
            actors.append(actor)
        return actors

    def _create_defect_marker_actors(self, visible_layers: set[str]) -> list:
        actors: list = []
        for layer_name, defects in self.defects_by_layer.items():
            if layer_name not in visible_layers:
                continue
            marker_z = copper_bounds_for_layer(self.model, layer_name, self.layer_z_map)[1] + (copper_thickness_for_layer(self.model, layer_name) * 1.3)
            for defect in defects:
                half_side = max(defect.recovery_mm * 0.95, defect.track_width_mm * 1.75)
                corners = [
                    (defect.center_x - half_side, defect.center_y - half_side, marker_z),
                    (defect.center_x + half_side, defect.center_y - half_side, marker_z),
                    (defect.center_x + half_side, defect.center_y + half_side, marker_z),
                    (defect.center_x - half_side, defect.center_y + half_side, marker_z),
                    (defect.center_x - half_side, defect.center_y - half_side, marker_z),
                ]
                actors.append(Line(corners).c(defect_color(defect)).lw(3))
        return actors

    def _create_defect_region_actors(self, visible_layers: set[str]) -> list:
        actors: list = []
        for layer_name, defects in self.defects_by_layer.items():
            if layer_name not in visible_layers:
                continue
            z_bottom = copper_bounds_for_layer(self.model, layer_name, self.layer_z_map)[1] + 0.002
            height = max(copper_thickness_for_layer(self.model, layer_name) * 0.35, 0.01)
            for defect in defects:
                geometry = build_defect_geometry(defect)
                if geometry.is_empty:
                    continue
                for polygon in iter_polygons(geometry):
                    actor = extrude_polygon_to_vedo_mesh(
                        polygon=polygon,
                        z_bottom=z_bottom,
                        height=height,
                        color=defect_color(defect),
                    )
                    if actor is not None:
                        actor.alpha(0.45)
                        actors.append(actor)
        return actors

    def _build_defect_list_items(self) -> list[DefectListItem]:
        items: list[DefectListItem] = []
        index = 0
        for layer_name in self.model.active_layers:
            for defect in self.defects_by_layer.get(layer_name, []):
                items.append(DefectListItem(index=index, layer_name=layer_name, defect=defect))
                index += 1
        return items

    def _update_text(self) -> None:
        visible = [layer for layer in self.model.active_layers if self.visible_layers[layer]]
        self.info.text(
            "R: reset explode  |  1-4: toggle layers\n"
            f"Explode: {self.explode_scale:.2f}  |  Visible: {', '.join(visible) or 'none'}\n"
            f"Tracks: {'on' if self.show_tracks else 'off'}  Pads: {'on' if self.show_pads else 'off'}  "
            f"Vias: {'on' if self.show_vias else 'off'}  Dielectric: {'on' if self.show_dielectric else 'off'}  "
            f"Mask: {'on' if self.show_solder_mask else 'off'}\n"
            f"Defects: {len([item for items in self.defects_by_layer.values() for item in items])}  "
            f"Over: {'on' if self.overetch_settings.enabled else 'off'}  Under: {'on' if self.underetch_settings.enabled else 'off'}  "
            f"Bite: {'on' if self.mousebite_settings.enabled else 'off'}  Open: {'on' if self.opencircuit_settings.enabled else 'off'}  "
            f"Short: {'on' if self.shortcircuit_settings.enabled else 'off'}"
        )
        legend_lines = [
            f"PCB: {self.model.board_path.name}",
            f"Thickness: {self.model.board_thickness_mm:.2f} mm",
            "Layer colors:",
        ]
        for layer_name in self.model.active_layers:
            legend_lines.append(f"  {layer_name}: {LAYER_COLORS.get(layer_name, '#666666')}")
        legend_lines.append(f"Solder mask: {SOLDER_MASK_COLOR}")
        legend_lines.append("Defect colors:")
        legend_lines.append(f"  Over-etch: {DEFECT_COLORS['overetch']}")
        legend_lines.append(f"  Under-etch: {DEFECT_COLORS['underetch']}")
        legend_lines.append(f"  Mousebite: {DEFECT_COLORS['mousebite']}")
        legend_lines.append(f"  Open circuit: {DEFECT_COLORS['open_circuit']}")
        legend_lines.append(f"  Short circuit: {DEFECT_COLORS['short_circuit']}")
        self.legend.text("\n".join(legend_lines))

    def show(self) -> None:
        self.plotter.show(zoom="tight", interactive=False)

    def process_events(self) -> None:
        self.plotter.process_events()

    def close(self) -> None:
        self.plotter.close()

    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        if self.visible_layers.get(layer_name) == visible:
            return
        self.visible_layers[layer_name] = visible
        self._rebuild()

    def set_extras_visibility(
        self,
        *,
        dielectric: bool,
        solder_mask: bool,
        tracks: bool,
        pads: bool,
        vias: bool,
        outline: bool,
        defect_regions: bool,
    ) -> None:
        new_state = (dielectric, solder_mask, tracks, pads, vias, outline, defect_regions)
        old_state = (self.show_dielectric, self.show_solder_mask, self.show_tracks, self.show_pads, self.show_vias, self.show_outline, self.show_defect_regions)
        if new_state == old_state:
            return
        self.show_dielectric = dielectric
        self.show_solder_mask = solder_mask
        self.show_tracks = tracks
        self.show_pads = pads
        self.show_vias = vias
        self.show_outline = outline
        self.show_defect_regions = defect_regions
        self._rebuild()

    def set_explode_scale(self, value: float) -> None:
        value = float(value)
        if abs(self.explode_scale - value) < 1e-6:
            return
        self.explode_scale = value
        self._rebuild()

    def set_overetch_settings(self, settings: OverEtchSettings) -> None:
        if settings == self.overetch_settings:
            return
        self.overetch_settings = settings
        self._rebuild()

    def set_mousebite_settings(self, settings: MouseBiteSettings) -> None:
        if settings == self.mousebite_settings:
            return
        self.mousebite_settings = settings
        self._rebuild()

    def set_underetch_settings(self, settings: UnderEtchSettings) -> None:
        if settings == self.underetch_settings:
            return
        self.underetch_settings = settings
        self._rebuild()

    def set_opencircuit_settings(self, settings: OpenCircuitSettings) -> None:
        if settings == self.opencircuit_settings:
            return
        self.opencircuit_settings = settings
        self._rebuild()

    def set_shortcircuit_settings(self, settings: ShortCircuitSettings) -> None:
        if settings == self.shortcircuit_settings:
            return
        self.shortcircuit_settings = settings
        self._rebuild()

    def set_all_defect_settings(
        self,
        *,
        overetch: OverEtchSettings,
        mousebite: MouseBiteSettings,
        underetch: UnderEtchSettings,
        opencircuit: OpenCircuitSettings,
        shortcircuit: ShortCircuitSettings,
    ) -> None:
        if (
            overetch == self.overetch_settings
            and mousebite == self.mousebite_settings
            and underetch == self.underetch_settings
            and opencircuit == self.opencircuit_settings
            and shortcircuit == self.shortcircuit_settings
        ):
            return
        self.overetch_settings = overetch
        self.mousebite_settings = mousebite
        self.underetch_settings = underetch
        self.opencircuit_settings = opencircuit
        self.shortcircuit_settings = shortcircuit
        self._rebuild()

    def export_current_material_partition(self, output_dir: Path | None = None, selected_outputs: set[str] | None = None) -> Path:
        z_map = self.layer_z_map or build_z_map(
            board_thickness_mm=self.model.board_thickness_mm,
            stackup=self.model.stackup,
            copper_layers=self.model.active_layers,
            explode_scale=1.0,
        )
        return export_partition_from_state(
            model=self.model,
            output_dir=output_dir or self.export_output_dir,
            track_paths_by_layer=self.track_paths_by_layer,
            defects_by_layer=self.defects_by_layer,
            z_map=z_map,
            selected_outputs=selected_outputs,
        )

    def focus_defect_by_index(self, index: int) -> None:
        if index < 0 or index >= len(self.defect_list_items):
            return
        item = self.defect_list_items[index]
        defect = item.defect
        if self.selected_defect_actor is not None:
            try:
                self.plotter.remove(self.selected_defect_actor)
            except Exception:
                pass
            self.selected_defect_actor = None
        marker_z = copper_bounds_for_layer(self.model, item.layer_name, self.layer_z_map)[1] + (copper_thickness_for_layer(self.model, item.layer_name) * 1.8)
        half_side = max(defect.recovery_mm * 1.1, defect.track_width_mm * 2.0)
        corners = [
            (defect.center_x - half_side, defect.center_y - half_side, marker_z),
            (defect.center_x + half_side, defect.center_y - half_side, marker_z),
            (defect.center_x + half_side, defect.center_y + half_side, marker_z),
            (defect.center_x - half_side, defect.center_y + half_side, marker_z),
            (defect.center_x - half_side, defect.center_y - half_side, marker_z),
        ]
        self.selected_defect_actor = Line(corners).c("#ffffff").lw(5)
        self.plotter += self.selected_defect_actor
        camera = self.plotter.camera
        if camera is not None:
            position = camera.GetPosition()
            focal = camera.GetFocalPoint()
            dx = position[0] - focal[0]
            dy = position[1] - focal[1]
            dz = position[2] - focal[2]
            target_z = marker_z
            camera.SetFocalPoint(defect.center_x, defect.center_y, target_z)
            camera.SetPosition(defect.center_x + dx, defect.center_y + dy, target_z + dz)
        self.plotter.render()


class StackupControlPanel:
    def __init__(self, viewer: VedoStackupViewer) -> None:
        self.viewer = viewer
        self.root = tk.Tk()
        self.root.title("PCB Stackup Viewer")
        self.root.geometry("680x860")
        self.root.minsize(620, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.display_tab = ttk.Frame(self.notebook, padding=14)
        self.defects_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.display_tab, text="Display")
        self.notebook.add(self.defects_tab, text="Defects")
        self.notebook.add(self.settings_tab, text="Settings")
        self.defects_canvas = tk.Canvas(self.defects_tab, highlightthickness=0)
        self.defects_scrollbar = ttk.Scrollbar(self.defects_tab, orient="vertical", command=self.defects_canvas.yview)
        self.defects_content = ttk.Frame(self.defects_canvas, padding=14)
        self.defects_window = self.defects_canvas.create_window((0, 0), window=self.defects_content, anchor="nw")
        self.defects_canvas.configure(yscrollcommand=self.defects_scrollbar.set)
        self.defects_canvas.pack(side="left", fill="both", expand=True)
        self.defects_scrollbar.pack(side="right", fill="y")
        self.defects_content.bind("<Configure>", self._on_content_configure)
        self.defects_canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()
        self.board_path_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.layer_vars: dict[str, tk.BooleanVar] = {}
        self.export_option_vars = {}
        self.defect_list_window: tk.Toplevel | None = None
        self.defect_listbox: tk.Listbox | None = None
        self.inline_defect_listbox: tk.Listbox | None = None
        self.settings_window: tk.Toplevel | None = None
        self._sync_vars_from_viewer()
        self._build_ui()
        self.viewer.show()
        self.root.after(16, self._pump_viewer)

    def _sync_vars_from_viewer(self) -> None:
        self.board_path_var.set(str(self.viewer.model.board_path))
        self.status_var.set(f"Loaded: {self.viewer.model.board_path.name}")
        self.layer_vars = {
            layer: tk.BooleanVar(value=self.viewer.visible_layers.get(layer, True))
            for layer in self.viewer.model.active_layers
        }
        self.show_dielectric_var = tk.BooleanVar(value=self.viewer.show_dielectric)
        self.show_solder_mask_var = tk.BooleanVar(value=self.viewer.show_solder_mask)
        self.show_tracks_var = tk.BooleanVar(value=self.viewer.show_tracks)
        self.show_pads_var = tk.BooleanVar(value=self.viewer.show_pads)
        self.show_vias_var = tk.BooleanVar(value=self.viewer.show_vias)
        self.show_outline_var = tk.BooleanVar(value=self.viewer.show_outline)
        self.show_defect_regions_var = tk.BooleanVar(value=self.viewer.show_defect_regions)
        self.explode_var = tk.DoubleVar(value=self.viewer.explode_scale)
        self.overetch_enabled_var = tk.BooleanVar(value=self.viewer.overetch_settings.enabled)
        self.overetch_count_var = tk.IntVar(value=self.viewer.overetch_settings.count)
        self.overetch_severity_var = tk.DoubleVar(value=self.viewer.overetch_settings.severity)
        self.overetch_recovery_var = tk.DoubleVar(value=self.viewer.overetch_settings.recovery_mm)
        self.overetch_noise_var = tk.DoubleVar(value=self.viewer.overetch_settings.noise_amount)
        self.overetch_seed_var = tk.IntVar(value=self.viewer.overetch_settings.seed)
        self.overetch_falloff_var = tk.StringVar(value=self.viewer.overetch_settings.falloff_mode)
        self.mousebite_enabled_var = tk.BooleanVar(value=self.viewer.mousebite_settings.enabled)
        self.mousebite_count_var = tk.IntVar(value=self.viewer.mousebite_settings.count)
        self.mousebite_recovery_var = tk.DoubleVar(value=self.viewer.mousebite_settings.recovery_mm)
        self.mousebite_noise_var = tk.DoubleVar(value=self.viewer.mousebite_settings.noise_amount)
        self.mousebite_blob_count_var = tk.IntVar(value=self.viewer.mousebite_settings.blob_count)
        self.mousebite_blob_size_var = tk.DoubleVar(value=self.viewer.mousebite_settings.blob_size_mm)
        self.mousebite_seed_var = tk.IntVar(value=self.viewer.mousebite_settings.seed)
        self.underetch_enabled_var = tk.BooleanVar(value=self.viewer.underetch_settings.enabled)
        self.underetch_count_var = tk.IntVar(value=self.viewer.underetch_settings.count)
        self.underetch_severity_var = tk.DoubleVar(value=self.viewer.underetch_settings.severity)
        self.underetch_recovery_var = tk.DoubleVar(value=self.viewer.underetch_settings.recovery_mm)
        self.underetch_noise_var = tk.DoubleVar(value=self.viewer.underetch_settings.noise_amount)
        self.underetch_blob_count_var = tk.IntVar(value=self.viewer.underetch_settings.blob_count)
        self.underetch_blob_size_var = tk.DoubleVar(value=self.viewer.underetch_settings.blob_size_mm)
        self.underetch_seed_var = tk.IntVar(value=self.viewer.underetch_settings.seed)
        self.underetch_falloff_var = tk.StringVar(value=self.viewer.underetch_settings.falloff_mode)
        self.opencircuit_enabled_var = tk.BooleanVar(value=self.viewer.opencircuit_settings.enabled)
        self.opencircuit_count_var = tk.IntVar(value=self.viewer.opencircuit_settings.count)
        self.opencircuit_gap_var = tk.DoubleVar(value=self.viewer.opencircuit_settings.gap_mm)
        self.opencircuit_recovery_var = tk.DoubleVar(value=self.viewer.opencircuit_settings.recovery_mm)
        self.opencircuit_noise_var = tk.DoubleVar(value=self.viewer.opencircuit_settings.noise_amount)
        self.opencircuit_seed_var = tk.IntVar(value=self.viewer.opencircuit_settings.seed)
        self.shortcircuit_enabled_var = tk.BooleanVar(value=self.viewer.shortcircuit_settings.enabled)
        self.shortcircuit_count_var = tk.IntVar(value=self.viewer.shortcircuit_settings.count)
        self.shortcircuit_gap_var = tk.DoubleVar(value=self.viewer.shortcircuit_settings.max_gap_mm)
        self.shortcircuit_width_var = tk.DoubleVar(value=self.viewer.shortcircuit_settings.bridge_width_mm)
        self.shortcircuit_noise_var = tk.DoubleVar(value=self.viewer.shortcircuit_settings.noise_amount)
        self.shortcircuit_seed_var = tk.IntVar(value=self.viewer.shortcircuit_settings.seed)
        self.export_option_vars = {
            option_name: self.export_option_vars.get(option_name, tk.BooleanVar(value=False))
            for option_name, _label in EXPORT_OPTION_LABELS
        }

    def _build_ui(self) -> None:
        for parent in (self.display_tab, self.defects_content, self.settings_tab):
            for child in parent.winfo_children():
                child.destroy()

        display = self.display_tab
        defects = self.defects_content
        settings = self.settings_tab

        ttk.Label(display, text=self.viewer.model.board_path.stem, font=("Georgia", 18, "bold")).pack(anchor="w")
        ttk.Label(
            display,
            text="Choose which board and material layers are visible in the vedo window.",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 14))

        source_box = ttk.LabelFrame(display, text="PCB Source", padding=10)
        source_box.pack(fill="x", pady=(0, 12))
        ttk.Label(source_box, text="Current .kicad_pcb", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Entry(source_box, textvariable=self.board_path_var).pack(fill="x", pady=(6, 0))
        source_button_row = ttk.Frame(source_box)
        source_button_row.pack(fill="x", pady=(8, 0))
        ttk.Button(source_button_row, text="Browse And Load PCB", command=self._choose_and_load_pcb).pack(side="left", fill="x", expand=True)
        ttk.Button(source_button_row, text="Reload Current PCB", command=self._reload_current_pcb).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(source_box, textvariable=self.status_var, wraplength=520).pack(anchor="w", pady=(8, 0))

        layers_box = ttk.LabelFrame(display, text="Copper Layers", padding=10)
        layers_box.pack(fill="x")
        for layer_name in self.viewer.model.active_layers:
            ttk.Checkbutton(
                layers_box,
                text=layer_name,
                variable=self.layer_vars[layer_name],
                command=self._apply_layer_visibility,
            ).pack(anchor="w", pady=2)

        extras_box = ttk.LabelFrame(display, text="Other Geometry", padding=10)
        extras_box.pack(fill="x", pady=(12, 0))
        extras = [
            ("Fiberglass / dielectric", self.show_dielectric_var),
            ("Solder mask (F/B)", self.show_solder_mask_var),
            ("Copper tracks", self.show_tracks_var),
            ("Copper pads", self.show_pads_var),
            ("Vias", self.show_vias_var),
            ("Board outline", self.show_outline_var),
            ("Defect regions", self.show_defect_regions_var),
        ]
        for label, variable in extras:
            ttk.Checkbutton(
                extras_box,
                text=label,
                variable=variable,
                command=self._apply_extras_visibility,
            ).pack(anchor="w", pady=2)

        explode_box = ttk.LabelFrame(display, text="Explode Spacing", padding=10)
        explode_box.pack(fill="x", pady=(12, 0))
        ttk.Scale(
            explode_box,
            from_=0.6,
            to=2.6,
            variable=self.explode_var,
            orient="horizontal",
            command=self._apply_explode_scale,
        ).pack(fill="x")
        ttk.Label(
            explode_box,
            text="Higher values separate the material stack more clearly.",
            wraplength=520,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Button(display, text="Reset Explode", command=self._reset_explode).pack(fill="x", pady=(14, 0))

        ttk.Label(defects, text="Defect Controls", font=("Georgia", 16, "bold")).pack(anchor="w")
        ttk.Label(
            defects,
            text="Adjust defect generation here, then select a defect from the list to focus it in the 3D viewer.",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 14))
        self._build_overetch_box(defects)
        self._build_mousebite_box(defects)
        self._build_underetch_box(defects)
        self._build_opencircuit_box(defects)
        self._build_shortcircuit_box(defects)

        defect_browser_box = ttk.LabelFrame(defects, text="Defect Browser", padding=10)
        defect_browser_box.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Button(defect_browser_box, text="Refresh Defect List", command=self._refresh_inline_defect_list).pack(fill="x")
        ttk.Button(defect_browser_box, text="Focus Selected Defect", command=self._focus_selected_inline_defect).pack(fill="x", pady=(8, 0))
        list_frame = ttk.Frame(defect_browser_box)
        list_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.inline_defect_listbox = tk.Listbox(list_frame, activestyle="dotbox", height=12)
        inline_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.inline_defect_listbox.yview)
        self.inline_defect_listbox.configure(yscrollcommand=inline_scrollbar.set)
        self.inline_defect_listbox.pack(side="left", fill="both", expand=True)
        inline_scrollbar.pack(side="right", fill="y")
        self.inline_defect_listbox.bind("<<ListboxSelect>>", self._on_inline_defect_select)
        self._refresh_inline_defect_list()

        ttk.Label(settings, text="Settings", font=("Georgia", 16, "bold")).pack(anchor="w")
        ttk.Label(
            settings,
            text="Choose which STL outputs to generate when exporting the current partition.",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 14))
        export_box = ttk.LabelFrame(settings, text="Export Outputs", padding=10)
        export_box.pack(fill="x")
        label_by_option = dict(EXPORT_OPTION_LABELS)
        for group_name, option_names in EXPORT_OPTION_GROUPS:
            group_box = ttk.LabelFrame(export_box, text=group_name, padding=8)
            group_box.pack(fill="x", pady=(0, 8))
            for option_name in option_names:
                label = label_by_option.get(option_name, option_name)
                ttk.Checkbutton(group_box, text=label, variable=self.export_option_vars[option_name]).pack(anchor="w", pady=2)
        action_box = ttk.LabelFrame(settings, text="Actions", padding=10)
        action_box.pack(fill="x", pady=(12, 0))
        ttk.Button(action_box, text="Export STL Partition", command=self._save_settings).pack(fill="x")
        ttk.Button(action_box, text="Open Defect Popup List", command=self._open_defect_list_window).pack(fill="x", pady=(8, 0))

        self.defects_canvas.yview_moveto(0.0)

    def _make_spinbox(self, parent: ttk.Frame, variable, from_: float, to: float, command) -> ttk.Spinbox:
        spin = ttk.Spinbox(parent, from_=from_, to=to, width=8, textvariable=variable, command=command)
        spin.pack(side="right")
        spin.bind("<FocusOut>", lambda _event: command())
        spin.bind("<Return>", lambda _event: command())
        return spin

    def _build_overetch_box(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Over-Etch", padding=10)
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Enable", variable=self.overetch_enabled_var, command=self._apply_all_defect_settings).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Count").pack(side="left"); self._make_spinbox(row, self.overetch_count_var, 0, 24, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Severity").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=1.4, variable=self.overetch_severity_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Recovery Span (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.3, to=4.0, variable=self.overetch_recovery_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Noise / center jitter").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=0.6, variable=self.overetch_noise_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Falloff").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.overetch_falloff_var, values=("gaussian", "exponential"), state="readonly", width=14)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_all_defect_settings())
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Seed").pack(side="left"); self._make_spinbox(row, self.overetch_seed_var, 0, 999999, self._apply_all_defect_settings)
        ttk.Button(box, text="New Layout", command=lambda: self._randomize_seed_var(self.overetch_seed_var)).pack(fill="x", pady=(10, 0))

    def _build_mousebite_box(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Mousebite", padding=10)
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Enable", variable=self.mousebite_enabled_var, command=self._apply_all_defect_settings).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Count").pack(side="left"); self._make_spinbox(row, self.mousebite_count_var, 0, 24, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Blob Count").pack(side="left"); self._make_spinbox(row, self.mousebite_blob_count_var, 1, 16, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Blob Size (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.04, to=0.55, variable=self.mousebite_blob_size_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Spread (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.2, to=3.0, variable=self.mousebite_recovery_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Noise").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=0.6, variable=self.mousebite_noise_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Seed").pack(side="left"); self._make_spinbox(row, self.mousebite_seed_var, 0, 999999, self._apply_all_defect_settings)
        ttk.Button(box, text="New Layout", command=lambda: self._randomize_seed_var(self.mousebite_seed_var)).pack(fill="x", pady=(10, 0))

    def _build_underetch_box(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Under-Etch", padding=10)
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Enable", variable=self.underetch_enabled_var, command=self._apply_all_defect_settings).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Count").pack(side="left"); self._make_spinbox(row, self.underetch_count_var, 0, 24, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Severity").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=1.6, variable=self.underetch_severity_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Recovery Span (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.3, to=4.5, variable=self.underetch_recovery_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Noise").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=0.6, variable=self.underetch_noise_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Scrap Count").pack(side="left"); self._make_spinbox(row, self.underetch_blob_count_var, 0, 16, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Scrap Size (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.03, to=0.4, variable=self.underetch_blob_size_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Falloff").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.underetch_falloff_var, values=("gaussian", "exponential"), state="readonly", width=14)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_all_defect_settings())
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Seed").pack(side="left"); self._make_spinbox(row, self.underetch_seed_var, 0, 999999, self._apply_all_defect_settings)
        ttk.Button(box, text="New Layout", command=lambda: self._randomize_seed_var(self.underetch_seed_var)).pack(fill="x", pady=(10, 0))

    def _build_opencircuit_box(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Open Circuit", padding=10)
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Enable", variable=self.opencircuit_enabled_var, command=self._apply_all_defect_settings).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Count").pack(side="left"); self._make_spinbox(row, self.opencircuit_count_var, 0, 24, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Gap Size (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.05, to=1.0, variable=self.opencircuit_gap_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Cut Span (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.15, to=2.0, variable=self.opencircuit_recovery_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Noise").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=0.4, variable=self.opencircuit_noise_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Seed").pack(side="left"); self._make_spinbox(row, self.opencircuit_seed_var, 0, 999999, self._apply_all_defect_settings)
        ttk.Button(box, text="New Layout", command=lambda: self._randomize_seed_var(self.opencircuit_seed_var)).pack(fill="x", pady=(10, 0))

    def _build_shortcircuit_box(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Short Circuit", padding=10)
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Enable", variable=self.shortcircuit_enabled_var, command=self._apply_all_defect_settings).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Count").pack(side="left"); self._make_spinbox(row, self.shortcircuit_count_var, 0, 24, self._apply_all_defect_settings)
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Max Gap (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.05, to=1.5, variable=self.shortcircuit_gap_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Bridge Width (mm)").pack(anchor="w"); ttk.Scale(row, from_=0.03, to=0.7, variable=self.shortcircuit_width_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Noise").pack(anchor="w"); ttk.Scale(row, from_=0.0, to=0.4, variable=self.shortcircuit_noise_var, orient="horizontal", command=self._apply_defect_settings_from_scale).pack(fill="x")
        row = ttk.Frame(box); row.pack(fill="x", pady=(8, 0)); ttk.Label(row, text="Seed").pack(side="left"); self._make_spinbox(row, self.shortcircuit_seed_var, 0, 999999, self._apply_all_defect_settings)
        ttk.Button(box, text="New Layout", command=lambda: self._randomize_seed_var(self.shortcircuit_seed_var)).pack(fill="x", pady=(10, 0))

    def _apply_layer_visibility(self) -> None:
        for layer_name, variable in self.layer_vars.items():
            self.viewer.set_layer_visibility(layer_name, variable.get())

    def _apply_extras_visibility(self) -> None:
        self.viewer.set_extras_visibility(
            dielectric=self.show_dielectric_var.get(),
            solder_mask=self.show_solder_mask_var.get(),
            tracks=self.show_tracks_var.get(),
            pads=self.show_pads_var.get(),
            vias=self.show_vias_var.get(),
            outline=self.show_outline_var.get(),
            defect_regions=self.show_defect_regions_var.get(),
        )

    def _apply_explode_scale(self, _value: str) -> None:
        self.viewer.set_explode_scale(self.explode_var.get())

    def _apply_defect_settings_from_scale(self, _value: str) -> None:
        self._apply_all_defect_settings()

    def _apply_all_defect_settings(self) -> None:
        self.viewer.set_all_defect_settings(
            overetch=OverEtchSettings(
                enabled=self.overetch_enabled_var.get(),
                count=max(0, int(self.overetch_count_var.get())),
                severity=max(0.0, float(self.overetch_severity_var.get())),
                recovery_mm=max(0.2, float(self.overetch_recovery_var.get())),
                falloff_mode=str(self.overetch_falloff_var.get() or "gaussian"),
                noise_amount=max(0.0, float(self.overetch_noise_var.get())),
                seed=max(0, int(self.overetch_seed_var.get())),
            ),
            mousebite=MouseBiteSettings(
                enabled=self.mousebite_enabled_var.get(),
                count=max(0, int(self.mousebite_count_var.get())),
                recovery_mm=max(0.15, float(self.mousebite_recovery_var.get())),
                noise_amount=max(0.0, float(self.mousebite_noise_var.get())),
                blob_count=max(1, int(self.mousebite_blob_count_var.get())),
                blob_size_mm=max(0.02, float(self.mousebite_blob_size_var.get())),
                seed=max(0, int(self.mousebite_seed_var.get())),
            ),
            underetch=UnderEtchSettings(
                enabled=self.underetch_enabled_var.get(),
                count=max(0, int(self.underetch_count_var.get())),
                severity=max(0.0, float(self.underetch_severity_var.get())),
                recovery_mm=max(0.2, float(self.underetch_recovery_var.get())),
                falloff_mode=str(self.underetch_falloff_var.get() or "gaussian"),
                noise_amount=max(0.0, float(self.underetch_noise_var.get())),
                blob_count=max(0, int(self.underetch_blob_count_var.get())),
                blob_size_mm=max(0.02, float(self.underetch_blob_size_var.get())),
                seed=max(0, int(self.underetch_seed_var.get())),
            ),
            opencircuit=OpenCircuitSettings(
                enabled=self.opencircuit_enabled_var.get(),
                count=max(0, int(self.opencircuit_count_var.get())),
                gap_mm=max(0.05, float(self.opencircuit_gap_var.get())),
                recovery_mm=max(0.15, float(self.opencircuit_recovery_var.get())),
                noise_amount=max(0.0, float(self.opencircuit_noise_var.get())),
                seed=max(0, int(self.opencircuit_seed_var.get())),
            ),
            shortcircuit=ShortCircuitSettings(
                enabled=self.shortcircuit_enabled_var.get(),
                count=max(0, int(self.shortcircuit_count_var.get())),
                max_gap_mm=max(0.05, float(self.shortcircuit_gap_var.get())),
                bridge_width_mm=max(0.03, float(self.shortcircuit_width_var.get())),
                noise_amount=max(0.0, float(self.shortcircuit_noise_var.get())),
                seed=max(0, int(self.shortcircuit_seed_var.get())),
            ),
        )
        self._refresh_inline_defect_list()
        self._refresh_defect_listbox()

    def _randomize_seed_var(self, variable: tk.IntVar) -> None:
        variable.set(random.randint(0, 999999))
        self._apply_all_defect_settings()

    def _choose_and_load_pcb(self) -> None:
        current_path = Path(self.board_path_var.get()).expanduser() if self.board_path_var.get().strip() else SOURCE_PCB
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a KiCad PCB file",
            initialdir=str(current_path.resolve().parent if current_path.exists() else SOURCE_PCB.parent),
            filetypes=[("KiCad PCB files", "*.kicad_pcb"), ("All files", "*.*")],
        )
        if selected:
            self._load_pcb(Path(selected))

    def _reload_current_pcb(self) -> None:
        raw_path = self.board_path_var.get().strip()
        if not raw_path:
            messagebox.showerror("Missing file", "No PCB file is selected.", parent=self.root)
            return
        self._load_pcb(Path(raw_path))

    def _load_pcb(self, pcb_path: Path) -> None:
        try:
            resolved_pcb = pcb_path.expanduser().resolve()
            if resolved_pcb.suffix.lower() != ".kicad_pcb":
                raise ExportRefreshError(f"Selected file is not a .kicad_pcb file: {resolved_pcb}")
            board_key = re.sub(r"[^a-z0-9]+", "_", resolved_pcb.stem.strip().lower()).strip("_") or "board"
            copper_json = REPO_ROOT / "output" / board_key / "copper_paths.json"
            defect_output_dir = REPO_ROOT / "output" / board_key / "material_partition_defects"
            self.status_var.set(f"Loading {resolved_pcb.name}...")
            self.root.update_idletasks()
            export_copper_json(resolved_pcb, copper_json)
            model = load_board_view_model(copper_json)
            self.viewer.replace_model(model, export_output_dir=defect_output_dir)
            self._sync_vars_from_viewer()
            self._build_ui()
            self.status_var.set(f"Loaded: {resolved_pcb.name}")
        except Exception as exc:
            self.status_var.set("Load failed.")
            messagebox.showerror("PCB Load Failed", str(exc), parent=self.root)

    def _reset_explode(self) -> None:
        self.explode_var.set(1.0)
        self.viewer.set_explode_scale(1.0)

    def _export_material_partition(self) -> None:
        try:
            output_dir = self.viewer.export_current_material_partition()
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export STL partition.\n\n{exc}")
            return
        messagebox.showinfo("Export Complete", f"Exported current STL partition to:\n{output_dir}")

    def _open_settings_window(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            self.settings_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Settings")
        window.geometry("320x360")
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_settings_window)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Export Settings", font=("Georgia", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Tick the STL outputs to generate when you click Save.", wraplength=280).pack(anchor="w", pady=(4, 10))
        for option_name, label in EXPORT_OPTION_LABELS:
            ttk.Checkbutton(frame, text=label, variable=self.export_option_vars[option_name]).pack(anchor="w", pady=2)
        ttk.Button(frame, text="Save", command=self._save_settings).pack(fill="x", pady=(16, 0))
        self.settings_window = window

    def _save_settings(self) -> None:
        selected_outputs = {
            option_name
            for option_name, variable in self.export_option_vars.items()
            if variable.get()
        }
        if selected_outputs:
            try:
                output_dir = self.viewer.export_current_material_partition(selected_outputs=selected_outputs)
            except Exception as exc:
                messagebox.showerror("Save Failed", f"Could not generate STL partition.\n\n{exc}")
                return
            self._close_settings_window()
            messagebox.showinfo("Settings Saved", f"Settings saved and STLs generated in:\n{output_dir}")
            return
        self._close_settings_window()
        messagebox.showinfo("Settings Saved", "Settings saved.")

    def _close_settings_window(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
        self.settings_window = None

    def _open_defect_list_window(self) -> None:
        if self.defect_list_window is not None and self.defect_list_window.winfo_exists():
            self._refresh_defect_listbox()
            self.defect_list_window.lift()
            self.defect_list_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Defect List")
        window.geometry("420x460")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_defect_list_window)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Click a defect to focus it in the 3D viewer.").pack(anchor="w", pady=(0, 8))
        listbox = tk.Listbox(frame, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        listbox.bind("<<ListboxSelect>>", self._on_defect_list_select)
        self.defect_list_window = window
        self.defect_listbox = listbox
        self._refresh_defect_listbox()

    def _close_defect_list_window(self) -> None:
        if self.defect_list_window is not None and self.defect_list_window.winfo_exists():
            self.defect_list_window.destroy()
        self.defect_list_window = None
        self.defect_listbox = None

    def _refresh_inline_defect_list(self) -> None:
        if self.inline_defect_listbox is None:
            return
        self.inline_defect_listbox.delete(0, tk.END)
        for item in self.viewer.defect_list_items:
            self.inline_defect_listbox.insert(tk.END, defect_display_name(item.defect, item.layer_name, item.index))

    def _on_inline_defect_select(self, _event) -> None:
        self._focus_selected_inline_defect()

    def _focus_selected_inline_defect(self) -> None:
        if self.inline_defect_listbox is None:
            return
        selection = self.inline_defect_listbox.curselection()
        if not selection:
            return
        self.viewer.focus_defect_by_index(int(selection[0]))

    def _refresh_defect_listbox(self) -> None:
        if self.defect_listbox is None:
            return
        self.defect_listbox.delete(0, tk.END)
        for item in self.viewer.defect_list_items:
            self.defect_listbox.insert(tk.END, defect_display_name(item.defect, item.layer_name, item.index))

    def _on_defect_list_select(self, _event) -> None:
        if self.defect_listbox is None:
            return
        selection = self.defect_listbox.curselection()
        if not selection:
            return
        self.viewer.focus_defect_by_index(int(selection[0]))

    def _pump_viewer(self) -> None:
        try:
            self.viewer.process_events()
        except Exception:
            self._on_close()
            return
        self.root.after(16, self._pump_viewer)

    def _on_content_configure(self, _event) -> None:
        self.defects_canvas.configure(scrollregion=self.defects_canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.defects_canvas.itemconfigure(self.defects_window, width=event.width)

    def _bind_mousewheel(self) -> None:
        self.defects_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.defects_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.defects_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        self.defects_canvas.unbind_all("<MouseWheel>")
        self.defects_canvas.unbind_all("<Button-4>")
        self.defects_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if getattr(event, "delta", 0):
            step = -1 * int(event.delta / 120) if event.delta else 0
        elif getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = 0
        if step != 0:
            self.defects_canvas.yview_scroll(step, "units")

    def _on_close(self) -> None:
        try:
            self._close_defect_list_window()
            self._close_settings_window()
            self._unbind_mousewheel()
            self.viewer.close()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def board_to_centered(model: BoardViewModel, point: PointMM) -> tuple[float, float]:
    x_centered = point.x_mm - model.left_mm - (model.width_mm / 2.0)
    y_centered = -1.0 * (point.y_mm - model.top_mm - (model.height_mm / 2.0))
    return x_centered, y_centered


def build_uniform_stackup(board_thickness_mm: float, active_layers: list[str]) -> StackupDefinition:
    copper_layers = [StackupCopperLayer(name=layer_name, thickness_mm=COPPER_THICKNESS_MM) for layer_name in active_layers]
    dielectric_total_mm = max(board_thickness_mm - (len(copper_layers) * COPPER_THICKNESS_MM), 0.0)
    dielectric_count = max(len(copper_layers) - 1, 0)
    dielectric_thickness_mm = dielectric_total_mm / dielectric_count if dielectric_count else 0.0
    dielectric_layers = [
        StackupDielectricLayer(
            name=f"Dielectric {index + 1}",
            upper_layer=active_layers[index],
            lower_layer=active_layers[index + 1],
            thickness_mm=dielectric_thickness_mm,
        )
        for index in range(dielectric_count)
    ]
    return StackupDefinition(copper_layers=copper_layers, dielectric_layers=dielectric_layers)


def load_stackup_definition(payload: dict, board_thickness_mm: float, active_layers: list[str]) -> StackupDefinition:
    stackup_payload = payload.get("stackup")
    if not isinstance(stackup_payload, dict):
        return build_uniform_stackup(board_thickness_mm, active_layers)
    return StackupDefinition(
        copper_layers=[
            StackupCopperLayer(name=item["name"], thickness_mm=float(item["thickness_mm"]))
            for item in stackup_payload.get("copper_layers", [])
        ],
        dielectric_layers=[
            StackupDielectricLayer(
                name=item["name"],
                upper_layer=item["upper_layer"],
                lower_layer=item["lower_layer"],
                thickness_mm=float(item["thickness_mm"]),
            )
            for item in stackup_payload.get("dielectric_layers", [])
        ],
    )


def layer_thickness_map(stackup: StackupDefinition) -> dict[str, float]:
    return {layer.name: layer.thickness_mm for layer in stackup.copper_layers}


def dielectric_layer_map(stackup: StackupDefinition) -> dict[tuple[str, str], StackupDielectricLayer]:
    return {(layer.upper_layer, layer.lower_layer): layer for layer in stackup.dielectric_layers}


def copper_thickness_for_layer(model: BoardViewModel, layer_name: str) -> float:
    return layer_thickness_map(model.stackup).get(layer_name, COPPER_THICKNESS_MM)


def copper_bounds_for_layer(model: BoardViewModel, layer_name: str, z_map: dict[str, float]) -> tuple[float, float]:
    thickness_mm = copper_thickness_for_layer(model, layer_name)
    center_z = z_map[layer_name]
    return (center_z - (thickness_mm / 2.0), center_z + (thickness_mm / 2.0))


def build_z_map(board_thickness_mm: float, stackup: StackupDefinition, copper_layers: list[str], explode_scale: float) -> dict[str, float]:
    thickness_by_layer = layer_thickness_map(stackup)
    dielectric_by_pair = dielectric_layer_map(stackup)
    current_top = board_thickness_mm / 2.0
    z_map: dict[str, float] = {}
    for index, layer_name in enumerate(copper_layers):
        thickness_mm = thickness_by_layer.get(layer_name, COPPER_THICKNESS_MM)
        current_bottom = current_top - thickness_mm
        z_center = (current_top + current_bottom) / 2.0
        z_map[layer_name] = z_center * explode_scale
        current_top = current_bottom
        if index < len(copper_layers) - 1:
            next_layer_name = copper_layers[index + 1]
            dielectric = dielectric_by_pair.get((layer_name, next_layer_name))
            current_top -= dielectric.thickness_mm if dielectric is not None else 0.0
    return z_map


def sample_quadratic_arc(start: PointMM, mid: PointMM, end: PointMM, segments: int) -> list[PointMM]:
    points: list[PointMM] = []
    for index in range(segments + 1):
        t = index / segments
        one_minus_t = 1.0 - t
        points.append(
            PointMM(
                x_mm=(one_minus_t * one_minus_t * start.x_mm) + (2.0 * one_minus_t * t * mid.x_mm) + (t * t * end.x_mm),
                y_mm=(one_minus_t * one_minus_t * start.y_mm) + (2.0 * one_minus_t * t * mid.y_mm) + (t * t * end.y_mm),
            )
        )
    return points


def points_close(a: PointMM, b: PointMM, tolerance_mm: float = 0.08) -> bool:
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) <= tolerance_mm


def reverse_outline_item(item: OutlineData) -> OutlineData:
    return OutlineData(kind=item.kind, start_mm=item.end_mm, end_mm=item.start_mm, mid_mm=item.mid_mm)


def order_outline_items(items: list[OutlineData]) -> list[OutlineData]:
    if not items:
        return []
    remaining = list(items)
    ordered = [remaining.pop(0)]
    while remaining:
        last_end = ordered[-1].end_mm
        next_index = -1
        reverse = False
        for index, item in enumerate(remaining):
            if points_close(last_end, item.start_mm):
                next_index = index
                break
            if points_close(last_end, item.end_mm):
                next_index = index
                reverse = True
                break
        if next_index == -1:
            ordered.extend(remaining)
            break
        item = remaining.pop(next_index)
        ordered.append(reverse_outline_item(item) if reverse else item)
    return ordered


def build_board_polygon(model: BoardViewModel) -> Polygon:
    if not model.outline:
        half_w = model.width_mm / 2.0
        half_h = model.height_mm / 2.0
        return Polygon([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
    ordered = order_outline_items(model.outline)
    points: list[tuple[float, float]] = []
    for item_index, item in enumerate(ordered):
        if item.kind == "segment":
            segment_points = [item.start_mm, item.end_mm]
        elif item.kind == "arc" and item.mid_mm is not None:
            segment_points = sample_quadratic_arc(item.start_mm, item.mid_mm, item.end_mm, 16)
        else:
            continue
        for point_index, point in enumerate(segment_points):
            if item_index > 0 and point_index == 0:
                continue
            points.append(board_to_centered(model, point))
    if len(points) < 3:
        half_w = model.width_mm / 2.0
        half_h = model.height_mm / 2.0
        return Polygon([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
    return Polygon(points).buffer(0)


def make_track_mesh(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width_mm: float,
    z_center_mm: float,
    thickness_mm: float,
    color: str,
) -> Mesh:
    delta_x = end_xy[0] - start_xy[0]
    delta_y = end_xy[1] - start_xy[1]
    length = math.hypot(delta_x, delta_y)
    if length == 0:
        return Cylinder(pos=(start_xy[0], start_xy[1], z_center_mm), r=width_mm / 2.0, height=thickness_mm, axis=(0, 0, 1), res=8).c(color).alpha(0.96)

    direction_x = delta_x / length
    direction_y = delta_y / length
    normal_x = -direction_y
    normal_y = direction_x
    half_width = width_mm / 2.0
    half_thickness = thickness_mm / 2.0

    p0 = (start_xy[0] + (normal_x * half_width), start_xy[1] + (normal_y * half_width))
    p1 = (start_xy[0] - (normal_x * half_width), start_xy[1] - (normal_y * half_width))
    p2 = (end_xy[0] - (normal_x * half_width), end_xy[1] - (normal_y * half_width))
    p3 = (end_xy[0] + (normal_x * half_width), end_xy[1] + (normal_y * half_width))

    vertices = [
        [p0[0], p0[1], z_center_mm - half_thickness],
        [p1[0], p1[1], z_center_mm - half_thickness],
        [p2[0], p2[1], z_center_mm - half_thickness],
        [p3[0], p3[1], z_center_mm - half_thickness],
        [p0[0], p0[1], z_center_mm + half_thickness],
        [p1[0], p1[1], z_center_mm + half_thickness],
        [p2[0], p2[1], z_center_mm + half_thickness],
        [p3[0], p3[1], z_center_mm + half_thickness],
    ]
    faces = [
        [4, 5, 6, 7],
        [3, 2, 1, 0],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return Mesh([vertices, faces]).c(color).alpha(0.96)


def make_pad_mesh(
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    rotation_deg: float,
    shape: str,
    roundrect_radius_mm: float | None,
    color: str,
) -> Mesh:
    if shape == "rect":
        profile_xy = make_rect_profile(size_x, size_y)
    elif shape == "oval":
        profile_xy = make_oval_profile(size_x, size_y, segments=8)
    elif shape == "roundrect":
        profile_xy = make_roundrect_profile(size_x, size_y, roundrect_radius_mm, segments=4)
    else:
        profile_xy = make_rect_profile(size_x, size_y)

    return make_extruded_profile_mesh(profile_xy, center_x, center_y, center_z, size_z, rotation_deg, color)


def make_extruded_profile_mesh(
    profile_xy: list[tuple[float, float]],
    center_x: float,
    center_y: float,
    center_z: float,
    size_z: float,
    rotation_deg: float,
    color: str,
) -> Mesh:
    half_z = size_z / 2.0
    angle = math.radians(-rotation_deg)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    bottom_vertices: list[list[float]] = []
    top_vertices: list[list[float]] = []
    for local_x, local_y in profile_xy:
        x = (local_x * cos_angle) - (local_y * sin_angle) + center_x
        y = (local_x * sin_angle) + (local_y * cos_angle) + center_y
        bottom_vertices.append([x, y, center_z - half_z])
        top_vertices.append([x, y, center_z + half_z])

    vertices = bottom_vertices + top_vertices
    count = len(profile_xy)
    faces = [
        list(range(count, count * 2)),
        list(reversed(range(count))),
    ]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append([index, next_index, count + next_index, count + index])
    return Mesh([vertices, faces]).c(color).alpha(0.96)


def make_rect_profile(size_x: float, size_y: float) -> list[tuple[float, float]]:
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    return [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]


def make_oval_profile(size_x: float, size_y: float, segments: int) -> list[tuple[float, float]]:
    if abs(size_x - size_y) < 1e-6:
        radius = size_x / 2.0
        return [
            (radius * math.cos((2.0 * math.pi * index) / segments), radius * math.sin((2.0 * math.pi * index) / segments))
            for index in range(segments)
        ]
    if size_x > size_y:
        radius = size_y / 2.0
        half_straight = (size_x / 2.0) - radius
        return capsule_profile(True, half_straight, radius, segments)
    radius = size_x / 2.0
    half_straight = (size_y / 2.0) - radius
    return capsule_profile(False, half_straight, radius, segments)


def capsule_profile(horizontal: bool, half_straight: float, radius: float, segments: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    half_segments = max(segments // 2, 4)
    if horizontal:
        for index in range(half_segments + 1):
            angle = -math.pi / 2.0 + (math.pi * index / half_segments)
            points.append((half_straight + radius * math.cos(angle), radius * math.sin(angle)))
        for index in range(half_segments + 1):
            angle = math.pi / 2.0 + (math.pi * index / half_segments)
            points.append((-half_straight + radius * math.cos(angle), radius * math.sin(angle)))
    else:
        for index in range(half_segments + 1):
            angle = math.pi * index / half_segments
            points.append((radius * math.cos(angle), half_straight + radius * math.sin(angle)))
        for index in range(half_segments + 1):
            angle = math.pi + (math.pi * index / half_segments)
            points.append((radius * math.cos(angle), -half_straight + radius * math.sin(angle)))
    return deduplicate_consecutive_points(points)


def make_roundrect_profile(size_x: float, size_y: float, roundrect_radius_mm: float | None, segments: int) -> list[tuple[float, float]]:
    radius = roundrect_radius_mm if roundrect_radius_mm is not None else min(size_x, size_y) * 0.25
    radius = max(0.0, min(radius, size_x / 2.0, size_y / 2.0))
    if radius <= 1e-6:
        return make_rect_profile(size_x, size_y)
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    corners = [
        (half_x - radius, -half_y + radius, -math.pi / 2.0, 0.0),
        (half_x - radius, half_y - radius, 0.0, math.pi / 2.0),
        (-half_x + radius, half_y - radius, math.pi / 2.0, math.pi),
        (-half_x + radius, -half_y + radius, math.pi, 3.0 * math.pi / 2.0),
    ]
    points: list[tuple[float, float]] = []
    for corner_x, corner_y, start_angle, end_angle in corners:
        for index in range(segments + 1):
            angle = start_angle + ((end_angle - start_angle) * index / segments)
            points.append((corner_x + radius * math.cos(angle), corner_y + radius * math.sin(angle)))
    return deduplicate_consecutive_points(points)


def build_connected_track_paths(
    model: BoardViewModel,
    tracks: list[TrackData],
) -> list[tuple[list[tuple[float, float]], float]]:
    grouped: dict[tuple[str, float], list[TrackData]] = defaultdict(list)
    for track in tracks:
        grouped[(track.net, round(track.width_mm, 6))].append(track)

    paths: list[tuple[list[tuple[float, float]], float]] = []
    for (_, width_mm), grouped_tracks in grouped.items():
        paths.extend(build_paths_for_width_group(model, grouped_tracks, width_mm))
    return paths


def build_paths_for_width_group(
    model: BoardViewModel,
    tracks: list[TrackData],
    width_mm: float,
) -> list[tuple[list[tuple[float, float]], float]]:
    node_to_edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for track in tracks:
        start_xy = board_to_centered(model, track.start_mm)
        end_xy = board_to_centered(model, track.end_mm)
        if math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]) < 1e-6:
            continue
        edge_index = len(edges)
        edges.append((start_xy, end_xy))
        node_to_edges[point_key(start_xy)].append(edge_index)
        node_to_edges[point_key(end_xy)].append(edge_index)

    visited_edges: set[int] = set()
    paths: list[tuple[list[tuple[float, float]], float]] = []

    def trace_path(start_node: tuple[int, int], first_edge: int) -> list[tuple[float, float]]:
        polyline: list[tuple[float, float]] = []
        current_node = start_node
        current_edge = first_edge

        while True:
            visited_edges.add(current_edge)
            edge_start, edge_end = edges[current_edge]
            if point_key(edge_start) == current_node:
                next_point = edge_end
                current_point = edge_start
            else:
                next_point = edge_start
                current_point = edge_end

            if not polyline:
                polyline.append(current_point)
            polyline.append(next_point)

            next_node = point_key(next_point)
            next_candidates = [edge_index for edge_index in node_to_edges[next_node] if edge_index not in visited_edges]
            if len(node_to_edges[next_node]) != 2 or not next_candidates:
                break
            current_node = next_node
            current_edge = next_candidates[0]

        return deduplicate_consecutive_points(polyline)

    for node_key_value, connected_edges in node_to_edges.items():
        if len(connected_edges) == 2:
            continue
        for edge_index in connected_edges:
            if edge_index in visited_edges:
                continue
            polyline = trace_path(node_key_value, edge_index)
            if len(polyline) >= 2:
                paths.append((polyline, width_mm))

    for edge_index in range(len(edges)):
        if edge_index in visited_edges:
            continue
        start_point, _ = edges[edge_index]
        polyline = trace_path(point_key(start_point), edge_index)
        if len(polyline) >= 2:
            paths.append((polyline, width_mm))

    return paths


def polyline_length(polyline_xy: list[tuple[float, float]]) -> float:
    total = 0.0
    for start_point, end_point in zip(polyline_xy, polyline_xy[1:]):
        total += math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    return total


def sample_polyline_at_distance(
    polyline_xy: list[tuple[float, float]],
    distance_mm: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(polyline_xy) < 2:
        raise ValueError("Polyline must contain at least two points.")

    remaining = max(0.0, distance_mm)
    for start_point, end_point in zip(polyline_xy, polyline_xy[1:]):
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-9:
            continue
        if remaining <= segment_length:
            ratio = remaining / segment_length
            point = (start_point[0] + (dx * ratio), start_point[1] + (dy * ratio))
            tangent = (dx / segment_length, dy / segment_length)
            return point, tangent
        remaining -= segment_length

    last_start = polyline_xy[-2]
    last_end = polyline_xy[-1]
    dx = last_end[0] - last_start[0]
    dy = last_end[1] - last_start[1]
    segment_length = max(math.hypot(dx, dy), 1e-9)
    return last_end, (dx / segment_length, dy / segment_length)


def sample_polyline_window(
    polyline_xy: list[tuple[float, float]],
    center_distance_mm: float,
    half_span_mm: float,
    step_mm: float,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    if len(polyline_xy) < 2:
        return []
    total_length = polyline_length(polyline_xy)
    start_distance = max(0.0, center_distance_mm - half_span_mm)
    end_distance = min(total_length, center_distance_mm + half_span_mm)
    if end_distance <= start_distance:
        point_xy, tangent_xy = sample_polyline_at_distance(polyline_xy, center_distance_mm)
        return [(point_xy, tangent_xy, center_distance_mm)]
    sample_step = max(step_mm, 0.05)
    distances: list[float] = []
    current = start_distance
    while current < end_distance:
        distances.append(current)
        current += sample_step
    distances.append(end_distance)
    return [
        (*sample_polyline_at_distance(polyline_xy, distance_mm), distance_mm)
        for distance_mm in distances
    ]


def point_key(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0] * 1000), round(point[1] * 1000))


def deduplicate_consecutive_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 1e-6:
            continue
        cleaned.append(point)
    return cleaned


def arc_points(
    center_x: float,
    center_y: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    point_count: int,
) -> list[tuple[float, float]]:
    if point_count <= 1:
        return [(center_x + radius * math.cos(start_angle), center_y + radius * math.sin(start_angle))]
    points: list[tuple[float, float]] = []
    for index in range(point_count):
        angle = start_angle + ((end_angle - start_angle) * index / (point_count - 1))
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return points


def capsule_profile(horizontal: bool, half_straight: float, radius: float, point_count: int) -> list[tuple[float, float]]:
    if horizontal:
        right_arc = arc_points(half_straight, 0.0, radius, -math.pi / 2.0, math.pi / 2.0, point_count)
        left_arc = arc_points(-half_straight, 0.0, radius, math.pi / 2.0, 3.0 * math.pi / 2.0, point_count)
        return deduplicate_consecutive_points(right_arc + left_arc)
    top_arc = arc_points(0.0, half_straight, radius, 0.0, math.pi, point_count)
    bottom_arc = arc_points(0.0, -half_straight, radius, math.pi, 2.0 * math.pi, point_count)
    return deduplicate_consecutive_points(top_arc + bottom_arc)


def roundrect_profile(size_x: float, size_y: float, radius: float, point_count: int) -> list[tuple[float, float]]:
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    corners = [
        (half_x - radius, -half_y + radius, -math.pi / 2.0, 0.0),
        (half_x - radius, half_y - radius, 0.0, math.pi / 2.0),
        (-half_x + radius, half_y - radius, math.pi / 2.0, math.pi),
        (-half_x + radius, -half_y + radius, math.pi, 3.0 * math.pi / 2.0),
    ]
    points: list[tuple[float, float]] = []
    for corner_x, corner_y, start_angle, end_angle in corners:
        points.extend(arc_points(corner_x, corner_y, radius, start_angle, end_angle, point_count))
    return deduplicate_consecutive_points(points)


def defect_falloff_weight(mode: str, normalized_distance: float) -> float:
    normalized = max(0.0, min(normalized_distance, 1.0))
    if normalized >= 1.0:
        return 0.0
    if mode == "exponential":
        edge_value = math.exp(-1.0 / 0.28)
        current = math.exp(-normalized / 0.28)
    else:
        edge_value = math.exp(-0.5 * ((1.0 / 0.33) ** 2))
        current = math.exp(-0.5 * ((normalized / 0.33) ** 2))
    return max(0.0, (current - edge_value) / max(1e-9, 1.0 - edge_value))


def defect_color(defect: TraceDefect) -> str:
    return DEFECT_COLORS.get(defect.defect_type, "#d62828")


def defect_type_label(defect_type: str) -> str:
    return {
        "overetch": "Over-Etch",
        "underetch": "Under-Etch",
        "mousebite": "Mousebite",
        "open_circuit": "Open Circuit",
        "short_circuit": "Short Circuit",
    }.get(defect_type, defect_type.replace("_", " ").title())


def defect_display_name(defect: TraceDefect, layer_name: str, index: int) -> str:
    return (
        f"{index + 1}. {defect_type_label(defect.defect_type)} | {layer_name} | "
        f"({defect.center_x:.2f}, {defect.center_y:.2f})"
    )


def build_defect_geometry(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    if defect.defect_type == "overetch":
        return build_overetch_polygon(defect)
    if defect.defect_type == "underetch":
        return build_underetch_polygon(defect)
    if defect.defect_type == "mousebite":
        return build_mousebite_polygon(defect)
    if defect.defect_type == "open_circuit":
        return build_open_circuit_polygon(defect)
    if defect.defect_type == "short_circuit":
        return build_short_circuit_polygon(defect)
    return GeometryCollection()


def build_overetch_polygon(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    half_width = defect.track_width_mm / 2.0
    if half_width <= 1e-6 or defect.severity <= 0.0 or defect.recovery_mm <= 1e-6:
        return GeometryCollection()
    outer_margin = max(0.12, half_width * 0.6)
    max_bite = half_width * defect.severity
    rng = random.Random(defect.noise_seed)
    positive_inner: list[tuple[float, float]] = []
    positive_outer: list[tuple[float, float]] = []
    negative_inner: list[tuple[float, float]] = []
    negative_outer: list[tuple[float, float]] = []
    if defect.path_polyline_xy and defect.path_distance_mm is not None:
        samples = sample_polyline_window(
            defect.path_polyline_xy,
            defect.path_distance_mm,
            defect.recovery_mm,
            max(defect.recovery_mm / 8.0, defect.track_width_mm * 0.6),
        )
    else:
        samples = [((defect.center_x, defect.center_y), (defect.tangent_x, defect.tangent_y), defect.recovery_mm)]
    if not samples:
        return GeometryCollection()
    center_jitter_limit = half_width * defect.noise_amount * 0.9
    raw_center_offsets = [rng.uniform(-center_jitter_limit, center_jitter_limit) for _ in samples]
    smoothed_center_offsets: list[float] = []

    for index, current in enumerate(raw_center_offsets):
        previous = raw_center_offsets[index - 1] if index > 0 else current
        following = raw_center_offsets[index + 1] if index < len(raw_center_offsets) - 1 else current
        smoothed_center_offsets.append((previous + (current * 2.0) + following) / 4.0)

    center_distance = defect.path_distance_mm if defect.path_distance_mm is not None else (samples[len(samples) // 2][2] if len(samples[0]) > 2 else 0.0)
    for index, (point_xy, tangent_xy, distance_mm) in enumerate(samples):
        normal_x = -tangent_xy[1]
        normal_y = tangent_xy[0]
        normalized = abs(distance_mm - center_distance) / defect.recovery_mm
        bite_scale = defect_falloff_weight(defect.falloff_mode, normalized)
        bite_noise_scale = 1.0
        if index not in (0, len(samples) - 1):
            bite_noise_scale += defect.noise_amount * 0.25 * rng.uniform(-1.0, 1.0)
        bite = max(0.0, max_bite * bite_scale * max(0.0, bite_noise_scale))
        center_offset = smoothed_center_offsets[index]
        positive_inner.append((point_xy[0] + (normal_x * (center_offset + half_width - bite)), point_xy[1] + (normal_y * (center_offset + half_width - bite))))
        positive_outer.append((point_xy[0] + (normal_x * (center_offset + half_width + outer_margin)), point_xy[1] + (normal_y * (center_offset + half_width + outer_margin))))
        negative_inner.append((point_xy[0] + (normal_x * (center_offset - half_width + bite)), point_xy[1] + (normal_y * (center_offset - half_width + bite))))
        negative_outer.append((point_xy[0] + (normal_x * (center_offset - half_width - outer_margin)), point_xy[1] + (normal_y * (center_offset - half_width - outer_margin))))

    positive_cut = Polygon(positive_inner + list(reversed(positive_outer))).buffer(0)
    negative_cut = Polygon(negative_inner + list(reversed(negative_outer))).buffer(0)
    geometries: list[Polygon | MultiPolygon | GeometryCollection] = [positive_cut, negative_cut]

    return unary_union(geometries).buffer(0)


def build_mousebite_polygon(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    half_width = defect.track_width_mm / 2.0
    if half_width <= 1e-6 or defect.recovery_mm <= 1e-6 or defect.blob_count <= 0 or defect.blob_size_mm <= 1e-6:
        return GeometryCollection()
    rng = random.Random(defect.noise_seed)
    tangent_angle_deg = math.degrees(math.atan2(defect.tangent_y, defect.tangent_x))
    normal_x = -defect.tangent_y
    normal_y = defect.tangent_x
    jitter_limit = half_width * max(0.1, defect.noise_amount) * 0.9

    def to_world(local_x: float, local_y: float) -> tuple[float, float]:
        return (
            defect.center_x + (defect.tangent_x * local_x) + (normal_x * local_y),
            defect.center_y + (defect.tangent_y * local_x) + (normal_y * local_y),
        )

    geometries: list[Polygon | MultiPolygon | GeometryCollection] = []
    for _index in range(defect.blob_count):
        local_x = rng.uniform(-defect.recovery_mm * 0.95, defect.recovery_mm * 0.95)
        side = -1.0 if rng.random() < 0.5 else 1.0
        local_y = side * half_width * rng.uniform(0.15, 0.95) + rng.uniform(-jitter_limit, jitter_limit)
        blob_center = to_world(local_x, local_y)
        blob_radius_x = defect.blob_size_mm * rng.uniform(0.7, 1.45)
        blob_radius_y = defect.blob_size_mm * rng.uniform(0.55, 1.2)
        blob = regular_ngon(0.0, 0.0, 1.0, 7)
        blob = shapely.affinity.scale(blob, xfact=blob_radius_x, yfact=blob_radius_y, origin=(0, 0))
        blob = shapely.affinity.rotate(blob, tangent_angle_deg + rng.uniform(-55.0, 55.0), origin=(0, 0), use_radians=False)
        blob = shapely.affinity.translate(blob, xoff=blob_center[0], yoff=blob_center[1])
        geometries.append(blob.buffer(0))
    return unary_union(geometries).buffer(0)


def build_underetch_polygon(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    half_width = defect.track_width_mm / 2.0
    if half_width <= 1e-6 or defect.severity <= 0.0 or defect.recovery_mm <= 1e-6:
        return GeometryCollection()

    normal_x = -defect.tangent_y
    normal_y = defect.tangent_x
    sample_count = 17
    rng = random.Random(defect.noise_seed)
    grow_limit = max(0.02, half_width * defect.severity)

    def to_world(local_x: float, local_y: float) -> tuple[float, float]:
        return (
            defect.center_x + (defect.tangent_x * local_x) + (normal_x * local_y),
            defect.center_y + (defect.tangent_y * local_x) + (normal_y * local_y),
        )

    xs = [
        -defect.recovery_mm + ((2.0 * defect.recovery_mm) * index / (sample_count - 1))
        for index in range(sample_count)
    ]
    positive_outer: list[tuple[float, float]] = []
    negative_outer: list[tuple[float, float]] = []
    center_jitter_limit = half_width * defect.noise_amount * 0.7

    for local_x in xs:
        normalized = abs(local_x) / defect.recovery_mm
        grow_scale = defect_falloff_weight(defect.falloff_mode, normalized)
        noise_scale = 1.0 + (defect.noise_amount * 0.35 * rng.uniform(-1.0, 1.0))
        grow = max(0.0, grow_limit * grow_scale * max(0.0, noise_scale))
        center_offset = rng.uniform(-center_jitter_limit, center_jitter_limit)
        positive_outer.append(to_world(local_x, center_offset + half_width + grow))
        negative_outer.append(to_world(local_x, center_offset - half_width - grow))

    grown_outline = deduplicate_consecutive_points(positive_outer + list(reversed(negative_outer)))
    geometries: list[Polygon | MultiPolygon | GeometryCollection] = []
    if len(grown_outline) >= 3:
        geometries.append(Polygon(grown_outline).buffer(0))

    if defect.blob_count > 0 and defect.blob_size_mm > 1e-6:
        tangent_angle_deg = math.degrees(math.atan2(defect.tangent_y, defect.tangent_x))
        for _index in range(defect.blob_count):
            local_x = rng.uniform(-defect.recovery_mm, defect.recovery_mm)
            local_y = rng.uniform(-(half_width + grow_limit * 1.3), half_width + grow_limit * 1.3)
            blob_center = to_world(local_x, local_y)
            blob = regular_ngon(0.0, 0.0, 1.0, 6 + (_index % 3))
            blob = shapely.affinity.scale(
                blob,
                xfact=defect.blob_size_mm * rng.uniform(0.5, 1.3),
                yfact=defect.blob_size_mm * rng.uniform(0.45, 1.1),
                origin=(0, 0),
            )
            blob = shapely.affinity.rotate(blob, tangent_angle_deg + rng.uniform(-70.0, 70.0), origin=(0, 0), use_radians=False)
            blob = shapely.affinity.translate(blob, xoff=blob_center[0], yoff=blob_center[1])
            geometries.append(blob.buffer(0))
    return unary_union(geometries).buffer(0)


def build_open_circuit_polygon(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    half_width = defect.track_width_mm / 2.0
    if half_width <= 1e-6 or defect.severity <= 0.0 or defect.recovery_mm <= 1e-6:
        return GeometryCollection()
    normal_x = -defect.tangent_y
    normal_y = defect.tangent_x
    gap_half = defect.severity / 2.0
    span_half = max(defect.track_width_mm * 1.2, defect.recovery_mm)
    rng = random.Random(defect.noise_seed)
    center_dx = rng.uniform(-defect.noise_amount, defect.noise_amount) * gap_half
    center_dy = rng.uniform(-defect.noise_amount, defect.noise_amount) * half_width
    cx = defect.center_x + (defect.tangent_x * center_dx) + (normal_x * center_dy)
    cy = defect.center_y + (defect.tangent_y * center_dx) + (normal_y * center_dy)
    polygon = Polygon(
        [
            (cx - (defect.tangent_x * gap_half) - (normal_x * span_half), cy - (defect.tangent_y * gap_half) - (normal_y * span_half)),
            (cx + (defect.tangent_x * gap_half) - (normal_x * span_half), cy + (defect.tangent_y * gap_half) - (normal_y * span_half)),
            (cx + (defect.tangent_x * gap_half) + (normal_x * span_half), cy + (defect.tangent_y * gap_half) + (normal_y * span_half)),
            (cx - (defect.tangent_x * gap_half) + (normal_x * span_half), cy - (defect.tangent_y * gap_half) + (normal_y * span_half)),
        ]
    )
    return polygon.buffer(0)


def build_short_circuit_polygon(defect: TraceDefect) -> Polygon | MultiPolygon | GeometryCollection:
    if defect.secondary_center_x is None or defect.secondary_center_y is None:
        return GeometryCollection()
    bridge_width_mm = defect.bridge_width_mm if defect.bridge_width_mm is not None else min(defect.track_width_mm, defect.secondary_track_width_mm or defect.track_width_mm)
    rng = random.Random(defect.noise_seed)
    start = (
        defect.center_x + rng.uniform(-defect.noise_amount, defect.noise_amount) * bridge_width_mm,
        defect.center_y + rng.uniform(-defect.noise_amount, defect.noise_amount) * bridge_width_mm,
    )
    end = (
        defect.secondary_center_x + rng.uniform(-defect.noise_amount, defect.noise_amount) * bridge_width_mm,
        defect.secondary_center_y + rng.uniform(-defect.noise_amount, defect.noise_amount) * bridge_width_mm,
    )
    return LineString([start, end]).buffer(
        max(bridge_width_mm / 2.0, 0.02),
        cap_style="round",
        join_style="round",
        quad_segs=3,
    ).buffer(0)


def collect_trace_candidates(
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]],
    recovery_mm: float,
) -> list[tuple[str, list[tuple[float, float]], float, float, float, float]]:
    candidates: list[tuple[str, list[tuple[float, float]], float, float, float, float]] = []
    for layer_name, paths in track_paths_by_layer.items():
        for polyline_xy, width_mm in paths:
            length_mm = polyline_length(polyline_xy)
            endpoint_margin = max(width_mm * 1.25, recovery_mm * 0.8)
            usable_length = length_mm - (2.0 * endpoint_margin)
            if usable_length <= max(0.15, width_mm * 0.5):
                continue
            candidates.append((layer_name, polyline_xy, width_mm, length_mm, endpoint_margin, usable_length))
    return candidates


def build_trace_localized_defects(
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]],
    *,
    defect_type: str,
    enabled: bool,
    count: int,
    severity: float,
    recovery_mm: float,
    falloff_mode: str,
    noise_amount: float,
    blob_count: int,
    blob_size_mm: float,
    seed: int,
) -> dict[str, list[TraceDefect]]:
    if not enabled or count <= 0:
        return {}
    candidates = collect_trace_candidates(track_paths_by_layer, recovery_mm)
    if not candidates:
        return {}
    rng = random.Random(seed)
    total_usable = sum(item[5] for item in candidates)
    defects_by_layer: dict[str, list[TraceDefect]] = defaultdict(list)
    accepted_points: list[tuple[str, float, float]] = []
    max_attempts = max(count * 20, 40)
    attempts = 0

    while sum(len(items) for items in defects_by_layer.values()) < count and attempts < max_attempts:
        attempts += 1
        pick = rng.random() * total_usable
        chosen = candidates[-1]
        running = 0.0
        for candidate in candidates:
            running += candidate[5]
            if pick <= running:
                chosen = candidate
                break

        layer_name, polyline_xy, width_mm, _length_mm, endpoint_margin, usable_length = chosen
        sample_distance = endpoint_margin + (rng.random() * usable_length)
        center_xy, tangent_xy = sample_polyline_at_distance(polyline_xy, sample_distance)

        min_spacing = max(width_mm * 2.0, recovery_mm * 0.85)
        if any(
            existing_layer == layer_name and math.hypot(center_xy[0] - existing_x, center_xy[1] - existing_y) < min_spacing
            for existing_layer, existing_x, existing_y in accepted_points
        ):
            continue

        defect = TraceDefect(
            defect_type=defect_type,
            layer_name=layer_name,
            center_x=center_xy[0],
            center_y=center_xy[1],
            tangent_x=tangent_xy[0],
            tangent_y=tangent_xy[1],
            track_width_mm=width_mm,
            recovery_mm=recovery_mm,
            severity=severity,
            falloff_mode=falloff_mode,
            noise_amount=noise_amount,
            noise_seed=rng.randint(0, 2_000_000_000),
            blob_count=blob_count,
            blob_size_mm=blob_size_mm,
            path_polyline_xy=list(polyline_xy),
            path_distance_mm=sample_distance,
        )
        defects_by_layer[layer_name].append(defect)
        accepted_points.append((layer_name, center_xy[0], center_xy[1]))

    return dict(defects_by_layer)


def merge_defect_maps(*maps: dict[str, list[TraceDefect]]) -> dict[str, list[TraceDefect]]:
    merged: dict[str, list[TraceDefect]] = defaultdict(list)
    for defect_map in maps:
        for layer_name, defects in defect_map.items():
            merged[layer_name].extend(defects)
    return dict(merged)


def build_short_circuit_defects_by_layer(
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]],
    settings: ShortCircuitSettings,
) -> dict[str, list[TraceDefect]]:
    if not settings.enabled or settings.count <= 0:
        return {}
    rng = random.Random(settings.seed)
    defects_by_layer: dict[str, list[TraceDefect]] = defaultdict(list)
    for layer_name, paths in track_paths_by_layer.items():
        if len(paths) < 2:
            continue
        samples: list[tuple[int, tuple[float, float], tuple[float, float], float]] = []
        for path_index, (polyline_xy, width_mm) in enumerate(paths):
            sample_total = max(2, int(polyline_length(polyline_xy) / 0.9))
            for sample_index in range(sample_total):
                distance = (polyline_length(polyline_xy) * sample_index) / max(1, sample_total - 1)
                center_xy, tangent_xy = sample_polyline_at_distance(polyline_xy, distance)
                samples.append((path_index, center_xy, tangent_xy, width_mm))
        candidate_pairs: list[tuple[float, TraceDefect]] = []
        for left_index in range(len(samples)):
            left_path_index, left_center, left_tangent, left_width = samples[left_index]
            for right_index in range(left_index + 1, len(samples)):
                right_path_index, right_center, _right_tangent, right_width = samples[right_index]
                if left_path_index == right_path_index:
                    continue
                gap_mm = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
                if gap_mm <= 0.02 or gap_mm > settings.max_gap_mm:
                    continue
                candidate_pairs.append(
                    (
                        gap_mm,
                        TraceDefect(
                            defect_type="short_circuit",
                            layer_name=layer_name,
                            center_x=left_center[0],
                            center_y=left_center[1],
                            tangent_x=left_tangent[0],
                            tangent_y=left_tangent[1],
                            track_width_mm=left_width,
                            recovery_mm=gap_mm,
                            severity=gap_mm,
                            falloff_mode="gaussian",
                            noise_amount=settings.noise_amount,
                            noise_seed=rng.randint(0, 2_000_000_000),
                            secondary_center_x=right_center[0],
                            secondary_center_y=right_center[1],
                            secondary_track_width_mm=right_width,
                            bridge_width_mm=settings.bridge_width_mm,
                        ),
                    )
                )
        candidate_pairs.sort(key=lambda item: item[0])
        accepted: list[tuple[float, float]] = []
        for _gap, defect in candidate_pairs:
            if len(defects_by_layer[layer_name]) >= settings.count:
                break
            if any(math.hypot(defect.center_x - x, defect.center_y - y) < settings.max_gap_mm for x, y in accepted):
                continue
            defects_by_layer[layer_name].append(defect)
            accepted.append((defect.center_x, defect.center_y))
    return dict(defects_by_layer)


def build_all_defects_by_layer(
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]],
    overetch_settings: OverEtchSettings,
    mousebite_settings: MouseBiteSettings,
    underetch_settings: UnderEtchSettings,
    opencircuit_settings: OpenCircuitSettings,
    shortcircuit_settings: ShortCircuitSettings,
) -> dict[str, list[TraceDefect]]:
    return merge_defect_maps(
        build_trace_localized_defects(
            track_paths_by_layer,
            defect_type="overetch",
            enabled=overetch_settings.enabled,
            count=overetch_settings.count,
            severity=overetch_settings.severity,
            recovery_mm=overetch_settings.recovery_mm,
            falloff_mode=overetch_settings.falloff_mode,
            noise_amount=overetch_settings.noise_amount,
            blob_count=0,
            blob_size_mm=0.0,
            seed=overetch_settings.seed,
        ),
        build_trace_localized_defects(
            track_paths_by_layer,
            defect_type="mousebite",
            enabled=mousebite_settings.enabled,
            count=mousebite_settings.count,
            severity=1.0,
            recovery_mm=mousebite_settings.recovery_mm,
            falloff_mode="gaussian",
            noise_amount=mousebite_settings.noise_amount,
            blob_count=mousebite_settings.blob_count,
            blob_size_mm=mousebite_settings.blob_size_mm,
            seed=mousebite_settings.seed,
        ),
        build_trace_localized_defects(
            track_paths_by_layer,
            defect_type="underetch",
            enabled=underetch_settings.enabled,
            count=underetch_settings.count,
            severity=underetch_settings.severity,
            recovery_mm=underetch_settings.recovery_mm,
            falloff_mode=underetch_settings.falloff_mode,
            noise_amount=underetch_settings.noise_amount,
            blob_count=underetch_settings.blob_count,
            blob_size_mm=underetch_settings.blob_size_mm,
            seed=underetch_settings.seed,
        ),
        build_trace_localized_defects(
            track_paths_by_layer,
            defect_type="open_circuit",
            enabled=opencircuit_settings.enabled,
            count=opencircuit_settings.count,
            severity=opencircuit_settings.gap_mm,
            recovery_mm=opencircuit_settings.recovery_mm,
            falloff_mode="gaussian",
            noise_amount=opencircuit_settings.noise_amount,
            blob_count=0,
            blob_size_mm=0.0,
            seed=opencircuit_settings.seed,
        ),
        build_short_circuit_defects_by_layer(track_paths_by_layer, shortcircuit_settings),
    )


def make_polyline_trace_mesh(
    polyline_xy: list[tuple[float, float]],
    width_mm: float,
    z_center_mm: float,
    thickness_mm: float,
    color: str,
) -> Mesh:
    if len(polyline_xy) == 2:
        return make_track_mesh(polyline_xy[0], polyline_xy[1], width_mm, z_center_mm, thickness_mm, color)

    left_points, right_points = offset_polyline(polyline_xy, width_mm / 2.0)
    return make_ribbon_prism_mesh(left_points, right_points, z_center_mm, thickness_mm, color)


def regular_ngon(center_x: float, center_y: float, radius: float, sides: int) -> Polygon:
    points = []
    for index in range(sides):
        angle = (2.0 * math.pi * index) / sides
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return Polygon(points)


def via_drill_radius(via: ViaData) -> float:
    return via.drill_mm / 2.0


def via_barrel_outer_radius(via: ViaData) -> float:
    inner_radius = via_drill_radius(via)
    requested_outer_radius = inner_radius + VIA_PLATING_THICKNESS_MM
    max_allowed_radius = min((diameter / 2.0) for diameter in via.diameter_by_layer_mm.values())
    outer_radius = min(requested_outer_radius, max_allowed_radius)
    if outer_radius <= inner_radius:
        outer_radius = inner_radius + min(VIA_PLATING_THICKNESS_MM, 0.01)
    return outer_radius


def make_via_drill_polygon(center_x: float, center_y: float, via: ViaData) -> Polygon:
    return regular_ngon(center_x, center_y, via_drill_radius(via), VIA_SIDE_COUNT)


def make_via_land_polygon(center_x: float, center_y: float, via: ViaData, layer_name: str) -> Polygon:
    return regular_ngon(center_x, center_y, via.diameter_by_layer_mm[layer_name] / 2.0, VIA_SIDE_COUNT)


def make_via_barrel_ring_polygon(center_x: float, center_y: float, outer_radius: float, inner_radius: float) -> Polygon:
    return regular_ngon(center_x, center_y, outer_radius, VIA_SIDE_COUNT).difference(
        regular_ngon(center_x, center_y, inner_radius, VIA_SIDE_COUNT)
    ).buffer(0)


def create_layer_copper_polygon(
    model: BoardViewModel,
    layer_name: str,
    track_paths: list[tuple[list[tuple[float, float]], float]] | None = None,
    defects: list[TraceDefect] | None = None,
    include_zones: bool = True,
    include_tracks: bool = True,
    include_pads: bool = True,
):
    geometries = []

    if include_zones:
        zone_geometry = create_zone_geometry(model, layer_name)
        if not zone_geometry.is_empty:
            geometries.append(zone_geometry)

    if include_tracks:
        if track_paths is None:
            layer_tracks = [track for track in model.tracks if track.layer == layer_name]
            track_paths = build_connected_track_paths(model, layer_tracks)
        track_geometries = [
            LineString(polyline_xy).buffer(
                width_mm / 2.0,
                cap_style="flat",
                join_style="mitre",
                mitre_limit=2.0,
                quad_segs=4,
            )
            for polyline_xy, width_mm in track_paths
        ]
        if track_geometries:
            track_geometry = unary_union(track_geometries).buffer(0)
            if defects:
                subtract_geometries = []
                add_geometries = []
                for defect in defects:
                    if defect.defect_type == "overetch":
                        geometry = build_overetch_polygon(defect)
                        if not geometry.is_empty:
                            subtract_geometries.append(geometry)
                    elif defect.defect_type == "mousebite":
                        geometry = build_mousebite_polygon(defect)
                        if not geometry.is_empty:
                            subtract_geometries.append(geometry)
                    elif defect.defect_type == "open_circuit":
                        geometry = build_open_circuit_polygon(defect)
                        if not geometry.is_empty:
                            subtract_geometries.append(geometry)
                    elif defect.defect_type == "underetch":
                        geometry = build_underetch_polygon(defect)
                        if not geometry.is_empty:
                            add_geometries.append(geometry)
                    elif defect.defect_type == "short_circuit":
                        geometry = build_short_circuit_polygon(defect)
                        if not geometry.is_empty:
                            add_geometries.append(geometry)
                if subtract_geometries:
                    track_geometry = track_geometry.difference(unary_union(subtract_geometries)).buffer(0)
                if add_geometries:
                    track_geometry = unary_union([track_geometry, unary_union(add_geometries)]).buffer(0)
            if not track_geometry.is_empty:
                geometries.append(track_geometry)

    if include_pads:
        for pad in model.pads:
            if pad.layer != layer_name:
                continue
            geometries.append(create_pad_polygon(model, pad))
    for via in model.vias:
        if layer_name not in via.diameter_by_layer_mm:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
        geometries.append(make_via_land_polygon(center_x, center_y, via, layer_name))
    if not geometries:
        return GeometryCollection()
    copper = unary_union(geometries)
    holes = []
    seen_pad_holes: set[tuple[str, str, int, int]] = set()
    for pad in model.pads:
        if pad.layer != layer_name:
            continue
        pad_key = pad_physical_key(pad)
        if pad_key in seen_pad_holes:
            continue
        seen_pad_holes.add(pad_key)
        drill_polygon = create_pad_drill_polygon(model, pad)
        if drill_polygon is not None:
            holes.append(drill_polygon)
    for via in model.vias:
        if layer_name not in via.diameter_by_layer_mm:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
        holes.append(make_via_drill_polygon(center_x, center_y, via))
    if holes:
        copper = copper.difference(unary_union(holes))
    return copper.buffer(0)


def iter_via_gap_spans(model: BoardViewModel, via: ViaData, active_layers: list[str], z_map: dict[str, float]) -> list[tuple[float, float]]:
    layer_indices = [index for index, layer_name in enumerate(active_layers) if layer_name in via.diameter_by_layer_mm]
    if len(layer_indices) < 2:
        return []
    spans: list[tuple[float, float]] = []
    for upper_index, lower_index in zip(layer_indices, layer_indices[1:]):
        upper_layer = active_layers[upper_index]
        lower_layer = active_layers[lower_index]
        z_top = copper_bounds_for_layer(model, upper_layer, z_map)[0]
        z_bottom = copper_bounds_for_layer(model, lower_layer, z_map)[1]
        if z_top > z_bottom:
            spans.append((z_bottom, z_top))
    return spans


def create_dielectric_slab_polygon(
    board_polygon: Polygon,
    model: BoardViewModel,
    upper_layer_index: int,
    lower_layer_index: int,
    clearance_mm: float,
) -> Polygon | MultiPolygon | GeometryCollection:
    subtractors: list[Any] = []
    for via in model.vias:
        via_layers = [layer for layer in model.active_layers if layer in via.diameter_by_layer_mm]
        if not via_layers:
            continue
        first_index = model.active_layers.index(via_layers[0])
        last_index = model.active_layers.index(via_layers[-1])
        if first_index <= upper_layer_index and last_index >= lower_layer_index:
            center_x, center_y = board_to_centered(model, via.position_mm)
            outer_radius = via_barrel_outer_radius(via)
            subtractors.append(regular_ngon(center_x, center_y, outer_radius + clearance_mm, VIA_SIDE_COUNT))
    seen_pad_holes: set[tuple[str, str, int, int]] = set()
    for pad in model.pads:
        pad_key = pad_physical_key(pad)
        if pad_key in seen_pad_holes:
            continue
        seen_pad_holes.add(pad_key)
        drill_polygon = create_pad_drill_polygon(model, pad)
        if drill_polygon is not None:
            subtractors.append(drill_polygon.buffer(clearance_mm))
    if not subtractors:
        return board_polygon
    return board_polygon.difference(unary_union(subtractors)).buffer(0)


def mask_layer_order(model: BoardViewModel, side: str) -> str | None:
    if side == "top":
        return "F.Cu" if "F.Cu" in model.active_layers else (model.active_layers[0] if model.active_layers else None)
    if side == "bottom":
        return "B.Cu" if "B.Cu" in model.active_layers else (model.active_layers[-1] if model.active_layers else None)
    return None


def create_solder_mask_geometry(
    model: BoardViewModel,
    side: str,
    board_polygon: Polygon,
    track_paths_by_layer: dict[str, list[tuple[list[tuple[float, float]], float]]] | None = None,
    defects_by_layer: dict[str, list[TraceDefect]] | None = None,
) -> Polygon | MultiPolygon | GeometryCollection:
    layer_name = mask_layer_order(model, side)
    if layer_name is None:
        return GeometryCollection()
    copper_geometry = create_layer_copper_polygon(
        model=model,
        layer_name=layer_name,
        track_paths=None if track_paths_by_layer is None else track_paths_by_layer.get(layer_name, []),
        defects=None if defects_by_layer is None else defects_by_layer.get(layer_name, []),
        include_zones=True,
        include_tracks=True,
        include_pads=True,
    ).intersection(board_polygon)
    if copper_geometry.is_empty:
        return board_polygon
    return board_polygon.difference(copper_geometry).buffer(0)


def create_pad_polygon(model: BoardViewModel, pad: PadData) -> Polygon:
    center_x, center_y = board_to_centered(model, pad.center_mm)
    if pad.shape == "circle":
        polygon = regular_ngon(0.0, 0.0, max(pad.size_x_mm, pad.size_y_mm) / 2.0, PAD_CIRCLE_SIDE_COUNT)
    elif pad.shape == "oval":
        if abs(pad.size_x_mm - pad.size_y_mm) < 1e-6:
            polygon = regular_ngon(0.0, 0.0, pad.size_x_mm / 2.0, PAD_CIRCLE_SIDE_COUNT)
        elif pad.size_x_mm > pad.size_y_mm:
            radius = pad.size_y_mm / 2.0
            half_straight = (pad.size_x_mm / 2.0) - radius
            polygon = Polygon(capsule_profile(True, half_straight, radius, PAD_CURVE_POINT_COUNT))
        else:
            radius = pad.size_x_mm / 2.0
            half_straight = (pad.size_y_mm / 2.0) - radius
            polygon = Polygon(capsule_profile(False, half_straight, radius, PAD_CURVE_POINT_COUNT))
    elif pad.shape == "roundrect":
        radius = pad.roundrect_radius_mm if pad.roundrect_radius_mm is not None else min(pad.size_x_mm, pad.size_y_mm) * 0.25
        radius = max(0.0, min(radius, pad.size_x_mm / 2.0, pad.size_y_mm / 2.0))
        if radius <= 1e-6:
            polygon = Polygon(
                [
                    (-pad.size_x_mm / 2.0, -pad.size_y_mm / 2.0),
                    (pad.size_x_mm / 2.0, -pad.size_y_mm / 2.0),
                    (pad.size_x_mm / 2.0, pad.size_y_mm / 2.0),
                    (-pad.size_x_mm / 2.0, pad.size_y_mm / 2.0),
                ]
            )
        else:
            polygon = Polygon(roundrect_profile(pad.size_x_mm, pad.size_y_mm, radius, PAD_CURVE_POINT_COUNT))
    else:
        polygon = Polygon(
            [
                (-pad.size_x_mm / 2.0, -pad.size_y_mm / 2.0),
                (pad.size_x_mm / 2.0, -pad.size_y_mm / 2.0),
                (pad.size_x_mm / 2.0, pad.size_y_mm / 2.0),
                (-pad.size_x_mm / 2.0, pad.size_y_mm / 2.0),
            ]
        )
    polygon = shapely.affinity.rotate(polygon, -pad.rotation_deg, origin=(0, 0), use_radians=False)
    polygon = shapely.affinity.translate(polygon, center_x, center_y)
    return polygon


def create_pad_drill_polygon(model: BoardViewModel, pad: PadData) -> Polygon | None:
    if pad.drill_x_mm <= 1e-6 or pad.drill_y_mm <= 1e-6:
        return None
    center_x, center_y = board_to_centered(model, pad.center_mm)
    if abs(pad.drill_x_mm - pad.drill_y_mm) < 1e-6:
        polygon = regular_ngon(0.0, 0.0, pad.drill_x_mm / 2.0, PAD_CIRCLE_SIDE_COUNT)
    elif pad.drill_x_mm > pad.drill_y_mm:
        radius = pad.drill_y_mm / 2.0
        half_straight = (pad.drill_x_mm / 2.0) - radius
        polygon = Polygon(capsule_profile(True, half_straight, radius, PAD_CURVE_POINT_COUNT))
    else:
        radius = pad.drill_x_mm / 2.0
        half_straight = (pad.drill_y_mm / 2.0) - radius
        polygon = Polygon(capsule_profile(False, half_straight, radius, PAD_CURVE_POINT_COUNT))
    polygon = shapely.affinity.rotate(polygon, -pad.rotation_deg, origin=(0, 0), use_radians=False)
    polygon = shapely.affinity.translate(polygon, center_x, center_y)
    return polygon


def zone_contour_to_polygon(model: BoardViewModel, contour: ZoneContourData) -> Polygon | None:
    exterior = [board_to_centered(model, point) for point in contour.exterior_mm]
    holes = [[board_to_centered(model, point) for point in hole] for hole in contour.holes_mm]
    if len(exterior) < 3:
        return None
    polygon = Polygon(exterior, holes)
    if polygon.is_empty:
        return None
    return polygon.buffer(0)


def create_zone_geometry(model: BoardViewModel, layer_name: str) -> Polygon | MultiPolygon | GeometryCollection:
    polygons: list[Polygon] = []
    for zone in model.zones:
        if zone.layer != layer_name:
            continue
        for contour in zone.contours:
            polygon = zone_contour_to_polygon(model, contour)
            if polygon is not None and not polygon.is_empty:
                polygons.append(polygon)
    if not polygons:
        return GeometryCollection()
    return unary_union(polygons).buffer(0)


def iter_polygons(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for item in geometry.geoms:
            polygons.extend(iter_polygons(item))
        return polygons
    return []


def extrude_polygon_to_vedo_mesh(polygon: Polygon, z_bottom: float, height: float, color: str) -> Mesh | None:
    if polygon.area <= 1e-8:
        return None
    mesh = trimesh.creation.extrude_polygon(polygon, height)
    mesh.apply_translation((0.0, 0.0, z_bottom))
    return Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(color).alpha(0.96)


def extrude_geometry(geometry: Any, height_mm: float, z_bottom_mm: float) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    for polygon in iter_polygons(geometry):
        if polygon.area <= 1e-8:
            continue
        mesh = trimesh.creation.extrude_polygon(polygon, height_mm)
        mesh.apply_translation((0.0, 0.0, z_bottom_mm))
        for part in mesh.split(only_watertight=False):
            if len(part.faces) < 4:
                continue
            if not part.is_watertight:
                continue
            meshes.append(trimesh.Trimesh(vertices=part.vertices.copy(), faces=part.faces.copy(), process=False))
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def sequential_union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    working = [
        trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        for mesh in meshes
        if mesh is not None and len(mesh.faces) > 0
    ]
    if not working:
        return None
    result = working[0]
    for mesh in working[1:]:
        result = trimesh.boolean.union([result, mesh], engine="manifold")
        if result is None:
            raise RuntimeError("Boolean union returned no mesh while building copper_all.stl.")
        result = trimesh.Trimesh(vertices=result.vertices.copy(), faces=result.faces.copy(), process=False)
    return result


def boolean_difference(base_mesh: trimesh.Trimesh, cutter_mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    result = trimesh.boolean.difference([base_mesh, cutter_mesh], engine="manifold")
    if result is None:
        return None
    return trimesh.Trimesh(vertices=result.vertices.copy(), faces=result.faces.copy(), process=False)


def make_tube_mesh(
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    sides: int,
    angle_offset_deg: float = 0.0,
    cap_ends: bool = True,
) -> trimesh.Trimesh | None:
    if sides < 3 or outer_radius <= inner_radius or z_max <= z_min:
        return None
    bottom_outer: list[list[float]] = []
    bottom_inner: list[list[float]] = []
    top_outer: list[list[float]] = []
    top_inner: list[list[float]] = []
    angle_offset = math.radians(angle_offset_deg)
    for index in range(sides):
        angle = angle_offset + ((2.0 * math.pi * index) / sides)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        ox = center_x + (outer_radius * cos_angle)
        oy = center_y + (outer_radius * sin_angle)
        ix = center_x + (inner_radius * cos_angle)
        iy = center_y + (inner_radius * sin_angle)
        bottom_outer.append([ox, oy, z_min])
        bottom_inner.append([ix, iy, z_min])
        top_outer.append([ox, oy, z_max])
        top_inner.append([ix, iy, z_max])
    vertices = bottom_outer + bottom_inner + top_outer + top_inner

    def bo(index: int) -> int:
        return index

    def bi(index: int) -> int:
        return sides + index

    def to(index: int) -> int:
        return (2 * sides) + index

    def ti(index: int) -> int:
        return (3 * sides) + index

    faces: list[list[int]] = []
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append([bo(index), bo(next_index), to(next_index)])
        faces.append([bo(index), to(next_index), to(index)])
        faces.append([bi(index), ti(next_index), bi(next_index)])
        faces.append([bi(index), ti(index), ti(next_index)])
        if cap_ends:
            faces.append([to(index), to(next_index), ti(next_index)])
            faces.append([to(index), ti(next_index), ti(index)])
            faces.append([bo(index), bi(next_index), bo(next_index)])
            faces.append([bo(index), bi(index), bi(next_index)])
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def make_hollow_via_mesh(
    center_x: float,
    center_y: float,
    z_min: float,
    z_max: float,
    outer_radius: float,
    inner_radius: float,
    color: str,
) -> Mesh:
    ring = make_via_barrel_ring_polygon(center_x, center_y, outer_radius, inner_radius)
    mesh = trimesh.creation.extrude_polygon(ring, z_max - z_min)
    mesh.apply_translation((0.0, 0.0, z_min))
    return Mesh([mesh.vertices.tolist(), mesh.faces.tolist()]).c(color).alpha(0.72)


def create_via_barrel_mesh(
    model: BoardViewModel,
    z_map: dict[str, float],
    cap_ends: bool = True,
    overlap_mm: float = BARREL_LAYER_OVERLAP_MM,
) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    for via in model.vias:
        layer_names = [layer for layer in model.active_layers if layer in via.diameter_by_layer_mm]
        if len(layer_names) < 2:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
        layer_indices = [model.active_layers.index(layer_name) for layer_name in layer_names]
        for upper_index, lower_index in zip(layer_indices, layer_indices[1:]):
            upper_layer = model.active_layers[upper_index]
            lower_layer = model.active_layers[lower_index]
            z_min = copper_bounds_for_layer(model, lower_layer, z_map)[1] - overlap_mm
            z_max = copper_bounds_for_layer(model, upper_layer, z_map)[0] + overlap_mm
            mesh = make_tube_mesh(
                center_x=center_x,
                center_y=center_y,
                inner_radius=via_drill_radius(via),
                outer_radius=via_barrel_outer_radius(via),
                z_min=z_min,
                z_max=z_max,
                sides=VIA_SIDE_COUNT,
                cap_ends=cap_ends,
            )
            if mesh is not None:
                meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def create_pad_barrel_ring_polygon(model: BoardViewModel, pad: PadData) -> Polygon | None:
    drill_polygon = create_pad_drill_polygon(model, pad)
    if drill_polygon is None:
        return None
    outer_polygon = drill_polygon.buffer(COPPER_THICKNESS_MM, cap_style="round", join_style="round", quad_segs=4)
    pad_polygon = create_pad_polygon(model, pad)
    ring_polygon = outer_polygon.intersection(pad_polygon).difference(drill_polygon).buffer(0)
    if ring_polygon.is_empty:
        return None
    return ring_polygon


def create_pad_barrel_mesh(
    model: BoardViewModel,
    z_map: dict[str, float],
    cap_ends: bool = True,
    overlap_mm: float = BARREL_LAYER_OVERLAP_MM,
) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    seen_pads: set[tuple[str, str, int, int]] = set()
    for pad in model.pads:
        pad_key = pad_physical_key(pad)
        if pad_key in seen_pads:
            continue
        seen_pads.add(pad_key)
        if pad.drill_x_mm <= 1e-6 or pad.drill_y_mm <= 1e-6:
            continue
        layer_names = sorted(
            [candidate.layer for candidate in model.pads if pad_physical_key(candidate) == pad_key],
            key=copper_layer_sort_key,
        )
        if not layer_names:
            continue
        ring_polygon = create_pad_barrel_ring_polygon(model, pad)
        if ring_polygon is None:
            continue
        layer_indices = [model.active_layers.index(layer_name) for layer_name in layer_names]
        if len(layer_indices) < 2:
            continue
        for upper_index, lower_index in zip(layer_indices, layer_indices[1:]):
            upper_layer = model.active_layers[upper_index]
            lower_layer = model.active_layers[lower_index]
            z_min = copper_bounds_for_layer(model, lower_layer, z_map)[1] - overlap_mm
            z_max = copper_bounds_for_layer(model, upper_layer, z_map)[0] + overlap_mm
            mesh = extrude_geometry(ring_polygon, height_mm=z_max - z_min, z_bottom_mm=z_min)
            if mesh is not None:
                meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def create_via_air_mesh(model: BoardViewModel, z_map: dict[str, float]) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    for via in model.vias:
        layer_names = [layer for layer in model.active_layers if layer in via.diameter_by_layer_mm]
        if not layer_names:
            continue
        bounds = [copper_bounds_for_layer(model, layer, z_map) for layer in layer_names]
        z_min = min(bound[0] for bound in bounds)
        z_max = max(bound[1] for bound in bounds)
        if z_max <= z_min:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
        hole = make_via_drill_polygon(center_x, center_y, via)
        mesh = trimesh.creation.extrude_polygon(hole, z_max - z_min)
        mesh.apply_translation((0.0, 0.0, z_min))
        meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def create_pad_air_mesh(model: BoardViewModel, z_map: dict[str, float]) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    seen_pads: set[tuple[str, str, int, int]] = set()
    for pad in model.pads:
        pad_key = pad_physical_key(pad)
        if pad_key in seen_pads:
            continue
        seen_pads.add(pad_key)
        if pad.drill_x_mm <= 1e-6 or pad.drill_y_mm <= 1e-6:
            continue
        layer_names = sorted(
            [candidate.layer for candidate in model.pads if pad_physical_key(candidate) == pad_key],
            key=copper_layer_sort_key,
        )
        if len(layer_names) < 2:
            continue
        z_bounds = [copper_bounds_for_layer(model, layer_name, z_map) for layer_name in layer_names]
        z_min = min(bounds[0] for bounds in z_bounds)
        z_max = max(bounds[1] for bounds in z_bounds)
        if z_max <= z_min:
            continue
        hole = create_pad_drill_polygon(model, pad)
        if hole is None:
            continue
        mesh = trimesh.creation.extrude_polygon(hole, z_max - z_min)
        mesh.apply_translation((0.0, 0.0, z_min))
        meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def offset_polyline(
    polyline_xy: list[tuple[float, float]],
    half_width: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    directions: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []
    for index in range(len(polyline_xy) - 1):
        dx = polyline_xy[index + 1][0] - polyline_xy[index][0]
        dy = polyline_xy[index + 1][1] - polyline_xy[index][1]
        length = math.hypot(dx, dy)
        if length == 0:
            directions.append((1.0, 0.0))
            normals.append((0.0, 1.0))
        else:
            direction = (dx / length, dy / length)
            directions.append(direction)
            normals.append((-direction[1], direction[0]))

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    for index, point in enumerate(polyline_xy):
        if index == 0:
            normal = normals[0]
            left_points.append((point[0] + (normal[0] * half_width), point[1] + (normal[1] * half_width)))
            right_points.append((point[0] - (normal[0] * half_width), point[1] - (normal[1] * half_width)))
            continue

        if index == len(polyline_xy) - 1:
            normal = normals[-1]
            left_points.append((point[0] + (normal[0] * half_width), point[1] + (normal[1] * half_width)))
            right_points.append((point[0] - (normal[0] * half_width), point[1] - (normal[1] * half_width)))
            continue

        prev_normal = normals[index - 1]
        next_normal = normals[index]
        left_points.append(intersect_offset_lines(polyline_xy[index - 1], directions[index - 1], prev_normal, polyline_xy[index], directions[index], next_normal, half_width))
        right_points.append(intersect_offset_lines(polyline_xy[index - 1], directions[index - 1], prev_normal, polyline_xy[index], directions[index], next_normal, -half_width))

    return left_points, right_points


def intersect_offset_lines(
    prev_point: tuple[float, float],
    prev_direction: tuple[float, float],
    prev_normal: tuple[float, float],
    point: tuple[float, float],
    next_direction: tuple[float, float],
    next_normal: tuple[float, float],
    offset_distance: float,
) -> tuple[float, float]:
    a1 = (
        prev_point[0] + (prev_normal[0] * offset_distance),
        prev_point[1] + (prev_normal[1] * offset_distance),
    )
    a2 = (
        point[0] + (prev_normal[0] * offset_distance),
        point[1] + (prev_normal[1] * offset_distance),
    )
    b1 = (
        point[0] + (next_normal[0] * offset_distance),
        point[1] + (next_normal[1] * offset_distance),
    )
    b2 = (
        point[0] + (next_direction[0] * 10.0) + (next_normal[0] * offset_distance),
        point[1] + (next_direction[1] * 10.0) + (next_normal[1] * offset_distance),
    )
    return line_intersection(a1, a2, b1, b2) or b1


def line_intersection(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> tuple[float, float] | None:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denominator) < 1e-9:
        return None
    determinant_a = (x1 * y2) - (y1 * x2)
    determinant_b = (x3 * y4) - (y3 * x4)
    px = ((determinant_a * (x3 - x4)) - ((x1 - x2) * determinant_b)) / denominator
    py = ((determinant_a * (y3 - y4)) - ((y1 - y2) * determinant_b)) / denominator
    return (px, py)


def pad_physical_key(pad: PadData) -> tuple[str, str, int, int]:
    return (
        pad.reference,
        pad.pad_number,
        round(pad.center_mm.x_mm * 1000.0),
        round(pad.center_mm.y_mm * 1000.0),
    )


def make_ribbon_prism_mesh(
    left_points: list[tuple[float, float]],
    right_points: list[tuple[float, float]],
    z_center_mm: float,
    thickness_mm: float,
    color: str,
) -> Mesh:
    if len(left_points) != len(right_points) or len(left_points) < 2:
        fallback = left_points[0] if left_points else (0.0, 0.0)
        return Cylinder(
            pos=(fallback[0], fallback[1], z_center_mm),
            r=0.1,
            height=thickness_mm,
            axis=(0, 0, 1),
            res=8,
        ).c(color).alpha(0.96)

    half_thickness = thickness_mm / 2.0
    count = len(left_points)

    bottom_left = [[x, y, z_center_mm - half_thickness] for x, y in left_points]
    bottom_right = [[x, y, z_center_mm - half_thickness] for x, y in right_points]
    top_left = [[x, y, z_center_mm + half_thickness] for x, y in left_points]
    top_right = [[x, y, z_center_mm + half_thickness] for x, y in right_points]

    vertices = bottom_left + bottom_right + top_left + top_right

    def bl(index: int) -> int:
        return index

    def br(index: int) -> int:
        return count + index

    def tl(index: int) -> int:
        return (2 * count) + index

    def tr(index: int) -> int:
        return (3 * count) + index

    faces: list[list[int]] = []

    for index in range(count - 1):
        next_index = index + 1
        faces.append([tl(index), tl(next_index), tr(next_index), tr(index)])
        faces.append([br(index), br(next_index), bl(next_index), bl(index)])
        faces.append([bl(index), bl(next_index), tl(next_index), tl(index)])
        faces.append([tr(index), tr(next_index), br(next_index), br(index)])

    faces.append([bl(0), br(0), tr(0), tl(0)])
    faces.append([tl(count - 1), tr(count - 1), br(count - 1), bl(count - 1)])

    return Mesh([vertices, faces]).c(color).alpha(0.96)


def copper_layer_sort_key(layer_name: str) -> tuple[int, int]:
    if layer_name == "F.Cu":
        return (0, 0)
    if layer_name.startswith("In") and layer_name.endswith(".Cu"):
        return (1, int(layer_name[2 : layer_name.index(".")]))
    if layer_name == "B.Cu":
        return (2, 0)
    return (3, 0)


def load_board_view_model(json_path: Path) -> BoardViewModel:
    resolved_path = json_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Copper path JSON not found. Run export_kicad_copper_paths.py with KiCad's Python first.\n"
            f"Expected file: {resolved_path}"
        )

    payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
    active_layers = sorted(payload["active_layers"], key=copper_layer_sort_key)
    stackup = load_stackup_definition(payload, float(payload.get("board_thickness_mm", 1.6)), active_layers)
    return BoardViewModel(
        board_path=Path(payload["source_pcb"]),
        board_thickness_mm=float(payload.get("board_thickness_mm", 1.6)),
        left_mm=float(payload["left_mm"]),
        top_mm=float(payload["top_mm"]),
        width_mm=float(payload["width_mm"]),
        height_mm=float(payload["height_mm"]),
        tracks=[
            TrackData(
                layer=item["layer"],
                net=item["net"],
                start_mm=PointMM(**item["start_mm"]),
                end_mm=PointMM(**item["end_mm"]),
                width_mm=float(item["width_mm"]),
            )
            for item in payload["tracks"]
        ],
        vias=[
            ViaData(
                net=item["net"],
                position_mm=PointMM(**item["position_mm"]),
                drill_mm=float(item["drill_mm"]),
                diameter_by_layer_mm={layer: float(value) for layer, value in item["diameter_by_layer_mm"].items()},
            )
            for item in payload["vias"]
        ],
        pads=[
            PadData(
                reference=item["reference"],
                pad_number=item["pad_number"],
                layer=item["layer"],
                net=item["net"],
                center_mm=PointMM(**item["center_mm"]),
                size_x_mm=float(item["size_x_mm"]),
                size_y_mm=float(item["size_y_mm"]),
                rotation_deg=float(item["rotation_deg"]),
                shape=item["shape"],
                roundrect_radius_mm=float(item["roundrect_radius_mm"]) if item.get("roundrect_radius_mm") is not None else None,
                drill_x_mm=float(item.get("drill_x_mm", 0.0)),
                drill_y_mm=float(item.get("drill_y_mm", 0.0)),
            )
            for item in payload["pads"]
        ],
        zones=[
            ZoneData(
                layer=item["layer"],
                net=item["net"],
                contours=[
                    ZoneContourData(
                        exterior_mm=[PointMM(**point) for point in contour["exterior_mm"]],
                        holes_mm=[[PointMM(**point) for point in hole] for hole in contour.get("holes_mm", [])],
                    )
                    for contour in item.get("contours", [])
                ],
            )
            for item in payload.get("zones", [])
        ],
        outline=[
            OutlineData(
                kind=item["kind"],
                start_mm=PointMM(**item["start_mm"]),
                end_mm=PointMM(**item["end_mm"]),
                mid_mm=PointMM(**item["mid_mm"]) if item.get("mid_mm") else None,
            )
            for item in payload["outline"]
        ],
        active_layers=active_layers,
        nets=payload["nets"],
        stackup=stackup,
    )


def find_kicad_python() -> Path | None:
    env_candidates = [
        os.environ.get("KICAD_PYTHON"),
        os.environ.get("KICAD_PYTHON_EXECUTABLE"),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return candidate_path
    for candidate_path in COMMON_KICAD_PYTHON_PATHS:
        if candidate_path.exists():
            return candidate_path
    return None


def export_copper_json(source_pcb: Path, output_json: Path) -> None:
    if not EXPORTER_SCRIPT.exists():
        raise ExportRefreshError(f"Exporter script not found: {EXPORTER_SCRIPT}")
    if not source_pcb.exists():
        raise ExportRefreshError(f"Source KiCad board not found: {source_pcb}")

    kicad_python = find_kicad_python()
    if kicad_python is None:
        raise ExportRefreshError(
            "Could not find KiCad's bundled Python interpreter.\n"
            "Install KiCad or set KICAD_PYTHON to something like:\n"
            r"  C:\Program Files\KiCad\10.0\bin\python.exe"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(kicad_python),
        str(EXPORTER_SCRIPT),
        "--input",
        str(source_pcb),
        "--output",
        str(output_json),
    ]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "No error output captured."
        raise ExportRefreshError(
            "KiCad export failed while refreshing copper_paths.json.\n"
            f"Command: {' '.join(command)}\n"
            f"Details:\n{details}"
        )


def refresh_copper_json() -> None:
    export_copper_json(SOURCE_PCB, DEFAULT_INPUT)


def main() -> None:
    try:
        refresh_copper_json()
    except ExportRefreshError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    model = load_board_view_model(DEFAULT_INPUT)
    viewer = VedoStackupViewer(model)
    StackupControlPanel(viewer).run()


if __name__ == "__main__":
    main()
