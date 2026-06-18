from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely.affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
import trimesh


DEFAULT_INPUT = Path("output/arduino_hat/copper_paths.json")
DEFAULT_OUTPUT_DIR = Path("output/arduino_hat/material_partition")
COPPER_THICKNESS_MM = 0.035
DEFAULT_CLEARANCE_MM = 0.01
BARREL_LAYER_OVERLAP_MM = 0.005
VIA_SIDE_COUNT = 10
VIA_PLATING_THICKNESS_MM = COPPER_THICKNESS_MM
PAD_CIRCLE_SIDE_COUNT = 10
PAD_CURVE_POINT_COUNT = 5


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
class MeshRecord:
    name: str
    path: str
    triangle_count: int


@dataclass(slots=True)
class PartitionManifest:
    source: str
    clearance_mm: float
    copper_thickness_mm: float
    meshes: list[MeshRecord]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Monte-Carlo-safe FR4 and copper partition meshes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to copper_paths.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported meshes.")
    parser.add_argument("--clearance-mm", type=float, default=DEFAULT_CLEARANCE_MM, help="Air gap between materials.")
    return parser


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


def load_stackup_definition(payload: dict[str, Any], board_thickness_mm: float, active_layers: list[str]) -> StackupDefinition:
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


def board_to_centered(model: BoardViewModel, point: PointMM) -> tuple[float, float]:
    x_centered = point.x_mm - model.left_mm - (model.width_mm / 2.0)
    y_centered = -1.0 * (point.y_mm - model.top_mm - (model.height_mm / 2.0))
    return x_centered, y_centered


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
    polygon = Polygon(points)
    return polygon.buffer(0)


def layer_thickness_map(stackup: StackupDefinition) -> dict[str, float]:
    return {layer.name: layer.thickness_mm for layer in stackup.copper_layers}


def dielectric_layer_map(stackup: StackupDefinition) -> dict[tuple[str, str], StackupDielectricLayer]:
    return {(layer.upper_layer, layer.lower_layer): layer for layer in stackup.dielectric_layers}


def build_z_map(board_thickness_mm: float, stackup: StackupDefinition, copper_layers: list[str]) -> dict[str, float]:
    thickness_by_layer = layer_thickness_map(stackup)
    dielectric_by_pair = dielectric_layer_map(stackup)
    current_top = board_thickness_mm / 2.0
    z_map: dict[str, float] = {}
    for index, layer_name in enumerate(copper_layers):
        thickness_mm = thickness_by_layer.get(layer_name, COPPER_THICKNESS_MM)
        current_bottom = current_top - thickness_mm
        z_map[layer_name] = (current_top + current_bottom) / 2.0
        current_top = current_bottom
        if index < len(copper_layers) - 1:
            next_layer_name = copper_layers[index + 1]
            dielectric = dielectric_by_pair.get((layer_name, next_layer_name))
            current_top -= dielectric.thickness_mm if dielectric is not None else 0.0
    return z_map


def copper_thickness_for_layer(model: BoardViewModel, layer_name: str) -> float:
    return layer_thickness_map(model.stackup).get(layer_name, COPPER_THICKNESS_MM)


def copper_bounds_for_layer(model: BoardViewModel, layer_name: str, z_map: dict[str, float]) -> tuple[float, float]:
    thickness_mm = copper_thickness_for_layer(model, layer_name)
    center_z = z_map[layer_name]
    return (center_z - (thickness_mm / 2.0), center_z + (thickness_mm / 2.0))


def dielectric_between_layers(model: BoardViewModel, upper_layer: str, lower_layer: str) -> StackupDielectricLayer | None:
    return dielectric_layer_map(model.stackup).get((upper_layer, lower_layer))


def regular_ngon(center_x: float, center_y: float, radius: float, sides: int) -> Polygon:
    points = []
    for index in range(sides):
        angle = (2.0 * math.pi * index) / sides
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return Polygon(points)


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


def make_via_barrel_ring_polygon(center_x: float, center_y: float, via: ViaData) -> Polygon:
    outer_polygon = regular_ngon(center_x, center_y, via_barrel_outer_radius(via), VIA_SIDE_COUNT)
    inner_polygon = make_via_drill_polygon(center_x, center_y, via)
    return outer_polygon.difference(inner_polygon).buffer(0)


def pad_has_plated_hole(pad: PadData) -> bool:
    return pad.drill_x_mm > 1e-6 and pad.drill_y_mm > 1e-6


def pad_physical_key(pad: PadData) -> tuple[str, str, int, int]:
    return (
        pad.reference,
        pad.pad_number,
        round(pad.center_mm.x_mm * 1000.0),
        round(pad.center_mm.y_mm * 1000.0),
    )


def point_key(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0] * 1000), round(point[1] * 1000))


def deduplicate_consecutive_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 1e-6:
            continue
        cleaned.append(point)
    return cleaned


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
                current_point = edge_start
                next_point = edge_end
            else:
                current_point = edge_end
                next_point = edge_start

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


def create_pad_barrel_ring_polygon(model: BoardViewModel, pad: PadData) -> Polygon | None:
    drill_polygon = create_pad_drill_polygon(model, pad)
    if drill_polygon is None:
        return None
    outer_polygon = drill_polygon.buffer(VIA_PLATING_THICKNESS_MM, cap_style="round", join_style="round", quad_segs=4)
    pad_polygon = create_pad_polygon(model, pad)
    ring_polygon = outer_polygon.intersection(pad_polygon).difference(drill_polygon).buffer(0)
    if ring_polygon.is_empty:
        return None
    return ring_polygon


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


def create_layer_copper_polygon(model: BoardViewModel, layer_name: str, board_polygon: Polygon) -> Polygon | MultiPolygon | GeometryCollection:
    geometries: list[Any] = []
    zone_geometry = create_zone_geometry(model, layer_name)
    if not zone_geometry.is_empty:
        geometries.append(zone_geometry)
    layer_tracks = [track for track in model.tracks if track.layer == layer_name]
    for polyline_xy, width_mm in build_connected_track_paths(model, layer_tracks):
        geometries.append(
            LineString(polyline_xy).buffer(
                width_mm / 2.0,
                cap_style="flat",
                join_style="mitre",
                mitre_limit=2.0,
                quad_segs=4,
            )
        )
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
    copper = unary_union(geometries).intersection(board_polygon)
    holes = []
    seen_pad_holes: set[tuple[str, str]] = set()
    for pad in model.pads:
        if pad.layer != layer_name:
            continue
        pad_key = (pad.reference, pad.pad_number)
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


def create_layer_trace_polygon(model: BoardViewModel, layer_name: str, board_polygon: Polygon) -> Polygon | MultiPolygon | GeometryCollection:
    geometries: list[Any] = []
    layer_tracks = [track for track in model.tracks if track.layer == layer_name]
    for polyline_xy, width_mm in build_connected_track_paths(model, layer_tracks):
        geometries.append(
            LineString(polyline_xy).buffer(
                width_mm / 2.0,
                cap_style="flat",
                join_style="mitre",
                mitre_limit=2.0,
                quad_segs=4,
            )
        )
    if not geometries:
        return GeometryCollection()
    traces = unary_union(geometries).intersection(board_polygon)
    holes = []
    for via in model.vias:
        if layer_name not in via.diameter_by_layer_mm:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
        holes.append(make_via_drill_polygon(center_x, center_y, via))
    if holes:
        traces = traces.difference(unary_union(holes))
    return traces.buffer(0)


def iter_polygons(geometry: Any) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for item in geometry.geoms:
            polygons.extend(iter_polygons(item))
        return polygons
    return []


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
            extents = part.bounds[1] - part.bounds[0]
            if not np.all(np.isfinite(extents)) or np.any(extents <= 1e-8):
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


def via_z_span(model: BoardViewModel, via: ViaData, active_layers: list[str], z_map: dict[str, float]) -> tuple[float, float] | None:
    layer_names = [layer for layer in active_layers if layer in via.diameter_by_layer_mm]
    if not layer_names:
        return None
    bounds = [copper_bounds_for_layer(model, layer, z_map) for layer in layer_names]
    z_min = min(bound[0] for bound in bounds)
    z_max = max(bound[1] for bound in bounds)
    if z_max <= z_min:
        return None
    return (z_min, z_max)


def create_via_barrel_mesh(
    model: BoardViewModel,
    z_map: dict[str, float],
    cap_ends: bool = True,
    overlap_mm: float = BARREL_LAYER_OVERLAP_MM,
) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    for via in model.vias:
        if not via.diameter_by_layer_mm:
            continue
        layer_indices = [index for index, layer_name in enumerate(model.active_layers) if layer_name in via.diameter_by_layer_mm]
        if len(layer_indices) < 2:
            continue
        center_x, center_y = board_to_centered(model, via.position_mm)
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
        if not pad_has_plated_hole(pad):
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
        if not via.diameter_by_layer_mm:
            continue
        span = via_z_span(model, via, model.active_layers, z_map)
        if span is None:
            continue
        z_min, z_max = span
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
        if not pad_has_plated_hole(pad):
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


def create_dielectric_slab_polygon(
    board_polygon: Polygon,
    model: BoardViewModel,
    upper_layer_index: int,
    lower_layer_index: int,
    clearance_mm: float,
) -> Polygon | MultiPolygon | GeometryCollection:
    subtractors: list[Any] = []
    upper_layer = model.active_layers[upper_layer_index]
    lower_layer = model.active_layers[lower_layer_index]
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


def export_material_partition(input_path: Path, output_dir: Path, clearance_mm: float) -> None:
    model = load_board_view_model(input_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    board_polygon = build_board_polygon(model)
    z_map = build_z_map(model.board_thickness_mm, model.stackup, model.active_layers)
    mesh_records: list[MeshRecord] = []
    scene = trimesh.Scene()
    copper_meshes: list[trimesh.Trimesh] = []
    trace_meshes: list[trimesh.Trimesh] = []
    layer_mesh_by_name: dict[str, trimesh.Trimesh] = {}
    partition_meshes: list[trimesh.Trimesh] = []

    for layer_name in model.active_layers:
        copper_thickness_mm = copper_thickness_for_layer(model, layer_name)
        trace_geometry = create_layer_trace_polygon(model, layer_name, board_polygon)
        trace_mesh = extrude_geometry(
            trace_geometry,
            height_mm=copper_thickness_mm,
            z_bottom_mm=z_map[layer_name] - (copper_thickness_mm / 2.0),
        )
        if trace_mesh is not None:
            trace_path = output_dir / f"trace_{layer_name.replace('.', '_')}.stl"
            trace_mesh.export(trace_path)
            trace_meshes.append(trace_mesh)
            mesh_records.append(MeshRecord(name=f"trace_{layer_name}", path=str(trace_path), triangle_count=len(trace_mesh.faces)))

        copper_geometry = create_layer_copper_polygon(model, layer_name, board_polygon)
        mesh = extrude_geometry(
            copper_geometry,
            height_mm=copper_thickness_mm,
            z_bottom_mm=z_map[layer_name] - (copper_thickness_mm / 2.0),
        )
        if mesh is None:
            continue
        path = output_dir / f"copper_{layer_name.replace('.', '_')}.stl"
        mesh.export(path)
        copper_meshes.append(mesh)
        layer_mesh_by_name[layer_name] = mesh
        mesh_records.append(MeshRecord(name=f"copper_{layer_name}", path=str(path), triangle_count=len(mesh.faces)))
        scene.add_geometry(mesh, node_name=f"copper_{layer_name}")

    via_mesh = create_via_barrel_mesh(model, z_map)
    if via_mesh is not None:
        via_path = output_dir / "via_barrels.stl"
        via_mesh.export(via_path)
        copper_meshes.append(via_mesh)
        mesh_records.append(MeshRecord(name="via_barrels", path=str(via_path), triangle_count=len(via_mesh.faces)))
        scene.add_geometry(via_mesh, node_name="via_barrels")

    pad_barrel_mesh = create_pad_barrel_mesh(model, z_map)
    if pad_barrel_mesh is not None:
        pad_barrel_path = output_dir / "pad_barrels.stl"
        pad_barrel_mesh.export(pad_barrel_path)
        copper_meshes.append(pad_barrel_mesh)
        mesh_records.append(MeshRecord(name="pad_barrels", path=str(pad_barrel_path), triangle_count=len(pad_barrel_mesh.faces)))
        scene.add_geometry(pad_barrel_mesh, node_name="pad_barrels")

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
        if copper_all_mesh is None:
            raise RuntimeError("Failed to build copper_all.stl from layer and via meshes.")
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
            if cut_mesh is None:
                raise RuntimeError("Failed to cut via and plated-hole air from copper_all.stl.")
            copper_all_mesh = cut_mesh
        copper_all_path = output_dir / "copper_all.stl"
        copper_all_mesh.export(copper_all_path)
        mesh_records.append(MeshRecord(name="copper_all", path=str(copper_all_path), triangle_count=len(copper_all_mesh.faces)))
        partition_meshes.append(copper_all_mesh)

    if trace_meshes:
        trace_all_mesh = trimesh.util.concatenate(trace_meshes)
        trace_all_path = output_dir / "trace_all.stl"
        trace_all_mesh.export(trace_all_path)
        mesh_records.append(MeshRecord(name="trace_all", path=str(trace_all_path), triangle_count=len(trace_all_mesh.faces)))

    via_air_mesh = create_via_air_mesh(model, z_map)
    if via_air_mesh is not None:
        via_air_path = output_dir / "via_air.stl"
        via_air_mesh.export(via_air_path)
        mesh_records.append(MeshRecord(name="via_air", path=str(via_air_path), triangle_count=len(via_air_mesh.faces)))
        scene.add_geometry(via_air_mesh, node_name="via_air")

    pad_air_mesh = create_pad_air_mesh(model, z_map)
    if pad_air_mesh is not None:
        pad_air_path = output_dir / "pad_air.stl"
        pad_air_mesh.export(pad_air_path)
        mesh_records.append(MeshRecord(name="pad_air", path=str(pad_air_path), triangle_count=len(pad_air_mesh.faces)))
        scene.add_geometry(pad_air_mesh, node_name="pad_air")

    for index in range(len(model.active_layers) - 1):
        upper_layer = model.active_layers[index]
        lower_layer = model.active_layers[index + 1]
        upper_bounds = copper_bounds_for_layer(model, upper_layer, z_map)
        lower_bounds = copper_bounds_for_layer(model, lower_layer, z_map)
        z_top = upper_bounds[0] - clearance_mm
        z_bottom = lower_bounds[1] + clearance_mm
        height_mm = z_top - z_bottom
        if height_mm <= 0:
            continue
        slab_geometry = create_dielectric_slab_polygon(board_polygon, model, index, index + 1, clearance_mm)
        mesh = extrude_geometry(slab_geometry, height_mm=height_mm, z_bottom_mm=z_bottom)
        if mesh is None:
            continue
        slab_name = f"fr4_{upper_layer.replace('.', '_')}_to_{lower_layer.replace('.', '_')}"
        path = output_dir / f"{slab_name}.stl"
        mesh.export(path)
        mesh_records.append(MeshRecord(name=slab_name, path=str(path), triangle_count=len(mesh.faces)))
        scene.add_geometry(mesh, node_name=slab_name)
        partition_meshes.append(mesh)

    if partition_meshes:
        pcb_parts_all_mesh = trimesh.util.concatenate(partition_meshes)
        pcb_parts_all_path = output_dir / "pcb_parts_all.stl"
        pcb_parts_all_mesh.export(pcb_parts_all_path)
        mesh_records.append(
            MeshRecord(
                name="pcb_parts_all",
                path=str(pcb_parts_all_path),
                triangle_count=len(pcb_parts_all_mesh.faces),
            )
        )

    glb_path = output_dir / "material_partition.glb"
    scene.export(glb_path)
    mesh_records.append(MeshRecord(name="material_partition_glb", path=str(glb_path), triangle_count=0))

    manifest = PartitionManifest(
        source=str(model.board_path),
        clearance_mm=clearance_mm,
        copper_thickness_mm=max((layer.thickness_mm for layer in model.stackup.copper_layers), default=COPPER_THICKNESS_MM),
        meshes=mesh_records,
    )
    manifest_path = output_dir / "material_partition_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    export_material_partition(args.input, args.output_dir, args.clearance_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
