from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("kicad_copper_paths_exporter")
DEFAULT_INPUT = Path("dataset/Kicad/Arduino hat/Arduino_hat.kicad_pcb")
DEFAULT_OUTPUT = Path("output/arduino_hat/copper_paths.json")


class ExportError(Exception):
    """Raised when the copper path export fails."""


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
class StackupData:
    copper_layers: list[StackupCopperLayer]
    dielectric_layers: list[StackupDielectricLayer]


@dataclass(slots=True)
class CopperPathsExport:
    source_pcb: str
    board_thickness_mm: float
    left_mm: float
    top_mm: float
    width_mm: float
    height_mm: float
    active_layers: list[str]
    tracks: list[TrackData]
    vias: list[ViaData]
    pads: list[PadData]
    outline: list[OutlineData]
    nets: list[str]
    stackup: StackupData


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Arduino hat copper pathways to JSON for the tkinter viewer."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the .kicad_pcb file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the output JSON file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def configure_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, level_name), format="%(levelname)s: %(message)s")


def load_pcbnew() -> Any:
    try:
        import pcbnew  # type: ignore
    except ImportError as exc:
        raise ExportError(
            "Could not import KiCad's pcbnew module. Run this exporter with KiCad's bundled Python interpreter."
        ) from exc
    return pcbnew


def to_mm(pcbnew: Any, value: int | float) -> float:
    return float(pcbnew.ToMM(value))


def pad_shape_name(pcbnew: Any, pad: Any) -> str:
    shape = pad.GetShape()
    for attribute in dir(pcbnew):
        if attribute.startswith("PAD_SHAPE_") and getattr(pcbnew, attribute) == shape:
            return attribute.removeprefix("PAD_SHAPE_").lower()
    return str(shape)


def pad_roundrect_radius_mm(pcbnew: Any, pad: Any, shape_name: str) -> float | None:
    if shape_name != "roundrect":
        return None
    if hasattr(pad, "GetRoundRectCornerRadius"):
        return to_mm(pcbnew, pad.GetRoundRectCornerRadius())
    return None


def pad_drill_size_mm(pcbnew: Any, pad: Any) -> tuple[float, float]:
    if not hasattr(pad, "GetDrillSize"):
        return (0.0, 0.0)
    drill = pad.GetDrillSize()
    return (to_mm(pcbnew, drill.x), to_mm(pcbnew, drill.y))


def rotation_degrees(item: Any) -> float:
    if hasattr(item, "GetOrientationDegrees"):
        return float(item.GetOrientationDegrees())
    orientation = item.GetOrientation()
    if hasattr(orientation, "AsDegrees"):
        return float(orientation.AsDegrees())
    return float(orientation) / 10.0


def copper_layer_sort_key(layer_name: str) -> tuple[int, int]:
    if layer_name == "F.Cu":
        return (0, 0)
    if layer_name.startswith("In") and layer_name.endswith(".Cu"):
        return (1, int(layer_name[2 : layer_name.index(".")]))
    if layer_name == "B.Cu":
        return (2, 0)
    return (3, 0)


def build_stackup(active_layers: list[str], board_thickness_mm: float) -> StackupData:
    if active_layers == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"] and abs(board_thickness_mm - 1.6) <= 0.01:
        return StackupData(
            copper_layers=[
                StackupCopperLayer(name="F.Cu", thickness_mm=0.035),
                StackupCopperLayer(name="In1.Cu", thickness_mm=0.018),
                StackupCopperLayer(name="In2.Cu", thickness_mm=0.018),
                StackupCopperLayer(name="B.Cu", thickness_mm=0.035),
            ],
            dielectric_layers=[
                StackupDielectricLayer(name="Top FR-4/prepreg", upper_layer="F.Cu", lower_layer="In1.Cu", thickness_mm=0.215),
                StackupDielectricLayer(name="FR-4 core", upper_layer="In1.Cu", lower_layer="In2.Cu", thickness_mm=1.064),
                StackupDielectricLayer(name="Bottom FR-4/prepreg", upper_layer="In2.Cu", lower_layer="B.Cu", thickness_mm=0.215),
            ],
        )

    copper_thickness_mm = 0.035
    copper_layers = [StackupCopperLayer(name=layer_name, thickness_mm=copper_thickness_mm) for layer_name in active_layers]
    dielectric_total_mm = max(board_thickness_mm - (len(copper_layers) * copper_thickness_mm), 0.0)
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
    return StackupData(copper_layers=copper_layers, dielectric_layers=dielectric_layers)


def export_copper_paths(input_path: Path, output_path: Path) -> None:
    pcbnew = load_pcbnew()
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.exists():
        raise ExportError(f"Board file not found: {resolved_input}")

    board = pcbnew.LoadBoard(str(resolved_input))
    if board is None:
        raise ExportError(f"KiCad could not load the board: {resolved_input}")

    bbox = board.GetBoardEdgesBoundingBox()
    board_thickness_mm = to_mm(pcbnew, board.GetDesignSettings().GetBoardThickness())
    left_mm = to_mm(pcbnew, bbox.GetLeft())
    top_mm = to_mm(pcbnew, bbox.GetTop())
    width_mm = to_mm(pcbnew, bbox.GetWidth())
    height_mm = to_mm(pcbnew, bbox.GetHeight())

    active_layers: list[str] = []
    for layer_id in range(64):
        layer_name = str(pcbnew.LayerName(layer_id))
        if layer_name.endswith(".Cu") and board.IsLayerEnabled(layer_id):
            active_layers.append(layer_name)

    tracks: list[TrackData] = []
    vias: list[ViaData] = []
    pads: list[PadData] = []
    outline: list[OutlineData] = []
    nets: set[str] = set()

    for item in board.GetTracks():
        if type(item).__name__ == "PCB_VIA":
            diameter_by_layer_mm: dict[str, float] = {}
            for layer_id in item.GetLayerSet().Seq():
                layer_name = str(pcbnew.LayerName(layer_id))
                if layer_name not in active_layers:
                    continue
                diameter_by_layer_mm[layer_name] = to_mm(pcbnew, item.GetWidth(layer_id))

            via = ViaData(
                net=str(item.GetNetname() or ""),
                position_mm=PointMM(
                    x_mm=to_mm(pcbnew, item.GetPosition().x),
                    y_mm=to_mm(pcbnew, item.GetPosition().y),
                ),
                drill_mm=to_mm(pcbnew, item.GetDrill()),
                diameter_by_layer_mm=diameter_by_layer_mm,
            )
            vias.append(via)
            if via.net:
                nets.add(via.net)
            continue

        track = TrackData(
            layer=str(pcbnew.LayerName(item.GetLayer())),
            net=str(item.GetNetname() or ""),
            start_mm=PointMM(
                x_mm=to_mm(pcbnew, item.GetStart().x),
                y_mm=to_mm(pcbnew, item.GetStart().y),
            ),
            end_mm=PointMM(
                x_mm=to_mm(pcbnew, item.GetEnd().x),
                y_mm=to_mm(pcbnew, item.GetEnd().y),
            ),
            width_mm=to_mm(pcbnew, item.GetWidth()),
        )
        tracks.append(track)
        if track.net:
            nets.add(track.net)

    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            for layer_id in pad.GetLayerSet().Seq():
                layer_name = str(pcbnew.LayerName(layer_id))
                if layer_name not in active_layers:
                    continue
                shape_name = pad_shape_name(pcbnew, pad)
                pad_data = PadData(
                    reference=str(footprint.GetReference()),
                    pad_number=str(pad.GetNumber()),
                    layer=layer_name,
                    net=str(pad.GetNetname() or ""),
                    center_mm=PointMM(
                        x_mm=to_mm(pcbnew, pad.GetPosition().x),
                        y_mm=to_mm(pcbnew, pad.GetPosition().y),
                    ),
                    size_x_mm=to_mm(pcbnew, pad.GetSize().x),
                    size_y_mm=to_mm(pcbnew, pad.GetSize().y),
                    rotation_deg=rotation_degrees(pad),
                    shape=shape_name,
                    roundrect_radius_mm=pad_roundrect_radius_mm(pcbnew, pad, shape_name),
                    drill_x_mm=pad_drill_size_mm(pcbnew, pad)[0],
                    drill_y_mm=pad_drill_size_mm(pcbnew, pad)[1],
                )
                pads.append(pad_data)
                if pad_data.net:
                    nets.add(pad_data.net)

    edge_cuts = pcbnew.Edge_Cuts
    for drawing in board.GetDrawings():
        if drawing.GetLayer() != edge_cuts:
            continue
        kind = str(drawing.GetShapeStr()).lower() if hasattr(drawing, "GetShapeStr") else ""
        if "arc" in kind:
            outline.append(
                OutlineData(
                    kind="arc",
                    start_mm=PointMM(
                        x_mm=to_mm(pcbnew, drawing.GetStart().x),
                        y_mm=to_mm(pcbnew, drawing.GetStart().y),
                    ),
                    end_mm=PointMM(
                        x_mm=to_mm(pcbnew, drawing.GetEnd().x),
                        y_mm=to_mm(pcbnew, drawing.GetEnd().y),
                    ),
                    mid_mm=PointMM(
                        x_mm=to_mm(pcbnew, drawing.GetArcMid().x),
                        y_mm=to_mm(pcbnew, drawing.GetArcMid().y),
                    ),
                )
            )
        elif "segment" in kind or "line" in kind:
            outline.append(
                OutlineData(
                    kind="segment",
                    start_mm=PointMM(
                        x_mm=to_mm(pcbnew, drawing.GetStart().x),
                        y_mm=to_mm(pcbnew, drawing.GetStart().y),
                    ),
                    end_mm=PointMM(
                        x_mm=to_mm(pcbnew, drawing.GetEnd().x),
                        y_mm=to_mm(pcbnew, drawing.GetEnd().y),
                    ),
                )
            )

    export_payload = CopperPathsExport(
        source_pcb=str(resolved_input),
        board_thickness_mm=board_thickness_mm,
        left_mm=left_mm,
        top_mm=top_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        active_layers=sorted(active_layers, key=copper_layer_sort_key),
        tracks=tracks,
        vias=vias,
        pads=pads,
        outline=outline,
        nets=sorted(nets),
        stackup=build_stackup(sorted(active_layers, key=copper_layer_sort_key), board_thickness_mm),
    )

    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(export_payload), output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")
    LOGGER.info("Wrote %s", resolved_output)


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        export_copper_paths(args.input, args.output)
    except ExportError as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Unexpected failure during copper path export: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
