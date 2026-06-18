from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from modules.arduino_hat_stackup_vedo import (
    COMMON_KICAD_PYTHON_PATHS,
    MouseBiteSettings,
    OpenCircuitSettings,
    OverEtchSettings,
    ShortCircuitSettings,
    StackupControlPanel,
    UnderEtchSettings,
    VedoStackupViewer,
    export_material_partition_with_defects,
    load_board_view_model,
)
from modules.export_material_partition import export_material_partition


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PCB = REPO_ROOT / "DataSet" / "KICAD" / "Arduino hat" / "Arduino_hat.kicad_pcb"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_CLEARANCE_MM = 0.01


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def find_kicad_python(explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return candidate
    for env_name in ("KICAD_PYTHON", "KICAD_PYTHON_EXECUTABLE"):
        candidate_value = os.environ.get(env_name)
        if not candidate_value:
            continue
        candidate = Path(candidate_value).expanduser()
        if candidate.exists():
            return candidate
    for candidate in COMMON_KICAD_PYTHON_PATHS:
        if candidate.exists():
            return candidate
    return None


def choose_input_pcb(initial_path: Path | None = None) -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_dir = str((initial_path or DEFAULT_INPUT_PCB).expanduser().resolve().parent)
    selected = filedialog.askopenfilename(
        title="Choose a KiCad PCB file",
        initialdir=initial_dir,
        filetypes=[("KiCad PCB files", "*.kicad_pcb"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected:
        raise PipelineError("No .kicad_pcb file was selected.")
    selected_path = Path(selected).expanduser().resolve()
    if selected_path.suffix.lower() != ".kicad_pcb":
        raise PipelineError(f"Selected file is not a .kicad_pcb file: {selected_path}")
    return selected_path


def slugify_board_name(board_path: Path) -> str:
    raw_name = board_path.stem.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_")
    return slug or "board"


def build_output_paths(board_path: Path, copper_json: Path | None, stl_output_dir: Path | None, defect_output_dir: Path | None) -> tuple[Path, Path, Path]:
    board_key = slugify_board_name(board_path)
    board_output_root = DEFAULT_OUTPUT_ROOT / board_key
    resolved_copper_json = copper_json.expanduser().resolve() if copper_json is not None else (board_output_root / "copper_paths.json")
    resolved_stl_output_dir = stl_output_dir.expanduser().resolve() if stl_output_dir is not None else (board_output_root / "material_partition")
    resolved_defect_output_dir = defect_output_dir.expanduser().resolve() if defect_output_dir is not None else (board_output_root / "material_partition_defects")
    return resolved_copper_json, resolved_stl_output_dir, resolved_defect_output_dir


def run_extract(input_pcb: Path, copper_json: Path, kicad_python: str | None, log_level: str) -> None:
    executable = find_kicad_python(kicad_python)
    if executable is None:
        raise PipelineError(
            "Could not find KiCad's Python interpreter. Set KICAD_PYTHON or pass --kicad-python."
        )

    command = [
        str(executable),
        str(REPO_ROOT / "modules" / "export_kicad_copper_paths.py"),
        "--input",
        str(input_pcb),
        "--output",
        str(copper_json),
        "--log-level",
        log_level,
    ]
    result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "No error output captured."
        raise PipelineError(f"Extractor failed.\nCommand: {' '.join(command)}\n{details}")


def run_base_stl_export(copper_json: Path, output_dir: Path, clearance_mm: float) -> None:
    export_material_partition(copper_json, output_dir, clearance_mm)


def run_viewer(copper_json: Path, defect_output_dir: Path) -> None:
    model = load_board_view_model(copper_json)
    viewer = VedoStackupViewer(model, export_output_dir=defect_output_dir)
    StackupControlPanel(viewer).run()


def run_defect_export(copper_json: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    model = load_board_view_model(copper_json)
    return export_material_partition_with_defects(
        model,
        output_dir=output_dir,
        overetch=OverEtchSettings(
            enabled=args.overetch_count > 0,
            count=args.overetch_count,
            severity=args.overetch_severity,
            recovery_mm=args.overetch_recovery_mm,
            falloff_mode=args.overetch_falloff,
            noise_amount=args.overetch_noise,
            seed=args.overetch_seed,
        ),
        mousebite=MouseBiteSettings(
            enabled=args.mousebite_count > 0,
            count=args.mousebite_count,
            recovery_mm=args.mousebite_recovery_mm,
            noise_amount=args.mousebite_noise,
            blob_count=args.mousebite_blob_count,
            blob_size_mm=args.mousebite_blob_size_mm,
            seed=args.mousebite_seed,
        ),
        underetch=UnderEtchSettings(
            enabled=args.underetch_count > 0,
            count=args.underetch_count,
            severity=args.underetch_severity,
            recovery_mm=args.underetch_recovery_mm,
            falloff_mode=args.underetch_falloff,
            noise_amount=args.underetch_noise,
            blob_count=args.underetch_blob_count,
            blob_size_mm=args.underetch_blob_size_mm,
            seed=args.underetch_seed,
        ),
        opencircuit=OpenCircuitSettings(
            enabled=args.opencircuit_count > 0,
            count=args.opencircuit_count,
            gap_mm=args.opencircuit_gap_mm,
            recovery_mm=args.opencircuit_recovery_mm,
            noise_amount=args.opencircuit_noise,
            seed=args.opencircuit_seed,
        ),
        shortcircuit=ShortCircuitSettings(
            enabled=args.shortcircuit_count > 0,
            count=args.shortcircuit_count,
            max_gap_mm=args.shortcircuit_gap_mm,
            bridge_width_mm=args.shortcircuit_bridge_width_mm,
            noise_amount=args.shortcircuit_noise,
            seed=args.shortcircuit_seed,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Root launcher for PCB extraction, STL creation, viewing, defect editing, and export."
    )
    parser.set_defaults(command="full")
    parser.add_argument("--input-pcb", type=Path, default=None, help="Optional path to the source .kicad_pcb board.")
    parser.add_argument("--copper-json", type=Path, default=None, help="Optional override for the extracted copper JSON path.")
    parser.add_argument("--stl-output-dir", type=Path, default=None, help="Optional override for the clean STL output directory.")
    parser.add_argument("--defect-output-dir", type=Path, default=None, help="Optional override for defect STL exports.")
    parser.add_argument("--clearance-mm", type=float, default=DEFAULT_CLEARANCE_MM, help="Material clearance for STL partition export.")
    parser.add_argument("--kicad-python", default=None, help="Path to KiCad's bundled Python executable.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Extractor log verbosity.")
    parser.add_argument("--no-picker", action="store_true", help="Do not show the file picker; use --input-pcb or the default sample board.")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("extract", help="Run the KiCad copper-path extractor only.")
    subparsers.add_parser("stl", help="Create the clean STL material partition from copper JSON.")
    subparsers.add_parser("viewer", help="Open the interactive viewer with defect controls and STL export.")

    defect_parser = subparsers.add_parser("defect-export", help="Export a defected STL partition directly from CLI settings.")
    add_defect_arguments(defect_parser)

    full_parser = subparsers.add_parser("full", help="Pick a board, extract it, create clean STLs, then open the interactive viewer.")
    full_parser.add_argument("--skip-viewer", action="store_true", help="Finish after clean STL creation.")

    return parser


def add_defect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overetch-count", type=int, default=0)
    parser.add_argument("--overetch-severity", type=float, default=0.55)
    parser.add_argument("--overetch-recovery-mm", type=float, default=1.2)
    parser.add_argument("--overetch-noise", type=float, default=0.12)
    parser.add_argument("--overetch-seed", type=int, default=1)
    parser.add_argument("--overetch-falloff", choices=["gaussian", "exponential"], default="gaussian")

    parser.add_argument("--mousebite-count", type=int, default=0)
    parser.add_argument("--mousebite-recovery-mm", type=float, default=1.0)
    parser.add_argument("--mousebite-noise", type=float, default=0.15)
    parser.add_argument("--mousebite-blob-count", type=int, default=4)
    parser.add_argument("--mousebite-blob-size-mm", type=float, default=0.18)
    parser.add_argument("--mousebite-seed", type=int, default=101)

    parser.add_argument("--underetch-count", type=int, default=0)
    parser.add_argument("--underetch-severity", type=float, default=0.45)
    parser.add_argument("--underetch-recovery-mm", type=float, default=1.4)
    parser.add_argument("--underetch-noise", type=float, default=0.15)
    parser.add_argument("--underetch-blob-count", type=int, default=4)
    parser.add_argument("--underetch-blob-size-mm", type=float, default=0.16)
    parser.add_argument("--underetch-seed", type=int, default=201)
    parser.add_argument("--underetch-falloff", choices=["gaussian", "exponential"], default="gaussian")

    parser.add_argument("--opencircuit-count", type=int, default=0)
    parser.add_argument("--opencircuit-gap-mm", type=float, default=0.35)
    parser.add_argument("--opencircuit-recovery-mm", type=float, default=0.7)
    parser.add_argument("--opencircuit-noise", type=float, default=0.05)
    parser.add_argument("--opencircuit-seed", type=int, default=301)

    parser.add_argument("--shortcircuit-count", type=int, default=0)
    parser.add_argument("--shortcircuit-gap-mm", type=float, default=0.5)
    parser.add_argument("--shortcircuit-bridge-width-mm", type=float, default=0.18)
    parser.add_argument("--shortcircuit-noise", type=float, default=0.05)
    parser.add_argument("--shortcircuit-seed", type=int, default=401)


def resolve_input_board(args: argparse.Namespace) -> Path:
    if args.input_pcb is not None:
        return args.input_pcb.expanduser().resolve()
    if args.no_picker:
        return DEFAULT_INPUT_PCB.expanduser().resolve()
    return choose_input_pcb(DEFAULT_INPUT_PCB)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_pcb = resolve_input_board(args)
        copper_json, stl_output_dir, defect_output_dir = build_output_paths(
            input_pcb,
            args.copper_json,
            args.stl_output_dir,
            args.defect_output_dir,
        )

        if args.command == "extract":
            run_extract(input_pcb, copper_json, args.kicad_python, args.log_level)
            return 0
        if args.command == "stl":
            run_base_stl_export(copper_json, stl_output_dir, args.clearance_mm)
            return 0
        if args.command == "viewer":
            run_viewer(copper_json, defect_output_dir)
            return 0
        if args.command == "defect-export":
            run_defect_export(copper_json, defect_output_dir, args)
            return 0

        run_extract(input_pcb, copper_json, args.kicad_python, args.log_level)
        run_base_stl_export(copper_json, stl_output_dir, args.clearance_mm)
        if not getattr(args, "skip_viewer", False):
            run_viewer(copper_json, defect_output_dir)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
