from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
DEFAULT_PCB_LIBRARY_DIR = DEFAULT_OUTPUT_ROOT / "pcb_library"
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
    initial_candidate = (initial_path or DEFAULT_INPUT_PCB).expanduser().resolve()
    initial_dir = str(initial_candidate.parent)
    chosen_path: Path | None = None

    root = tk.Tk()
    root.title("Choose PCB File")
    root.attributes("-topmost", True)
    root.geometry("760x170")
    root.minsize(680, 170)
    selected_value = tk.StringVar(master=root, value=str(initial_candidate))

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)

    ttk.Label(
        frame,
        text="Select the KiCad .kicad_pcb file to use for extraction, STL export, and viewing.",
        wraplength=700,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w")

    entry = ttk.Entry(frame, textvariable=selected_value)
    entry.grid(row=1, column=0, sticky="ew", pady=(14, 8))

    def browse() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose a KiCad PCB file",
            initialdir=initial_dir,
            filetypes=[("KiCad PCB files", "*.kicad_pcb"), ("All files", "*.*")],
        )
        if selected:
            selected_value.set(selected)

    def confirm() -> None:
        nonlocal chosen_path
        raw_value = selected_value.get().strip()
        if not raw_value:
            messagebox.showerror("Missing file", "Please choose a .kicad_pcb file.", parent=root)
            return
        candidate = Path(raw_value).expanduser().resolve()
        if candidate.suffix.lower() != ".kicad_pcb":
            messagebox.showerror("Invalid file", f"Selected file is not a .kicad_pcb file:\n{candidate}", parent=root)
            return
        if not candidate.exists():
            messagebox.showerror("File not found", f"Selected file does not exist:\n{candidate}", parent=root)
            return
        chosen_path = candidate
        root.destroy()

    def cancel() -> None:
        root.destroy()

    ttk.Button(frame, text="Browse...", command=browse).grid(row=1, column=1, padx=(8, 0), pady=(14, 8))

    button_row = ttk.Frame(frame)
    button_row.grid(row=2, column=0, columnspan=3, sticky="e", pady=(8, 0))
    ttk.Button(button_row, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(button_row, text="Use This File", command=confirm).pack(side="right", padx=(0, 8))

    root.bind("<Return>", lambda _event: confirm())
    root.bind("<Escape>", lambda _event: cancel())
    entry.focus_set()
    root.mainloop()

    if chosen_path is None:
        raise PipelineError("No .kicad_pcb file was selected.")
    return chosen_path


def slugify_board_name(board_path: Path) -> str:
    raw_name = board_path.stem.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_")
    return slug or "board"


def slugify_package_name(name: str) -> str:
    raw_name = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_")
    return slug or "pcb"


def build_output_paths(
    board_path: Path,
    copper_json: Path | None,
    stl_output_dir: Path | None,
    defect_output_dir: Path | None,
    pcb_package_name: str | None,
) -> tuple[Path, Path, Path, Path | None]:
    if pcb_package_name:
        package_root = DEFAULT_PCB_LIBRARY_DIR / slugify_package_name(pcb_package_name)
        resolved_copper_json = copper_json.expanduser().resolve() if copper_json is not None else (package_root / "copper_paths.json")
        resolved_stl_output_dir = stl_output_dir.expanduser().resolve() if stl_output_dir is not None else (package_root / "material_partition")
        resolved_defect_output_dir = defect_output_dir.expanduser().resolve() if defect_output_dir is not None else (package_root / "material_partition_defects")
        return resolved_copper_json, resolved_stl_output_dir, resolved_defect_output_dir, package_root

    board_key = slugify_board_name(board_path)
    board_output_root = DEFAULT_OUTPUT_ROOT / board_key
    resolved_copper_json = copper_json.expanduser().resolve() if copper_json is not None else (board_output_root / "copper_paths.json")
    resolved_stl_output_dir = stl_output_dir.expanduser().resolve() if stl_output_dir is not None else (board_output_root / "material_partition")
    resolved_defect_output_dir = defect_output_dir.expanduser().resolve() if defect_output_dir is not None else (board_output_root / "material_partition_defects")
    return resolved_copper_json, resolved_stl_output_dir, resolved_defect_output_dir, None


def write_pcb_package_manifest(
    package_root: Path,
    *,
    package_name: str,
    input_pcb: Path,
    copper_json: Path,
    stl_output_dir: Path,
    defect_output_dir: Path,
    settings_path: Path | None = None,
) -> Path:
    package_root.mkdir(parents=True, exist_ok=True)
    clean_manifest_path = stl_output_dir / "material_partition_manifest.json"
    defect_manifest_path = defect_output_dir / "material_partition_manifest.json"
    payload = {
        "package_name": package_name,
        "source_pcb": str(input_pcb),
        "copper_json": str(copper_json),
        "clean_export_dir": str(stl_output_dir),
        "defect_export_dir": str(defect_output_dir),
        "variants": {
            "clean": str(clean_manifest_path) if clean_manifest_path.exists() else "",
            "defect": str(defect_manifest_path) if defect_manifest_path.exists() else "",
        },
        "settings_snapshot": str(settings_path) if settings_path is not None and settings_path.exists() else "",
    }
    manifest_path = package_root / "pcb_package_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def write_pcb_settings_snapshot(package_root: Path, payload: dict) -> Path:
    package_root.mkdir(parents=True, exist_ok=True)
    settings_path = package_root / "pcb_package_settings.json"
    settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return settings_path


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


def run_base_stl_export(copper_json: Path, output_dir: Path, clearance_mm: float) -> Path:
    export_material_partition(copper_json, output_dir, clearance_mm)
    return output_dir / "material_partition_manifest.json"


def save_pcb_package_from_viewer(
    *,
    input_pcb: Path,
    package_name: str,
    settings_payload: dict,
    selected_outputs: set[str],
    clearance_mm: float,
    kicad_python: str | None,
    log_level: str,
    source_copper_json: Path,
) -> Path:
    package_slug = slugify_package_name(package_name)
    package_root = DEFAULT_PCB_LIBRARY_DIR / package_slug
    copper_json, stl_output_dir, defect_output_dir, _unused_package_root = build_output_paths(
        input_pcb,
        None,
        None,
        None,
        package_slug,
    )
    run_extract(input_pcb, copper_json, kicad_python, log_level)
    run_base_stl_export(copper_json, stl_output_dir, clearance_mm)
    model = load_board_view_model(copper_json)
    export_material_partition_with_defects(
        model,
        output_dir=defect_output_dir,
        overetch=OverEtchSettings(**settings_payload["defects"]["overetch"]),
        mousebite=MouseBiteSettings(**settings_payload["defects"]["mousebite"]),
        underetch=UnderEtchSettings(**settings_payload["defects"]["underetch"]),
        opencircuit=OpenCircuitSettings(**settings_payload["defects"]["open_circuit"]),
        shortcircuit=ShortCircuitSettings(**settings_payload["defects"]["short_circuit"]),
        selected_outputs=selected_outputs or None,
    )
    settings_payload = dict(settings_payload)
    settings_payload.update(
        {
            "package_name": package_name,
            "package_slug": package_slug,
            "source_copper_json": str(source_copper_json),
            "packaged_copper_json": str(copper_json),
            "clean_export_dir": str(stl_output_dir),
            "defect_export_dir": str(defect_output_dir),
        }
    )
    settings_path = write_pcb_settings_snapshot(package_root, settings_payload)
    write_pcb_package_manifest(
        package_root,
        package_name=package_name,
        input_pcb=input_pcb,
        copper_json=copper_json,
        stl_output_dir=stl_output_dir,
        defect_output_dir=defect_output_dir,
        settings_path=settings_path,
    )
    return package_root


def run_viewer(
    copper_json: Path,
    defect_output_dir: Path,
    *,
    input_pcb: Path,
    clearance_mm: float,
    kicad_python: str | None,
    log_level: str,
    initial_package_name: str = "",
) -> None:
    model = load_board_view_model(copper_json)
    viewer = VedoStackupViewer(model, export_output_dir=defect_output_dir)
    StackupControlPanel(
        viewer,
        initial_package_name=initial_package_name,
        save_package_callback=lambda package_name, settings_payload, selected_outputs: save_pcb_package_from_viewer(
            input_pcb=input_pcb,
            package_name=package_name,
            settings_payload=settings_payload,
            selected_outputs=selected_outputs,
            clearance_mm=clearance_mm,
            kicad_python=kicad_python,
            log_level=log_level,
            source_copper_json=copper_json,
        ),
    ).run()


def ensure_copper_json(input_pcb: Path, copper_json: Path, kicad_python: str | None, log_level: str) -> None:
    if copper_json.exists():
        return
    run_extract(input_pcb, copper_json, kicad_python, log_level)


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
    parser.set_defaults(command="viewer")
    parser.add_argument("--input-pcb", type=Path, default=None, help="Optional path to the source .kicad_pcb board.")
    parser.add_argument("--copper-json", type=Path, default=None, help="Optional override for the extracted copper JSON path.")
    parser.add_argument("--stl-output-dir", type=Path, default=None, help="Optional override for the clean STL output directory.")
    parser.add_argument("--defect-output-dir", type=Path, default=None, help="Optional override for defect STL exports.")
    parser.add_argument("--pcb-package-name", default=None, help="Optional named PCB package, for example PCB1.")
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
    return DEFAULT_INPUT_PCB.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "viewer"

    try:
        input_pcb = resolve_input_board(args)
        copper_json, stl_output_dir, defect_output_dir, package_root = build_output_paths(
            input_pcb,
            args.copper_json,
            args.stl_output_dir,
            args.defect_output_dir,
            args.pcb_package_name,
        )
        package_name = args.pcb_package_name.strip() if isinstance(args.pcb_package_name, str) else ""

        if command == "extract":
            run_extract(input_pcb, copper_json, args.kicad_python, args.log_level)
            if package_root is not None:
                write_pcb_package_manifest(
                    package_root,
                    package_name=package_name or slugify_package_name(input_pcb.stem),
                    input_pcb=input_pcb,
                    copper_json=copper_json,
                    stl_output_dir=stl_output_dir,
                    defect_output_dir=defect_output_dir,
                )
            return 0
        if command == "stl":
            run_base_stl_export(copper_json, stl_output_dir, args.clearance_mm)
            if package_root is not None:
                write_pcb_package_manifest(
                    package_root,
                    package_name=package_name or slugify_package_name(input_pcb.stem),
                    input_pcb=input_pcb,
                    copper_json=copper_json,
                    stl_output_dir=stl_output_dir,
                    defect_output_dir=defect_output_dir,
                )
            return 0
        if command == "viewer":
            ensure_copper_json(input_pcb, copper_json, args.kicad_python, args.log_level)
            if package_root is not None:
                write_pcb_package_manifest(
                    package_root,
                    package_name=package_name or slugify_package_name(input_pcb.stem),
                    input_pcb=input_pcb,
                    copper_json=copper_json,
                    stl_output_dir=stl_output_dir,
                    defect_output_dir=defect_output_dir,
                )
            run_viewer(
                copper_json,
                defect_output_dir,
                input_pcb=input_pcb,
                clearance_mm=args.clearance_mm,
                kicad_python=args.kicad_python,
                log_level=args.log_level,
                initial_package_name=package_name,
            )
            return 0
        if command == "defect-export":
            run_defect_export(copper_json, defect_output_dir, args)
            if package_root is not None:
                write_pcb_package_manifest(
                    package_root,
                    package_name=package_name or slugify_package_name(input_pcb.stem),
                    input_pcb=input_pcb,
                    copper_json=copper_json,
                    stl_output_dir=stl_output_dir,
                    defect_output_dir=defect_output_dir,
                )
            return 0

        if command == "full":
            run_extract(input_pcb, copper_json, args.kicad_python, args.log_level)
            run_base_stl_export(copper_json, stl_output_dir, args.clearance_mm)
            if package_root is not None:
                write_pcb_package_manifest(
                    package_root,
                    package_name=package_name or slugify_package_name(input_pcb.stem),
                    input_pcb=input_pcb,
                    copper_json=copper_json,
                    stl_output_dir=stl_output_dir,
                    defect_output_dir=defect_output_dir,
                )
            if not getattr(args, "skip_viewer", False):
                run_viewer(
                    copper_json,
                    defect_output_dir,
                    input_pcb=input_pcb,
                    clearance_mm=args.clearance_mm,
                    kicad_python=args.kicad_python,
                    log_level=args.log_level,
                    initial_package_name=package_name,
                )
            return 0

        raise PipelineError(f"Unknown command: {command}")
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
