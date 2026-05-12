#!/usr/bin/env python3
"""
Test script for EV loading. Loads EVs for a given subject and writes an HTML report
with the loaded (T, K) array and the raw contents of each .txt file.
Uses only pathlib and numpy (no torch/dataset import) so it runs in minimal envs.

Usage (from repo root):
  python fm-fmri/test_ev_loading.py --subject_id 100206 [--task_name emotion] [--ev_root .] [--output report.html]

Usage (from fm-fmri):
  python test_ev_loading.py --subject_id 100206 --ev_root ..
"""

import argparse
from pathlib import Path

import numpy as np

# Mirror dataset.py: task name -> EV folder under extracted_txt/EVs/
TASK_TO_EV_FOLDER = {
    "emotion": "EMOTION",
    "gambling": "GAMBLING",
    "language": "LANGUAGE",
    "motor": "MOTOR",
    "relational": "RELATIONAL",
    "social": "SOCIAL",
    "WM": "WM",
}


def get_ev_dir(ev_root: Path, task_name: str, subject_id: str) -> Path:
    """Path to subject EV dir: ev_root/extracted_txt/EVs/{task_folder}/{subject_id}/"""
    task_folder = TASK_TO_EV_FOLDER.get(task_name.strip(), task_name.upper())
    return ev_root / "extracted_txt" / "EVs" / task_folder / subject_id


def load_ev_subject(ev_dir: Path, task_name: str) -> np.ndarray:
    """
    Load EV: full content of each condition .txt file, add condition column (1, 2, ...), concatenate
    vertically (fear rows then neut rows). Sync is repeated as the first column (for time alignment).
    """
    files = sorted([f for f in ev_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])
    sync_path = next((f for f in files if f.name.lower() == "sync.txt"), None)
    if sync_path is None:
        raise FileNotFoundError(f"Sync.txt not found in {ev_dir}")
    sync_val = float(np.loadtxt(str(sync_path), dtype=np.float64, ndmin=0).flat[0])

    other_files = sorted([f for f in files if f.name.lower() != "sync.txt"])
    blocks = []
    for cond_code, f in enumerate(other_files, start=1):
        try:
            arr = np.loadtxt(str(f), dtype=np.float64, ndmin=2)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = arr.astype(np.float32)
            cond_col = np.full((arr.shape[0], 1), float(cond_code), dtype=np.float32)
            block = np.hstack([arr, cond_col])
            blocks.append(block)
        except Exception:
            continue

    if not blocks:
        raise FileNotFoundError(f"No condition .txt files in {ev_dir}")
    ev = np.vstack(blocks)
    sync_col = np.full((ev.shape[0], 1), sync_val, dtype=np.float32)
    return np.hstack([sync_col, ev])


def read_txt_file(path: Path) -> str:
    """Read file as text."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        return f"[Error reading file: {e}]"


def array_to_html_table(arr: np.ndarray, max_rows: int = 50, max_cols: int = 20) -> str:
    """Convert numpy array to HTML table. Truncate if too large."""
    rows, cols = arr.shape
    show_rows = min(rows, max_rows)
    show_cols = min(cols, max_cols)
    truncated = rows > max_rows or cols > max_cols
    html = ['<table class="ev-table">']
    html.append("<thead><tr><th>row</th>")
    for c in range(show_cols):
        html.append(f"<th>col {c}</th>")
    if cols > max_cols:
        html.append("<th>...</th>")
    html.append("</tr></thead><tbody>")
    for r in range(show_rows):
        html.append(f"<tr><td>{r}</td>")
        for c in range(show_cols):
            val = arr[r, c]
            html.append(f"<td>{val:.6g}</td>")
        if cols > max_cols:
            html.append("<td>...</td>")
        html.append("</tr>")
    if rows > max_rows:
        html.append(f"<tr><td colspan={show_cols + 2}>... ({rows - max_rows} more rows)</td></tr>")
    html.append("</tbody></table>")
    if truncated:
        html.append(f"<p><em>Showing first {show_rows} rows × {show_cols} cols (full shape: {rows} × {cols})</em></p>")
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser(description="Test EV loading and write HTML report")
    parser.add_argument("--subject_id", type=str, required=True, help="Subject ID (e.g. 100206)")
    parser.add_argument("--task_name", type=str, default="emotion", help="Task name (e.g. emotion)")
    parser.add_argument("--ev_root", type=str, default=".", help="Root path for EVs (contains extracted_txt/EVs/...)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: current directory/ev_loading_<subject>_<task>.html)")
    args = parser.parse_args()

    ev_root = Path(args.ev_root).resolve()
    if not ev_root.exists():
        print(f"Error: ev_root does not exist: {ev_root}")
        return 1

    subject_id = args.subject_id
    ev_dir = get_ev_dir(ev_root, args.task_name, subject_id)

    if not ev_dir.exists():
        print(f"Error: EV directory not found: {ev_dir}")
        return 1

    # Raw .txt file contents
    txt_files = sorted([f for f in ev_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"])
    file_contents = {f.name: read_txt_file(f) for f in txt_files}

    # Load EV (T, K)
    try:
        ev_array = load_ev_subject(ev_dir, args.task_name)
    except Exception as e:
        print(f"Error loading EV: {e}")
        ev_array = None

    # Output path: default to current working directory so the file appears where you ran the command
    out_path = Path(args.output).resolve() if args.output else (Path.cwd() / f"ev_loading_{subject_id}_{args.task_name}.html")

    title = f"EV loading test: subject {subject_id}, task {args.task_name}"
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1200px; }",
        "h1 { font-size: 1.25rem; }",
        "h2 { font-size: 1rem; margin-top: 1.5rem; border-bottom: 1px solid #ccc; }",
        "h3 { font-size: 0.95rem; margin-top: 1rem; }",
        "pre { background: #f5f5f5; padding: 0.75rem; overflow-x: auto; }",
        "table.ev-table { border-collapse: collapse; font-size: 0.85rem; }",
        "table.ev-table th, table.ev-table td { border: 1px solid #ccc; padding: 0.25rem 0.5rem; text-align: right; }",
        "table.ev-table th { background: #eee; }",
        "table.ev-table td:first-child { text-align: right; font-weight: 500; }",
        ".path { color: #666; font-size: 0.9rem; }",
        ".shape { font-weight: 600; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p><strong>Subject ID:</strong> {subject_id}</p>",
        f"<p><strong>Task:</strong> {args.task_name}</p>",
        f"<p class='path'><strong>EV directory:</strong> <code>{ev_dir}</code></p>",
    ]

    # Raw .txt contents
    html_parts.append("<h2>Raw .txt file contents</h2>")
    for fname in sorted(file_contents.keys()):
        content = file_contents[fname]
        html_parts.append(f"<h3>{fname}</h3>")
        html_parts.append("<pre>")
        html_parts.append(content.replace("<", "&lt;").replace(">", "&gt;"))
        html_parts.append("</pre>")

    # Loaded EV array
    html_parts.append("<h2>Loaded EV array (concatenated + condition code + Sync)</h2>")
    if ev_array is not None:
        html_parts.append(f"<p class='shape'>Shape: {ev_array.shape[0]} × {ev_array.shape[1]} — cols: Sync (repeated, for time alignment), onset, duration, amplitude, condition (1=fear, 2=neut, ...)</p>")
        html_parts.append(array_to_html_table(ev_array))
    else:
        html_parts.append("<p>Load failed (see error above).</p>")

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    abs_path = out_path.resolve()
    print(f"Wrote: {abs_path}")
    if ev_array is not None:
        print(f"  EV shape: {ev_array.shape}")
    print("  Open this file in a browser; if you see old values, try Ctrl+F5 (hard refresh) to avoid cache.")
    return 0


if __name__ == "__main__":
    exit(main())
