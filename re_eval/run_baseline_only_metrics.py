#!/usr/bin/env python3
"""
Run baseline-only evaluation: for each baseline entry in config, run that baseline's
--eval_only script, parse test_results.txt, compute cFID-FC in-process, and collect into one TSV.

cFID-FC is computed inside this script (via run_cfid_baseline.compute_cfid_baseline) and written
to load_dir/cfid_fc.txt. Use a config that contains only baseline entries (e.g. fm_baseline_only.json
from: python re_eval/re_eval/build_combined_config.py --baseline_only).

Usage (from repo root):
  python re_eval/run_baseline_only_metrics.py \
    --config re_eval/re_eval/fm_baseline_only.json \
    --out_csv re_eval/results_baseline_only.tsv
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure run_cfid_baseline can be imported (and its deps: run_discriminative_score, fc_utils, baselines)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "re_eval"))

try:
    from run_cfid_baseline import compute_cfid_baseline
except ImportError:
    compute_cfid_baseline = None

BASELINE_SCRIPTS = {
    "timegan": REPO_ROOT / "baselines" / "train_timegan.py",
    "timevae": REPO_ROOT / "baselines" / "timevae_baseline.py",
    "ddpm": REPO_ROOT / "baselines" / "ddpm_baseline.py",
    "diffusion_ts": REPO_ROOT / "baselines" / "diffusion_ts_baseline.py",
    "lstm_gan": REPO_ROOT / "baselines" / "train_lstm_gan.py",
}


def parse_cfid_fc(load_dir: Path) -> str:
    """Parse cfid_fc.txt in load_dir if present (e.g. written by timevae). Returns value as string or ''."""
    path = load_dir / "cfid_fc.txt"
    if not path.exists():
        return ""
    try:
        line = path.read_text(encoding="utf-8").strip()
        if "\t" in line:
            return line.split("\t")[-1].strip()
    except Exception:
        pass
    return ""


def parse_test_results(path: Path) -> dict:
    """Parse test_results.txt for MSE, MAE, PSD/freq_diff, FC similarity, precision@5."""
    out = {"mae": None, "psd": None, "fc_similarity": None, "fc_precision_at_5": None}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    # MAE
    m = re.search(r"MAE.*?([0-9.e+-]+)\s*±", text, re.IGNORECASE)
    if m:
        try:
            out["mae"] = float(m.group(1).strip())
        except ValueError:
            pass
    # PSD / freq diff
    m = re.search(
        r"(?:Freq diff|Frequency Difference|PSD MAE|PSD\s*\([^)]*\)|absolute power spectrum difference).*?([0-9.e+-]+)\s*±",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            out["psd"] = float(m.group(1).strip())
        except ValueError:
            pass
    # FC similarity
    m = re.search(
        r"(?:FC sim(?:ilarity)?|Functional Connectivity Similarity).*?([0-9.e+-]+)\s*±",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            out["fc_similarity"] = float(m.group(1).strip())
        except ValueError:
            pass
    # k=5% precision (TimeVAE/diffusion_ts write "  k=5%:  Precision X.XX ± ..."; accept "nan")
    for pattern in [
        r"k=5%\s*:\s*Precision\s+([0-9.e+-]+|nan)\s*±",
        r"k=5%[^\n]*?Precision\s+([0-9.e+-]+|nan)",  # fallback: same line, any spacing
        r"k\s*=\s*5\s*%[^\n]*?Precision\s+([0-9.e+-]+|nan)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            if val != "nan":
                try:
                    out["fc_precision_at_5"] = float(val)
                except ValueError:
                    pass
            break
    return out


def run_baseline_eval(script_path: Path, load_dir: str, task_name: str, data_root: str, task_root: str) -> bool:
    """Run baseline script with --eval_only. Returns True if success."""
    cmd = [
        sys.executable,
        str(script_path),
        "--eval_only",
        "--save_dir", load_dir,
        "--task_name", task_name,
        "--data_root", data_root,
        "--task_root", task_root,
    ]
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [WARN] {script_path.name} exited {result.returncode}: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [WARN] Timeout")
        return False
    except Exception as e:
        print(f"  [WARN] {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Run baseline --eval_only for each config entry and collect TSV")
    p.add_argument("--config", type=str, default=None, help="JSON config (default: re_eval/re_eval/fm_baseline_only.json)")
    p.add_argument("--out_csv", type=str, default=None, help="Output TSV path")
    p.add_argument("--results_base", type=str, default=None,
                   help="Override baseline load_dir: use <results_base>/<model>/<task> (default: use load_dir from config; set to repo/results to match baselines/slurm)")
    p.add_argument("--data_root", type=str, default=None, help="HCP rest data root (default: env DATA_ROOT or repo default)")
    p.add_argument("--task_root", type=str, default=None, help="HCP task data root (default: env TASK_ROOT or repo default)")
    p.add_argument("--skip_run", action="store_true", help="Only parse existing test_results.txt; do not run --eval_only")
    args = p.parse_args()

    import os
    data_root = args.data_root or os.getenv("DATA_ROOT", "./data/hcp-resting-fc")
    task_root = args.task_root or os.getenv("TASK_ROOT", "./data/hcp-task-ts")

    config_path = args.config or str(REPO_ROOT / "re_eval" / "re_eval" / "fm_baseline_only.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    if not isinstance(config, list):
        config = [config]

    # Only baseline model types
    baseline_types = set(BASELINE_SCRIPTS.keys())
    entries = [e for e in config if e.get("model_type") in baseline_types]
    if not entries:
        print("No baseline entries (model_type in timegan, timevae, ddpm, diffusion_ts) in config.")
        return

    results_base = Path(args.results_base).resolve() if args.results_base else None
    if results_base:
        print(f"Results base (override): {results_base}")
    print(f"Config: {config_path}  |  Baseline entries: {len(entries)}")
    if not args.skip_run:
        print("Running each baseline --eval_only (this may take a while)...")
    col_order = ["name", "mae", "psd", "fc_similarity", "fc_precision_at_5", "cfid_fc", "discriminative_score"]
    rows = []

    for idx, entry in enumerate(entries):
        name = entry.get("name", f"baseline_{idx}")
        load_dir = entry.get("load_dir")
        task_name = entry.get("task_name", "emotion")
        model_type = entry.get("model_type")
        if results_base is not None:
            load_dir = str(results_base / model_type / task_name)
        if not load_dir or model_type not in BASELINE_SCRIPTS:
            continue
        script_path = BASELINE_SCRIPTS[model_type]
        if not script_path.exists():
            print(f"[Skip] {name}: script not found {script_path}")
            rows.append({c: name if c == "name" else "" for c in col_order})
            continue

        print(f"  [{idx+1}/{len(entries)}] {name} ...", flush=True)
        cfid_val = ""
        if not args.skip_run:
            run_baseline_eval(script_path, load_dir, task_name, data_root, task_root)
            # Compute cFID-FC in-process (writes load_dir/cfid_fc.txt)
            if compute_cfid_baseline is not None:
                cfid_float = compute_cfid_baseline(
                    load_dir, model_type, task_name, data_root, task_root,
                    write_file=True,
                )
                if cfid_float is not None:
                    cfid_val = f"{cfid_float:.6f}"
                    print(f"    cFID-FC={cfid_float:.6f}", flush=True)
        if not cfid_val:
            cfid_val = parse_cfid_fc(Path(load_dir))
        tr_path = Path(load_dir) / "test_results.txt"
        meta = parse_test_results(tr_path)
        row = {
            "name": name,
            "mae": meta["mae"] if meta["mae"] is not None else "",
            "psd": meta["psd"] if meta["psd"] is not None else "",
            "fc_similarity": meta["fc_similarity"] if meta["fc_similarity"] is not None else "",
            "fc_precision_at_5": meta["fc_precision_at_5"] if meta["fc_precision_at_5"] is not None else "",
            "cfid_fc": cfid_val,
            "discriminative_score": "",
        }
        rows.append(row)

    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in col_order))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\t".join(col_order) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in col_order) + "\n")
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
