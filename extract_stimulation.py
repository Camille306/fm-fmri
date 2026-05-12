import re
import zipfile
from pathlib import Path
import glob

# Set to True to print paths in the zip (limited list); False to run extraction
LIST_ZIP_CONTENTS = False

def list_zip_contents(zip_paths, max_items=50):
    """Print first max_items paths in the zip(s) so you can see the directory structure."""
    for zip_file in zip_paths:
        print(f"\n=== Contents of {zip_file} (first {max_items}) ===\n")
        with zipfile.ZipFile(zip_file, "r") as z:
            names = z.namelist()
            for name in names[:max_items]:
                print(f"  {name}")
            if len(names) > max_items:
                print(f"  ... and {len(names) - max_items} more")
        print()

def subject_id_from_zip_path(zip_path: str) -> str:
    """Get 6-digit subject ID from zip filename (e.g. 100206_Task3TRecommended.zip -> 100206)."""
    name = Path(zip_path).stem
    match = re.match(r"^(\d{6})", name)
    if match:
        return match.group(1)
    raise ValueError(f"Zip path has no 6-digit subject ID: {zip_path}")

# Zip archives use forward slashes; extract all members under a path prefix
def extract_members(z, prefix: str, path: str):
    prefix = prefix.replace("\\", "/")
    extracted = []
    for name in z.namelist():
        norm = name.replace("\\", "/")
        if norm == prefix or norm.startswith(prefix + "/"):
            z.extract(name, path=path)
            extracted.append(name)
    return extracted

def extract_members_to(z, prefix: str, out_dir: Path):
    """Extract all members under prefix into out_dir, preserving only basenames (e.g. EVs/*.txt)."""
    prefix = prefix.replace("\\", "/")
    extracted = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in z.namelist():
        norm = name.replace("\\", "/")
        if norm == prefix or norm.startswith(prefix + "/"):
            if name.endswith("/"):
                continue
            data = z.read(name)
            target = out_dir / Path(name).name
            target.write_bytes(data)
            extracted.append(str(target))
    return extracted

TASKS = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
EXTRACT_BASE = Path("extracted_txt")

zip_path = glob.glob(r"./downloads\*Task3TRecommended.zip")
print(f"Found {len(zip_path)} zip file(s): {zip_path}")

if LIST_ZIP_CONTENTS:
    list_zip_contents(zip_path)
else:
    for zip_file in zip_path:
        subject_id = subject_id_from_zip_path(zip_file)
        print(f"Opening: {zip_file} (subject {subject_id})")
        list_zip_contents([zip_file], max_items=50)
        with zipfile.ZipFile(zip_file, "r") as z:
            for task in TASKS:
                prefix = f"{subject_id}/MNINonLinear/Results/tfMRI_{task}_RL/EVs"
                out_dir = EXTRACT_BASE / "EVs" / task / subject_id
                print(f"  Prefix: {prefix} -> {out_dir}")
                extracted = extract_members_to(z, prefix, out_dir)
                print(f"  Extracted {len(extracted)} item(s) to {out_dir}")
        print("Done.")