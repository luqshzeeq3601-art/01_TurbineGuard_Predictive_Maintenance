"""Download and extract NASA C-MAPSS FD001 dataset with provenance and checksum verification."""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
EXPECTED_HASHES = {
    "outer_archive": "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2",
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}
REQUIRED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"]
CITATION = (
    "A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, "
    "NASA Prognostics Data Repository, NASA Ames Research Center."
)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_extract_zip(zip_bytes: bytes, target_dir: Path, allowed_filenames: list[str]) -> dict[str, str]:
    """Safely extract specified files from zip bytes preventing directory traversal (Zip Slip)."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_hashes = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for member in z.namelist():
            # Security check against path traversal in zip entry name
            resolved_member_path = (target_dir / member).resolve()
            if not resolved_member_path.is_relative_to(target_dir):
                raise ValueError(f"Security error: Zip member {member} attempts path traversal outside target directory.")

            filename = os.path.basename(member)
            if filename in allowed_filenames:
                dest_path = (target_dir / filename).resolve()
                if not dest_path.is_relative_to(target_dir):
                    raise ValueError(f"Security error: Destination {dest_path} is outside target directory.")
                
                content = z.read(member)
                dest_path.write_bytes(content)
                extracted_hashes[filename] = compute_sha256(content)
    
    return extracted_hashes


def download_and_extract_fd001(output_dir: Path, manifest_path: Path) -> dict:
    """Download the outer archive, extract inner CMAPSSData.zip, extract FD001 files, and write manifest."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NASA C-MAPSS dataset from {DATASET_URL} ...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    outer_archive_bytes = response.content
    outer_hash = compute_sha256(outer_archive_bytes)
    print(f"Outer archive downloaded ({len(outer_archive_bytes)} bytes, SHA-256: {outer_hash})")

    # Inspect outer zip
    inner_zip_bytes = None
    with zipfile.ZipFile(io.BytesIO(outer_archive_bytes)) as outer_zip:
        for name in outer_zip.namelist():
            if name.endswith("CMAPSSData.zip"):
                inner_zip_bytes = outer_zip.read(name)
                break
    
    if inner_zip_bytes is None:
        raise ValueError("Could not find nested CMAPSSData.zip inside outer archive.")
    
    inner_hash = compute_sha256(inner_zip_bytes)
    print(f"Found inner CMAPSSData.zip ({len(inner_zip_bytes)} bytes, SHA-256: {inner_hash})")

    extracted_hashes = safe_extract_zip(inner_zip_bytes, output_dir, REQUIRED_FILES)
    print(f"Extracted {len(extracted_hashes)} files to {output_dir}")

    # Check that required files were extracted
    missing = [f for f in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"] if f not in extracted_hashes]
    if missing:
        raise ValueError(f"Missing required extracted files: {missing}")

    # Verify checksums against expected
    hash_checks = {}
    for fname, expected in EXPECTED_HASHES.items():
        if fname == "outer_archive":
            actual = outer_hash
        else:
            actual = extracted_hashes.get(fname)
        
        matches = (actual == expected) if actual else False
        hash_checks[fname] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            print(f"WARNING: Checksum mismatch for {fname}: expected {expected}, got {actual}")

    manifest_data = {
        "dataset_id": "FD001",
        "source_url": DATASET_URL,
        "download_timestamp_utc": datetime.now(UTC).isoformat(),
        "outer_archive_sha256": outer_hash,
        "inner_archive_sha256": inner_hash,
        "citation": CITATION,
        "terms": "NASA Open Data / Acknowledge repository and contributors (Saxena & Goebel 2008)",
        "files": {
            fname: {
                "sha256": extracted_hashes[fname],
                "size_bytes": (output_dir / fname).stat().st_size,
                "path": str((output_dir / fname).relative_to(output_dir.parent.parent.parent)),
            }
            for fname in extracted_hashes
        },
        "hash_checks": hash_checks,
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Saved source manifest to {manifest_path}")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Acquire and extract NASA C-MAPSS dataset.")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset identifier (e.g. FD001)")
    parser.add_argument("--output", type=str, default="data/raw/FD001", help="Target output directory")
    parser.add_argument("--manifest", type=str, default="data/source_manifest.json", help="Manifest output path")
    args = parser.parse_args()

    if args.dataset != "FD001":
        print(f"Error: Dataset {args.dataset} is outside current MVP scope (FD001 only).", file=sys.stderr)
        sys.exit(1)

    download_and_extract_fd001(Path(args.output), Path(args.manifest))


if __name__ == "__main__":
    main()
