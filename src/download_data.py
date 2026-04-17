from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


ROBOTCAR_SMALL_SAMPLE_URL = "https://robotcar-dataset.robots.ox.ac.uk/downloads/sample_small.tar"
ARCHIVE_NAME = "sample_small.tar"
REQUIRED_FILES = {"gps.csv", "ins.csv"}


def download_robotcar_subset(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / ARCHIVE_NAME

    if not archive_path.exists():
        urllib.request.urlretrieve(ROBOTCAR_SMALL_SAMPLE_URL, archive_path)

    extracted_paths = extract_trajectory_files(archive_path, data_dir)
    missing_files = REQUIRED_FILES.difference(extracted_paths)
    if missing_files:
        raise FileNotFoundError(f"Missing required files after extraction: {sorted(missing_files)}")
    return extracted_paths


def extract_trajectory_files(archive_path: Path, destination_dir: Path) -> dict[str, Path]:
    extracted_paths: dict[str, Path] = {}

    with tarfile.open(archive_path, "r") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers()}
        for file_name in REQUIRED_FILES:
            destination_path = destination_dir / file_name
            if destination_path.exists():
                extracted_paths[file_name] = destination_path
                continue

            member = members.get(file_name)
            if member is None:
                continue

            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                continue

            with destination_path.open("wb") as output_file:
                output_file.write(extracted_file.read())
            extracted_paths[file_name] = destination_path

    return extracted_paths

