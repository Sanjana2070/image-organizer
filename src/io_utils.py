import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional

import cv2
import numpy as np

from config import SUPPORTED_IMAGE_EXTENSIONS
from utils import make_unique_path

from PIL import Image

def is_image_file(path: Path) -> bool:
    """
    Check whether a file has a supported image extension.
    """
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def list_images(input_dir: Path, recursive: bool = False) -> List[Path]:
    """
    List all supported images in a directory.
    Returns sorted list of Paths.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if recursive:
        files = [p for p in input_dir.rglob("*") if is_image_file(p)]
    else:
        files = [p for p in input_dir.iterdir() if is_image_file(p)]

    return sorted(files)


def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Robust image loader:
    1. Try OpenCV
    2. Fallback to PIL (handles more corrupted JPEGs)
    3. Return None if both fail
    """

    # --- Try OpenCV ---
    try:
        image = cv2.imread(str(path))
        if image is not None:
            return image
    except Exception as e:
        print(f"[OpenCV ERROR] {path}: {e}")

    # --- Fallback to PIL ---
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            image = np.array(img)
            # Convert RGB -> BGR to stay consistent with OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
    except Exception as e:
        print(f"[PIL ERROR] {path}: {e}")

    # --- Fully failed ---
    print(f"[FAILED IMAGE] {path}")
    return None


def save_image(path: Path, image: np.ndarray) -> None:
    """
    Save image to disk. Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image to: {path}")


def copy_file(src: Path, dst: Path, unique_name: bool = False) -> Path:
    """
    Copy a file from src to dst.
    If dst is a directory, preserve original filename.
    If unique_name=True, avoid collisions.
    Returns final destination path.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    if dst.exists() and dst.is_dir():
        final_path = dst / src.name
    elif dst.suffix:
        final_path = dst
    else:
        dst.mkdir(parents=True, exist_ok=True)
        final_path = dst / src.name

    final_path.parent.mkdir(parents=True, exist_ok=True)

    if unique_name:
        final_path = make_unique_path(final_path.parent, final_path.name)

    shutil.copy2(src, final_path)
    return final_path


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """
    Save Python object to JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """
    Load JSON file.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(records: Iterable[dict], path: Path) -> None:
    """
    Save list of dictionaries to CSV.
    Assumes all dicts share the same keys.
    """
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        with path.open("w", newline="", encoding="utf-8") as f:
            pass
        return

    fieldnames = list(records[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_csv(path: Path) -> list[dict]:
    """
    Load CSV into list of dictionaries.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_npy(array: np.ndarray, path: Path) -> None:
    """
    Save NumPy array to .npy file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_npy(path: Path) -> np.ndarray:
    """
    Load NumPy array from .npy file.
    """
    return np.load(path)


def ensure_parent(path: Path) -> None:
    """
    Ensure parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)