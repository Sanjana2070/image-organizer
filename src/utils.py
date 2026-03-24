import hashlib
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


def ensure_dir(path: Path) -> Path:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    """
    Create multiple directories.
    """
    for path in paths:
        ensure_dir(path)


def reset_dir(path: Path) -> Path:
    """
    Delete and recreate a directory.
    Use with care.
    """
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(path: Path) -> str:
    """
    Return a filesystem-safe version of a file stem.
    """
    stem = path.stem.strip().lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "", stem)
    return stem or "file"


def get_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA1 hash for a file.
    Useful for stable IDs or duplicate checks.
    """
    sha1 = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def make_image_id(path: Path) -> str:
    """
    Create a stable image ID from filename stem + short file hash.
    """
    stem = safe_stem(path)
    short_hash = get_file_hash(path)[:10]
    return f"{stem}_{short_hash}"


def make_face_id(image_id: str, face_index: int) -> str:
    """
    Create a stable face ID for a detected face in an image.
    """
    return f"{image_id}_face_{face_index}"


def make_unique_path(directory: Path, filename: str) -> Path:
    """
    Create a non-colliding filepath inside a directory.
    Example: photo.jpg -> photo_1.jpg if photo.jpg already exists.
    """
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename

    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while True:
        new_candidate = directory / f"{stem}_{counter}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        counter += 1


def bbox_width_height(bbox) -> tuple[int, int]:
    """
    Compute width and height from bbox [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    width = max(0, int(x2 - x1))
    height = max(0, int(y2 - y1))
    return width, height


def bbox_area(bbox) -> int:
    """
    Compute bbox area from [x1, y1, x2, y2].
    """
    w, h = bbox_width_height(bbox)
    return w * h


def log(message: str, verbose: bool = True) -> None:
    """
    Simple logger.
    """
    if verbose:
        print(message)


def to_posix_str(path: Optional[Path]) -> Optional[str]:
    """
    Convert Path to POSIX string for JSON serialization.
    """
    if path is None:
        return None
    return path.as_posix()