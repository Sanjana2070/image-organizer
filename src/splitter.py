from pathlib import Path
from typing import Any, Dict, List

from config import (
    DETECTIONS_JSON,
    IMAGE_INDEX_CSV,
    INPUT_DIR,
    NO_FACES_DIR,
    VERBOSE,
)
from detector import FaceDetector
from io_utils import copy_file, list_images, load_image, save_csv, save_json
from utils import log, make_face_id, make_image_id, to_posix_str


def build_image_record(
    image_path: Path,
    detection_result: Dict[str, Any],
    no_faces_output_path: Path | None,
) -> Dict[str, Any]:
    """
    Build a serializable image-level metadata record.
    """
    image_id = make_image_id(image_path)

    faces = detection_result.get("faces", [])
    serialized_faces = []

    for local_face_index, face in enumerate(faces):
        face_record = {
            "face_id": make_face_id(image_id, local_face_index),
            "face_index": local_face_index,
            "bbox": face["bbox"],
            "score": face["score"],
            "width": face["width"],
            "height": face["height"],
            "kps": face.get("kps"),
            "embedding": face.get("embedding"),
        }
        serialized_faces.append(face_record)

    return {
        "image_id": image_id,
        "original_path": to_posix_str(image_path),
        "filename": image_path.name,
        "has_face": detection_result["has_face"],
        "num_faces": detection_result["num_faces"],
        "no_faces_output_path": to_posix_str(no_faces_output_path),
        "faces": serialized_faces,
    }


def build_image_index_row(image_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a flat CSV row from an image-level record.
    """
    return {
        "image_id": image_record["image_id"],
        "filename": image_record["filename"],
        "original_path": image_record["original_path"],
        "has_face": image_record["has_face"],
        "num_faces": image_record["num_faces"],
        "no_faces_output_path": image_record["no_faces_output_path"],
    }


def split_images_by_face_presence(
    detector: FaceDetector,
    input_dir: Path = INPUT_DIR,
    no_faces_dir: Path = NO_FACES_DIR,
    detections_json_path: Path = DETECTIONS_JSON,
    image_index_csv_path: Path = IMAGE_INDEX_CSV,
    recursive: bool = False,
    verbose: bool = VERBOSE,
) -> List[Dict[str, Any]]:
    """
    Process all images in input_dir:
    - detect faces
    - copy only images without faces to no_faces_dir
    - save metadata JSON and CSV

    Returns:
        List of image-level metadata records.
    """
    image_paths = list_images(input_dir, recursive=recursive)
    log(f"Found {len(image_paths)} image(s) in {input_dir}", verbose)

    image_records: List[Dict[str, Any]] = []
    image_index_rows: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        log(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}", verbose)

        image = load_image(image_path)
        if image is None:
            log(f"  Skipping unreadable image: {image_path}", verbose)
            continue

        detection_result = detector.process_image(image, image_path=str(image_path))

        no_faces_output_path = None

        if detection_result["has_face"]:
            log(
                f"  Detected {detection_result['num_faces']} face(s) -> keeping original path only",
                verbose,
            )
        else:
            no_faces_output_path = copy_file(image_path, no_faces_dir, unique_name=True)
            log(f"  No face detected -> copied to {no_faces_output_path}", verbose)

        image_record = build_image_record(
            image_path=image_path,
            detection_result=detection_result,
            no_faces_output_path=no_faces_output_path,
        )
        image_records.append(image_record)
        image_index_rows.append(build_image_index_row(image_record))

    save_json(image_records, detections_json_path)
    save_csv(image_index_rows, image_index_csv_path)

    log(f"Saved detections metadata to: {detections_json_path}", verbose)
    log(f"Saved image index CSV to: {image_index_csv_path}", verbose)

    return image_records