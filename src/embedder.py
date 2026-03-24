from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
from typing import Optional

import numpy as np

from config import (
    DETECTIONS_JSON,
    EMBEDDINGS_NPY,
    EMBEDDING_RECORDS_JSON,
    VERBOSE,
)
from io_utils import load_json, save_json, save_npy
from utils import log


from PIL import Image


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception:
        return None

class FaceEmbedder:
    """
    Builds face-level embedding records from detections.json.

    Expected input:
    - image-level detection records from splitter.py
    - each face record contains:
        - face_id
        - bbox
        - score
        - embedding

    Outputs:
    - embeddings.npy
    - embedding_records.json

    Note:
    - This version does NOT save global face crops
    - It reads directly from original image paths
    """

    def __init__(
        self,
        verbose: bool = VERBOSE,
    ) -> None:
        self.verbose = verbose

    @staticmethod
    def _clip_bbox_to_image(
        bbox: List[int],
        image_shape: Tuple[int, int, int],
    ) -> List[int]:
        """
        Clip bbox [x1, y1, x2, y2] so it stays inside image bounds.
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))

        return [x1, y1, x2, y2]

    @staticmethod
    def _is_valid_bbox(bbox: List[int]) -> bool:
        """
        Check if bbox defines a non-empty region.
        """
        x1, y1, x2, y2 = bbox
        return x2 > x1 and y2 > y1

    def build_face_record(
        self,
        image_record: Dict[str, Any],
        face: Dict[str, Any],
        clipped_bbox: List[int],
        embedding_index: int,
    ) -> Dict[str, Any]:
        """
        Build a serializable face-level metadata record.
        """
        return {
            "face_id": face["face_id"],
            "image_id": image_record["image_id"],
            "filename": image_record["filename"],
            "original_image_path": image_record["original_path"],
            "bbox": clipped_bbox,
            "score": face["score"],
            "width": clipped_bbox[2] - clipped_bbox[0],
            "height": clipped_bbox[3] - clipped_bbox[1],
            "kps": face.get("kps"),
            "embedding_index": embedding_index,
        }

    def process_image_record(
        self,
        image_record: Dict[str, Any],
        embeddings_list: List[np.ndarray],
        embedding_records: List[Dict[str, Any]],
    ) -> None:
        """
        Process one image-level record:
        - load original image
        - validate / clip each detected bbox
        - append embedding vector
        - append face-level metadata record

        No crops are saved here.
        """
        if not image_record.get("has_face", False):
            return

        original_path = image_record.get("original_path")
        if not original_path:
            log(f"Skipping image with missing original_path: {image_record['filename']}", self.verbose)
            return

        image_path = Path(original_path)
        image = load_image(image_path)

        if image is None:
            log(f"Skipping unreadable image: {image_path}", self.verbose)
            return

        faces = image_record.get("faces", [])
        for face in faces:
            embedding = face.get("embedding")
            if embedding is None:
                log(f"Skipping face with missing embedding: {face['face_id']}", self.verbose)
                continue

            clipped_bbox = self._clip_bbox_to_image(face["bbox"], image.shape)
            if not self._is_valid_bbox(clipped_bbox):
                log(f"Skipping face with invalid bbox after clipping: {face['face_id']}", self.verbose)
                continue

            embedding_vector = np.asarray(embedding, dtype=np.float32)
            embedding_index = len(embeddings_list)
            embeddings_list.append(embedding_vector)

            face_record = self.build_face_record(
                image_record=image_record,
                face=face,
                clipped_bbox=clipped_bbox,
                embedding_index=embedding_index,
            )
            embedding_records.append(face_record)

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Process all detection records.
        Returns:
        - embedding_records
        - embeddings_array of shape [N, D]
        """
        embedding_records: List[Dict[str, Any]] = []
        embeddings_list: List[np.ndarray] = []

        for idx, image_record in enumerate(detections, start=1):
            log(
                f"[Embedder] Processing {idx}/{len(detections)}: {image_record.get('filename', 'unknown')}",
                self.verbose,
            )
            self.process_image_record(image_record, embeddings_list, embedding_records)

        if embeddings_list:
            embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
        else:
            embeddings_array = np.empty((0, 0), dtype=np.float32)

        return embedding_records, embeddings_array

    def run(
        self,
        detections_json_path: Path = DETECTIONS_JSON,
        embedding_records_json_path: Path = EMBEDDING_RECORDS_JSON,
        embeddings_npy_path: Path = EMBEDDINGS_NPY,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        End-to-end run:
        - load detections.json
        - build embedding_records.json
        - save embeddings.npy
        """
        detections = load_json(detections_json_path)
        log(f"Loaded {len(detections)} image record(s) from {detections_json_path}", self.verbose)

        embedding_records, embeddings_array = self.process_detections(detections)

        save_json(embedding_records, embedding_records_json_path)
        save_npy(embeddings_array, embeddings_npy_path)

        log(f"Saved embedding records to: {embedding_records_json_path}", self.verbose)
        log(f"Saved embeddings array to: {embeddings_npy_path}", self.verbose)
        log(f"Total face records saved: {len(embedding_records)}", self.verbose)

        return embedding_records, embeddings_array