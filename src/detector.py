from typing import Any, Dict, List

import numpy as np
from insightface.app import FaceAnalysis

from config import (
    DETECTION_CONFIDENCE_THRESHOLD,
    INSIGHTFACE_CTX_ID,
    INSIGHTFACE_DET_SIZE,
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_PROVIDERS,
    MIN_FACE_SIZE,
    VERBOSE,
)
from utils import bbox_width_height, log


class FaceDetector:
    """
    Wrapper around InsightFace FaceAnalysis for:
    - face detection
    - filtering by score and size
    - returning standardized face records
    """

    def __init__(
        self,
        model_name: str = INSIGHTFACE_MODEL_NAME,
        providers: list[str] = INSIGHTFACE_PROVIDERS,
        ctx_id: int = INSIGHTFACE_CTX_ID,
        det_size: tuple[int, int] = INSIGHTFACE_DET_SIZE,
        score_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
        min_face_size: int = MIN_FACE_SIZE,
        verbose: bool = VERBOSE,
    ) -> None:
        self.model_name = model_name
        self.providers = providers
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.score_threshold = score_threshold
        self.min_face_size = min_face_size
        self.verbose = verbose

        log(
            f"Initializing InsightFace model='{self.model_name}' with providers={self.providers}",
            self.verbose,
        )

        self.app = FaceAnalysis(name=self.model_name, providers=self.providers)
        self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

    def detect_faces(self, image: np.ndarray) -> List[Any]:
        """
        Run raw face detection using InsightFace.
        Input image must be a valid OpenCV BGR image.
        Returns raw face objects from InsightFace.
        """
        if image is None:
            raise ValueError("Input image is None.")

        faces = self.app.get(image)
        return faces

    def _is_valid_face(self, face: Any) -> bool:
        """
        Filter detections by score and minimum face size.
        """
        bbox = face.bbox.astype(int).tolist()
        score = float(getattr(face, "det_score", 0.0))
        width, height = bbox_width_height(bbox)

        if score < self.score_threshold:
            return False

        if width < self.min_face_size or height < self.min_face_size:
            return False

        return True

    def standardize_face(self, face: Any, face_index: int) -> Dict[str, Any]:
        """
        Convert InsightFace face object into a serializable dictionary.
        Includes embedding if available.
        """
        bbox = face.bbox.astype(int).tolist()
        score = float(getattr(face, "det_score", 0.0))
        width, height = bbox_width_height(bbox)

        embedding = getattr(face, "embedding", None)
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32).tolist()

        kps = getattr(face, "kps", None)
        if kps is not None:
            kps = np.asarray(kps, dtype=np.float32).tolist()

        standardized = {
            "face_index": face_index,
            "bbox": bbox,
            "score": score,
            "width": width,
            "height": height,
            "embedding": embedding,
            "kps": kps,
        }
        return standardized

    def detect_and_filter(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces, then filter and standardize them.
        Returns a list of serializable face dictionaries.
        """
        raw_faces = self.detect_faces(image)
        valid_faces = []

        for idx, face in enumerate(raw_faces):
            if self._is_valid_face(face):
                valid_faces.append(self.standardize_face(face, idx))

        return valid_faces

    def process_image(self, image: np.ndarray, image_path: str | None = None) -> Dict[str, Any]:
        """
        Full image-level processing:
        - detect
        - filter
        - return standardized metadata
        """
        faces = self.detect_and_filter(image)

        result = {
            "image_path": image_path,
            "has_face": len(faces) > 0,
            "num_faces": len(faces),
            "faces": faces,
        }
        return result