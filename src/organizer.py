from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config import (
    CLUSTERS_DIR,
    CLUSTERING_RESULTS_JSON,
    EMBEDDING_RECORDS_JSON,
    SAVE_FACE_CROPS_TO_CLUSTERS,
    SAVE_FULL_IMAGES_TO_CLUSTERS,
    VERBOSE,
)
from io_utils import copy_file, load_image, load_json, save_image
from utils import ensure_dir, log


class ClusterOrganizer:
    """
    Organize clustered outputs into person folders.

    For each cluster:
    - optionally create face crops from original images
    - optionally copy full images

    Noise / unknown cluster (-1) goes to 'unknown'.
    """

    def __init__(
        self,
        clusters_dir: Path = CLUSTERS_DIR,
        save_face_crops: bool = SAVE_FACE_CROPS_TO_CLUSTERS,
        save_full_images: bool = SAVE_FULL_IMAGES_TO_CLUSTERS,
        verbose: bool = VERBOSE,
    ) -> None:
        self.clusters_dir = clusters_dir
        self.save_face_crops = save_face_crops
        self.save_full_images = save_full_images
        self.verbose = verbose
        ensure_dir(self.clusters_dir)

    @staticmethod
    def cluster_name(cluster_label: int) -> str:
        """
        Convert cluster label to folder name.
        """
        if cluster_label == -1:
            return "unknown"
        return f"person_{cluster_label:03d}"

    def _cluster_dirs(self, cluster_label: int) -> Dict[str, Path]:
        """
        Return output dirs for a given cluster.
        """
        base = self.clusters_dir / self.cluster_name(cluster_label)
        dirs = {
            "base": base,
            "face_crops": base / "face_crops",
            "full_images": base / "full_images",
        }
        for path in dirs.values():
            ensure_dir(path)
        return dirs

    def _merge_records(
        self,
        clustering_results: List[Dict[str, Any]],
        embedding_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Join clustering results with full embedding metadata by face_id.
        """
        by_face_id = {record["face_id"]: record for record in embedding_records}

        merged: List[Dict[str, Any]] = []
        for result in clustering_results:
            face_id = result["face_id"]
            if face_id not in by_face_id:
                continue

            merged_record = {**by_face_id[face_id], **result}
            merged.append(merged_record)

        return merged

    @staticmethod
    def _clip_bbox_to_image(bbox: List[int], image_shape: tuple[int, int, int]) -> List[int]:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        return [x1, y1, x2, y2]

    @staticmethod
    def _build_crop_filename(record: Dict[str, Any], source_path: Path) -> str:
        suffix = source_path.suffix or ".jpg"
        return f"{record['face_id']}{suffix}"

    def _save_face_crop(self, record: Dict[str, Any], dst_dir: Path) -> bool:
        original_image_path = record.get("original_image_path")
        bbox = record.get("bbox")
        if not original_image_path or bbox is None:
            return False

        image_path = Path(original_image_path)
        if not image_path.exists():
            return False

        image = load_image(image_path)
        if image is None:
            return False

        x1, y1, x2, y2 = self._clip_bbox_to_image(bbox, image.shape)
        if x2 <= x1 or y2 <= y1:
            return False

        crop = image[y1:y2, x1:x2]
        crop_path = dst_dir / self._build_crop_filename(record, image_path)
        save_image(crop_path, crop)
        return True

    def organize(
        self,
        clustering_results: List[Dict[str, Any]],
        embedding_records: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Copy clustered outputs into cluster folders.
        """
        merged_records = self._merge_records(clustering_results, embedding_records)

        saved_face_crops = 0
        copied_full_images = 0

        # Avoid copying the same full image multiple times into the same cluster folder
        seen_full_image_targets = set()

        grouped = defaultdict(list)
        for record in merged_records:
            grouped[int(record["cluster_label"])].append(record)

        for cluster_label, records in grouped.items():
            dirs = self._cluster_dirs(cluster_label)
            cluster_folder_name = self.cluster_name(cluster_label)

            log(
                f"Organizing cluster '{cluster_folder_name}' with {len(records)} face record(s)",
                self.verbose,
            )

            for record in records:
                if self.save_face_crops:
                    if self._save_face_crop(record, dirs["face_crops"]):
                        saved_face_crops += 1

                if self.save_full_images:
                    full_image_path_str = record.get("original_image_path")
                    if full_image_path_str:
                        full_image_path = Path(full_image_path_str)
                        target_key = (cluster_label, full_image_path.name)

                        if full_image_path.exists() and target_key not in seen_full_image_targets:
                            copy_file(full_image_path, dirs["full_images"], unique_name=False)
                            copied_full_images += 1
                            seen_full_image_targets.add(target_key)

        summary = {
            "num_cluster_folders": len(grouped),
            "saved_face_crops": saved_face_crops,
            "copied_full_images": copied_full_images,
        }

        log(
            f"Organizer summary: cluster_folders={summary['num_cluster_folders']}, "
            f"face_crops={summary['saved_face_crops']}, "
            f"full_images={summary['copied_full_images']}",
            self.verbose,
        )

        return summary

    def run(
        self,
        clustering_results_json_path: Path = CLUSTERING_RESULTS_JSON,
        embedding_records_json_path: Path = EMBEDDING_RECORDS_JSON,
    ) -> Dict[str, int]:
        """
        End-to-end organizer run.
        """
        clustering_results = load_json(clustering_results_json_path)
        embedding_records = load_json(embedding_records_json_path)

        log(f"Loaded clustering results: {len(clustering_results)}", self.verbose)
        log(f"Loaded embedding records: {len(embedding_records)}", self.verbose)

        return self.organize(clustering_results, embedding_records)
