from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

from config import (
    CLUSTER_METHOD,
    CLUSTERING_RESULTS_JSON,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    EMBEDDINGS_NPY,
    EMBEDDING_RECORDS_JSON,
    KMEANS_K,
    VERBOSE,
)
from io_utils import load_json, load_npy, save_json
from utils import log


class FaceClusterer:
    """
    Cluster face embeddings using DBSCAN or KMeans.
    """

    def __init__(
        self,
        method: str = CLUSTER_METHOD,
        dbscan_eps: float = DBSCAN_EPS,
        dbscan_min_samples: int = DBSCAN_MIN_SAMPLES,
        kmeans_k: int = KMEANS_K,
        normalize_embeddings: bool = True,
        verbose: bool = VERBOSE,
    ) -> None:
        self.method = method.lower()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kmeans_k = kmeans_k
        self.normalize_embeddings = normalize_embeddings
        self.verbose = verbose

        if self.method not in {"dbscan", "kmeans"}:
            raise ValueError("method must be 'dbscan' or 'kmeans'")

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Optionally L2-normalize embeddings before clustering.
        """
        if embeddings.size == 0:
            return embeddings

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings array must be 2D of shape [N, D], got shape {embeddings.shape}."
            )

        if self.normalize_embeddings:
            return normalize(embeddings, norm="l2")

        return embeddings

    def _run_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run DBSCAN clustering.
        Noise points are labeled -1.
        """
        model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="euclidean",
        )
        return model.fit_predict(embeddings)

    def _run_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run KMeans clustering.
        """
        if len(embeddings) < self.kmeans_k:
            raise ValueError(
                f"KMeans requested k={self.kmeans_k}, but only {len(embeddings)} embeddings are available."
            )

        model = KMeans(
            n_clusters=self.kmeans_k,
            random_state=42,
            n_init=10,
        )
        return model.fit_predict(embeddings)

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using the configured method.
        """
        if embeddings.size == 0:
            return np.array([], dtype=int)

        prepared = self._prepare_embeddings(embeddings)

        if self.method == "dbscan":
            labels = self._run_dbscan(prepared)
        else:
            labels = self._run_kmeans(prepared)

        return labels.astype(int)

    def build_results(
        self,
        embedding_records: List[Dict[str, Any]],
        labels: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Build serializable clustering results aligned with embedding_records.
        """
        if len(embedding_records) != len(labels):
            raise ValueError(
                f"Mismatch: {len(embedding_records)} embedding records vs {len(labels)} cluster labels."
            )

        results: List[Dict[str, Any]] = []

        for record, label in zip(embedding_records, labels):
            result = {
                "face_id": record["face_id"],
                "image_id": record["image_id"],
                "filename": record["filename"],
                "original_image_path": record["original_image_path"],
                "bbox": record["bbox"],
                "score": record.get("score"),
                "width": record.get("width"),
                "height": record.get("height"),
                "kps": record.get("kps"),
                "embedding_index": record["embedding_index"],
                "cluster_label": int(label),
            }
            results.append(result)

        return results

    def summarize_labels(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Return summary statistics for cluster labels.
        """
        if labels.size == 0:
            return {
                "total_faces": 0,
                "num_clusters_excluding_noise": 0,
                "num_noise_faces": 0,
            }

        unique_labels = set(int(x) for x in labels.tolist())
        num_clusters = len([x for x in unique_labels if x != -1])
        num_noise = int(np.sum(labels == -1))

        return {
            "total_faces": int(len(labels)),
            "num_clusters_excluding_noise": num_clusters,
            "num_noise_faces": num_noise,
        }

    def run(
        self,
        embeddings_npy_path: Path = EMBEDDINGS_NPY,
        embedding_records_json_path: Path = EMBEDDING_RECORDS_JSON,
        clustering_results_json_path: Path = CLUSTERING_RESULTS_JSON,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        End-to-end clustering:
        - load embeddings
        - load embedding records
        - validate count match
        - cluster embeddings
        - save clustering results
        """
        embeddings = load_npy(embeddings_npy_path)
        embedding_records = load_json(embedding_records_json_path)

        log(f"Loaded embeddings: shape={embeddings.shape}", self.verbose)
        log(f"Loaded embedding records: {len(embedding_records)}", self.verbose)

        if embeddings.size == 0:
            log("No embeddings found. Saving empty clustering results.", self.verbose)
            labels = np.array([], dtype=int)
            results: List[Dict[str, Any]] = []
            save_json(results, clustering_results_json_path)
            return results, labels

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings array must be 2D of shape [N, D], got shape {embeddings.shape}."
            )

        if len(embedding_records) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between embedding records ({len(embedding_records)}) "
                f"and embeddings rows ({embeddings.shape[0]})."
            )

        labels = self.cluster_embeddings(embeddings)
        results = self.build_results(embedding_records, labels)

        save_json(results, clustering_results_json_path)

        summary = self.summarize_labels(labels)
        log(f"Saved clustering results to: {clustering_results_json_path}", self.verbose)
        log(
            "Clustering summary: "
            f"faces={summary['total_faces']}, "
            f"clusters={summary['num_clusters_excluding_noise']}, "
            f"noise={summary['num_noise_faces']}",
            self.verbose,
        )

        return results, labels