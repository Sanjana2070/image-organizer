from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config import (
    CLUSTERING_RESULTS_JSON,
    EMBEDDING_RECORDS_JSON,
    TOP_K_LARGEST_CLUSTERS_TO_PRINT,
    VERBOSE,
)
from io_utils import load_json
from utils import log


class ClusterEvaluator:
    """
    Simple evaluation and inspection helper for clustering results.

    Reports:
    - total clustered faces
    - number of identity clusters
    - number of unknown / noise faces
    - cluster size distribution
    - largest clusters
    """

    def __init__(
        self,
        top_k_largest_clusters: int = TOP_K_LARGEST_CLUSTERS_TO_PRINT,
        verbose: bool = VERBOSE,
    ) -> None:
        self.top_k_largest_clusters = top_k_largest_clusters
        self.verbose = verbose

    @staticmethod
    def _cluster_name(cluster_label: int) -> str:
        if cluster_label == -1:
            return "unknown"
        return f"person_{cluster_label:03d}"

    def build_summary(
        self,
        clustering_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build evaluation summary from clustering results.
        """
        labels = [int(record["cluster_label"]) for record in clustering_results]
        counts = Counter(labels)

        total_faces = len(labels)
        num_unknown_faces = counts.get(-1, 0)
        known_cluster_labels = sorted([label for label in counts.keys() if label != -1])
        num_clusters_excluding_unknown = len(known_cluster_labels)

        largest_clusters = []
        for label, count in counts.most_common():
            largest_clusters.append(
                {
                    "cluster_label": label,
                    "cluster_name": self._cluster_name(label),
                    "num_faces": count,
                }
            )

        cluster_size_distribution = sorted(
            [{"cluster_label": label, "num_faces": count} for label, count in counts.items()],
            key=lambda x: (-x["num_faces"], x["cluster_label"]),
        )

        return {
            "total_faces_clustered": total_faces,
            "num_clusters_excluding_unknown": num_clusters_excluding_unknown,
            "num_unknown_faces": num_unknown_faces,
            "largest_clusters": largest_clusters[: self.top_k_largest_clusters],
            "cluster_size_distribution": cluster_size_distribution,
        }

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Pretty-print evaluation summary.
        """
        print("\n===== Evaluation Summary =====")
        print(f"Total clustered faces     : {summary['total_faces_clustered']}")
        print(f"Identity clusters found   : {summary['num_clusters_excluding_unknown']}")
        print(f"Unknown / noise faces     : {summary['num_unknown_faces']}")
        print("==============================")

        print("\nLargest clusters:")
        if not summary["largest_clusters"]:
            print("  None")
        else:
            for item in summary["largest_clusters"]:
                print(f"  {item['cluster_name']}: {item['num_faces']} face(s)")

        print()

    def find_cluster_examples(
        self,
        clustering_results: List[Dict[str, Any]],
        embedding_records: List[Dict[str, Any]],
        max_examples_per_cluster: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return a few sample records per cluster for manual inspection.
        """
        by_face_id = {record["face_id"]: record for record in embedding_records}
        grouped = defaultdict(list)

        for result in clustering_results:
            face_id = result["face_id"]
            cluster_label = int(result["cluster_label"])
            cluster_name = self._cluster_name(cluster_label)

            full_record = by_face_id.get(face_id)
            if full_record is None:
                continue

            if len(grouped[cluster_name]) < max_examples_per_cluster:
                grouped[cluster_name].append(
                    {
                        "face_id": face_id,
                        "filename": full_record.get("filename"),
                        "original_image_path": full_record.get("original_image_path"),
                        "bbox": full_record.get("bbox"),
                    }
                )

        return dict(grouped)

    def run(
        self,
        clustering_results_json_path: Path = CLUSTERING_RESULTS_JSON,
        embedding_records_json_path: Path = EMBEDDING_RECORDS_JSON,
    ) -> Dict[str, Any]:
        """
        End-to-end evaluation helper:
        - load clustering results
        - load embedding records
        - print summary
        - return summary plus examples
        """
        clustering_results = load_json(clustering_results_json_path)
        embedding_records = load_json(embedding_records_json_path)

        log(f"Loaded clustering results: {len(clustering_results)}", self.verbose)
        log(f"Loaded embedding records: {len(embedding_records)}", self.verbose)

        summary = self.build_summary(clustering_results)
        self.print_summary(summary)

        examples = self.find_cluster_examples(clustering_results, embedding_records)
        summary["cluster_examples"] = examples

        return summary
