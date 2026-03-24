from config import (
    CLEAN_OUTPUT_BEFORE_RUN,
    CLUSTERS_DIR,
    INPUT_DIR,
    METADATA_DIR,
    OUTPUT_DIR,
    VERBOSE,
)
from clusterer import FaceClusterer
from detector import FaceDetector
from embedder import FaceEmbedder
from evaluator import ClusterEvaluator
from organizer import ClusterOrganizer
from splitter import split_images_by_face_presence
from utils import ensure_dirs, log, reset_dir


def setup_directories() -> None:
    """
    Create required project directories if they do not exist.
    """
    ensure_dirs(
        [
            INPUT_DIR,
            OUTPUT_DIR,
            CLUSTERS_DIR,
            METADATA_DIR,
        ]
    )


def clean_output_directories() -> None:
    """
    Remove old outputs and recreate clean directories.
    Does not touch INPUT_DIR.
    """
    reset_dir(CLUSTERS_DIR)
    reset_dir(METADATA_DIR)


def summarize_split_results(image_records: list[dict]) -> None:
    total_images = len(image_records)
    images_with_faces = sum(1 for record in image_records if record["has_face"])
    images_without_faces = total_images - images_with_faces
    total_detected_faces = sum(record["num_faces"] for record in image_records)

    print("\n===== Split Summary =====")
    print(f"Total images processed : {total_images}")
    print(f"Images with faces      : {images_with_faces}")
    print(f"Images without faces   : {images_without_faces}")
    print(f"Total detected faces   : {total_detected_faces}")
    print("=========================\n")


def summarize_embedding_results(embedding_records, embeddings) -> None:
    print("\n===== Embedding Summary =====")
    print(f"Face records created   : {len(embedding_records)}")
    print(f"Embeddings shape       : {embeddings.shape}")
    print("============================\n")


def summarize_clustering_results(clustering_results: list[dict]) -> None:
    labels = [record["cluster_label"] for record in clustering_results]
    unique_labels = sorted(set(labels))
    non_noise = [x for x in unique_labels if x != -1]
    noise_count = sum(1 for x in labels if x == -1)

    print("\n===== Clustering Summary =====")
    print(f"Clustered faces        : {len(clustering_results)}")
    print(f"Clusters found         : {len(non_noise)}")
    print(f"Noise / unknown faces  : {noise_count}")
    print("==============================\n")


def summarize_organizer_results(summary: dict) -> None:
    print("\n===== Organizer Summary =====")
    print(f"Cluster folders created: {summary.get('num_cluster_folders', 0)}")
    print(f"Face crops created     : {summary.get('saved_face_crops', 0)}")
    print(f"Full images copied     : {summary.get('copied_full_images', 0)}")
    print("=============================\n")


def main() -> None:
    """
    Full pipeline:
    1. setup directories
    2. optionally clean old outputs
    3. detect and split images into faces / no_faces
    4. create embedding records
    5. cluster embeddings
    6. organize clustered outputs
    7. evaluate cluster results
    """
    log("Setting up directories...", VERBOSE)
    setup_directories()

    if CLEAN_OUTPUT_BEFORE_RUN:
        log("Cleaning previous outputs...", VERBOSE)
        clean_output_directories()

    log("Initializing face detector...", VERBOSE)
    detector = FaceDetector()

    log("Running face / no-face split...", VERBOSE)
    image_records = split_images_by_face_presence(detector=detector)
    summarize_split_results(image_records)

    log("Creating embedding records...", VERBOSE)
    embedder = FaceEmbedder()
    embedding_records, embeddings = embedder.run()
    summarize_embedding_results(embedding_records, embeddings)

    if len(embedding_records) == 0 or embeddings.size == 0:
        log("No embeddings available. Stopping before clustering.", VERBOSE)
        return

    log("Clustering face embeddings...", VERBOSE)
    clusterer = FaceClusterer()
    clustering_results, labels = clusterer.run()
    summarize_clustering_results(clustering_results)

    log("Organizing clustered results into folders...", VERBOSE)
    organizer = ClusterOrganizer()
    organizer_summary = organizer.run()
    summarize_organizer_results(organizer_summary)

    log("Evaluating clusters...", VERBOSE)
    evaluator = ClusterEvaluator()
    evaluator_summary = evaluator.run()

    num_example_clusters = len(evaluator_summary.get("cluster_examples", {}))
    log(f"Prepared sample examples for {num_example_clusters} cluster(s).", VERBOSE)

    log("Done.", VERBOSE)


if __name__ == "__main__":
    main()
