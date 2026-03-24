from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

INPUT_DIR = DATA_DIR / "input_images"
OUTPUT_DIR = DATA_DIR / "output"

FACES_DIR = OUTPUT_DIR / "faces"
NO_FACES_DIR = OUTPUT_DIR / "no_faces"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Metadata file paths
IMAGE_INDEX_CSV = METADATA_DIR / "image_index.csv"
DETECTIONS_JSON = METADATA_DIR / "detections.json"
EMBEDDINGS_NPY = METADATA_DIR / "embeddings.npy"
EMBEDDING_RECORDS_JSON = METADATA_DIR / "embedding_records.json"
CLUSTERING_RESULTS_JSON = METADATA_DIR / "clustering_results.json"

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Face detection settings
DETECTION_CONFIDENCE_THRESHOLD = 0.60
MIN_FACE_SIZE = 40  # minimum width and height in pixels

# InsightFace settings
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]
INSIGHTFACE_CTX_ID = 0
INSIGHTFACE_DET_SIZE = (640, 640)

# Clustering defaults
CLUSTER_METHOD = "dbscan"  # options: "dbscan", "kmeans"
DBSCAN_EPS = 0.90
DBSCAN_MIN_SAMPLES = 3
KMEANS_K = 5

# Output behavior
SAVE_FULL_IMAGES_TO_CLUSTERS = True
SAVE_FACE_CROPS_TO_CLUSTERS = True

# Logging / debug
VERBOSE = True

# Cleanup behavior
CLEAN_OUTPUT_BEFORE_RUN = False

# Evaluation
TOP_K_LARGEST_CLUSTERS_TO_PRINT = 10
