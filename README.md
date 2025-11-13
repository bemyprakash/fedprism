# Fed-PRISM: Federated Personalized Relevance-based Intelligent Soft-assignment Models

A complete implementation of the Federated Personalized Relevance-based Intelligent Soft-assignment Models (Fed-PRISM) algorithm for federated learning with a YOLOv12 detection backbone.

## Algorithm Overview

Fed-PRISM combines three ingredients:
1. **Global Model** – standard FedAvg aggregation across all clients
2. **Cluster Models** – K cluster-specific models to capture distributional diversity
3. **Multi-Assignment** – each client is softly assigned to multiple clusters and receives a personalized ensemble model

## Project Structure

```
.
├── IID/                 # Real IID client datasets (client1 ... client4)
├── client.py            # Client-side Fed-PRISM logic (Algorithm 3)
├── main.py              # Entry point and training loop
├── model.py             # YOLO model factory
├── server.py            # Server-side Fed-PRISM logic (Algorithms 1 & 2)
├── utils.py             # Dataset loading and helpers
├── yolo.py              # YOLOv12 single-file implementation
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your client datasets in `IID/clientX/` following the YOLO format (`images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`).
2. Run training:

```bash
python main.py
```

The default configuration loads four clients (`client1`–`client4`) already present in the repository. Modify `main.py` if you want to change hyperparameters or client folders.

## Key Hyperparameters (`main.py`)

- `k`: Number of clusters (default: min(3, N))
- `m`: Number of cluster assignments per client (default: min(2, k))
- `c`: Clustering frequency in rounds (default: 2)
- `T`: Total communication rounds (default: 5 for the demo)
- `R`: Client sampling rate per round (default: 1.0)
- `alpha`: Ensemble mixing weight between global and cluster models (default: 0.4)
- `E`: Local epochs per client per round (default: 1)
- `lr`: Client learning rate (default: 1e-3)
- `batch_size`: Client batch size (default: 4)
- `img_size`: Image resize dimension passed to YOLO (default: 320)

## Algorithm Details

### Algorithm 1 – Server
- Maintains the global YOLO model plus `k` cluster-specific copies
- Periodically reclusters clients using K-means on flattened model weights
- Builds personalized models by ensembling global and cluster weights using `alpha`

### Algorithm 2 – UpdateClusters
- Requests client model states
- Runs K-means to obtain centroids
- Computes cosine similarities and assigns each client to the top-`m` centroids with softmax weights

### Algorithm 3 – ClientUpdate
- Receives a personalized YOLO model from the server
- Trains on the client's local detection dataset using SGD
- Returns the model delta (θ_end − θ_start) to the server
- Evaluates locally with mAP@0.5 for reporting

## Features

- ✅ Full Fed-PRISM workflow (global + clusters + multi-assignment)
- ✅ YOLOv12 detection backbone with training, validation, and inference helpers
- ✅ Real IID client datasets loaded from disk (no synthetic placeholders)
- ✅ Periodic clustering and personalization
- ✅ Weighted FedAvg updates for both global and cluster models
- ✅ Validation reporting via mAP@0.5

## Notes

- Images are resized to `img_size` while targets are rescaled automatically.
- Ensure all clients share the same class count and `data.yaml` metadata.
- Training settings in `main.py` are conservative for quick experimentation; adjust for full training runs.

