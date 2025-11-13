"""
Fed-PRISM: Federated Personalized Relevance-based Intelligent Soft-assignment Models
Main entry point for running federated learning experiments
"""

from pathlib import Path

import numpy as np
import torch

from client import FedPRISMClient
from model import create_yolo_model
from server import FedPRISMServer
from utils import load_client_yolo_dataset


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Algorithm hyperparameters
    IID_ROOT = Path("IID")
    client_dirs = sorted([p for p in IID_ROOT.iterdir() if p.is_dir() and p.name.startswith("client")])
    N = len(client_dirs)
    if N == 0:
        raise RuntimeError("No client folders found under IID/. Expected client1, client2, ...")

    k = min(3, N)  # number of clusters
    m = min(2, k)  # assignments per client
    c = 2  # clustering frequency (rounds)
    T = 5  # total communication rounds (keep small for demo)
    R = 1.0  # sample all clients each round
    alpha = 0.4  # ensemble hyperparameter

    # Training hyperparameters
    E = 1  # local epochs per round
    lr = 1e-3  # learning rate
    batch_size = 4
    img_size = 320

    print("=" * 60)
    print("Fed-PRISM: Federated Personalized Relevance-based Intelligent Soft-assignment Models")
    print("Device:", device)
    print("=" * 60)
    print("Configuration:")
    print(f"  Total clients: {N}")
    print(f"  Clusters: {k}")
    print(f"  Assignments per client: {m}")
    print(f"  Clustering frequency: every {c} rounds")
    print(f"  Total rounds: {T}")
    print(f"  Sampling rate: {R}")
    print(f"  Ensemble alpha: {alpha}")
    print("=" * 60)

    # Load datasets for each client
    client_datasets = []
    class_names = None
    num_classes = None

    print("\nLoading client datasets from IID/...")
    for client_dir in client_dirs:
        dataset = load_client_yolo_dataset(client_dir, img_size=img_size)
        client_datasets.append(dataset)
        if num_classes is None:
            num_classes = dataset.get("num_classes", 0)
            class_names = dataset.get("class_names", [])

        print(
            f"  {client_dir.name}: train_images={len(dataset['train'][0])}, val_images={len(dataset['val'][0])}"
        )

    if num_classes is None or num_classes == 0:
        raise RuntimeError("Could not determine number of classes from data.yaml files.")

    print(f"Detected {num_classes} classes: {class_names}")

    # Initialize global model
    global_model = create_yolo_model(
        num_classes=num_classes, img_size=img_size, device=device, width_mult=0.25, depth_mult=0.33
    )

    # Initialize server
    server = FedPRISMServer(
        global_model=global_model,
        num_clusters=k,
        num_assignments=m,
        clustering_freq=c,
        sampling_rate=R,
        ensemble_alpha=alpha,
    )

    # Initialize clients
    clients = []
    for idx, dataset in enumerate(client_datasets):
        client_model = create_yolo_model(
            num_classes=num_classes, img_size=img_size, device=device, width_mult=0.25, depth_mult=0.33
        )
        client = FedPRISMClient(
            client_id=idx,
            model=client_model,
            dataset=dataset,
            local_epochs=E,
            learning_rate=lr,
            batch_size=batch_size,
            device=device,
        )
        clients.append(client)

    print("\nStarting federated training...")
    for round_num in range(T):
        print(f"\n--- Round {round_num + 1}/{T} ---")

        # Phase 2: Dynamic Clustering (Periodic)
        if round_num % c == 0:
            print("Performing dynamic clustering...")
            client_states = [client.get_model_state() for client in clients]
            server.update_clusters(client_states)

        # Phase 3: Aggregation & Personalization
        sampled_clients = np.arange(N)
        print(f"Sampling {len(sampled_clients)} clients: {sampled_clients}")

        updates_global = []
        updates_clusters = {j: [] for j in range(k)}

        for client_id in sampled_clients:
            client = clients[client_id]

            personalized_model = server.get_personalized_model(client_id)
            client.set_model(personalized_model)

            delta = client.train()
            updates_global.append(delta)

            client_map = server.get_client_map(client_id)
            for cluster_id, weight in client_map.items():
                updates_clusters[cluster_id].append((weight, delta))

        server.aggregate_updates(updates_global, updates_clusters)

        # Periodic evaluation on validation data
        if (round_num + 1) % c == 0:
            total_map = 0.0
            total_images = 0
            for client_id, client in enumerate(clients):
                personalized_model = server.get_personalized_model(client_id)
                client.set_model(personalized_model)
                map50, images = client.evaluate()
                total_map += map50 * images
                total_images += images
            avg_map = total_map / total_images if total_images > 0 else 0.0
            print(f"Validation mAP@0.5 across sampled clients: {avg_map:.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

