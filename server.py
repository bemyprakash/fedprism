"""
Fed-PRISM Server Implementation
Implements Algorithm 1 (Server-Side Logic) and Algorithm 2 (UpdateClusters)
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import copy


class FedPRISMServer:
    """Fed-PRISM server implementing the server-side logic from Algorithm 1"""

    def __init__(
        self,
        global_model,
        num_clusters,
        num_assignments,
        clustering_freq,
        sampling_rate,
        ensemble_alpha,
    ):
        self.k = num_clusters
        self.m = num_assignments
        self.c = clustering_freq
        self.R = sampling_rate
        self.alpha = ensemble_alpha

        self.global_model = copy.deepcopy(global_model)
        self.cluster_models = {j: copy.deepcopy(global_model) for j in range(self.k)}
        self.client_map = {}

    def _model_state(self, model):
        return model.model.state_dict()

    def _load_state(self, model, state_dict):
        model.model.load_state_dict(state_dict)

    def update_clusters(self, client_states):
        N = len(client_states)
        if N == 0:
            return

        Theta = []
        for state_dict in client_states:
            params = []
            for key in sorted(state_dict.keys()):
                params.append(state_dict[key].flatten().cpu().numpy())
            theta_i = np.concatenate(params)
            Theta.append(theta_i)

        Theta = np.array(Theta)

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(Theta)
        centroids = kmeans.cluster_centers_

        new_client_map = {}

        for i, theta_i in enumerate(Theta):
            similarities = []
            for j, centroid in enumerate(centroids):
                sim = cosine_similarity([theta_i], [centroid])[0][0]
                similarities.append((sim, j))

            similarities.sort(reverse=True, key=lambda x: x[0])
            top_m = similarities[: self.m]

            scores = [np.exp(s) for s, _ in top_m]
            total_score = sum(scores)

            W_i = {}
            for (s, cluster_id), score in zip(top_m, scores):
                weight = score / total_score if total_score > 0 else 0.0
                W_i[cluster_id] = weight

            new_client_map[i] = W_i

        self.client_map = new_client_map

        print(f"  Clustering complete. Updated assignments for {N} clients.")
        cluster_counts = {}
        for weights in new_client_map.values():
            for cluster_id in weights.keys():
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        print(f"  Cluster distribution: {cluster_counts}")

    def get_personalized_model(self, client_id):
        W_i = self.client_map.get(client_id, {})

        if not W_i:
            return copy.deepcopy(self.global_model)

        personalized = copy.deepcopy(self.global_model)
        global_state = self._model_state(self.global_model)
        personalized_state = self._model_state(personalized)

        for key in personalized_state.keys():
            combined = self.alpha * global_state[key].clone()
            for cluster_id, weight in W_i.items():
                cluster_state = self._model_state(self.cluster_models[cluster_id])
                combined += (1 - self.alpha) * weight * cluster_state[key]
            personalized_state[key] = combined

        self._load_state(personalized, personalized_state)
        return personalized

    def get_client_map(self, client_id):
        return self.client_map.get(client_id, {})

    def aggregate_updates(self, updates_global, updates_clusters):
        if updates_global:
            global_state = self._model_state(self.global_model)
            num_updates = max(len(updates_global), 1)

            for key, param in global_state.items():
                if not torch.is_floating_point(param):
                    continue

                avg_delta = torch.zeros_like(param)
                for delta in updates_global:
                    if not delta or key not in delta:
                        continue
                    delta_tensor = delta[key].to(param.device, dtype=param.dtype)
                    avg_delta += delta_tensor / num_updates
                global_state[key] = param + avg_delta

            self._load_state(self.global_model, global_state)

        for cluster_id in range(self.k):
            cluster_updates = updates_clusters.get(cluster_id, [])
            if not cluster_updates:
                continue

            cluster_state = self._model_state(self.cluster_models[cluster_id])
            total_weight = sum(w for w, delta in cluster_updates if delta)

            if total_weight > 0:
                for key, param in cluster_state.items():
                    if not torch.is_floating_point(param):
                        continue

                    weighted_delta = torch.zeros_like(param)
                    for weight, delta in cluster_updates:
                        if not delta or key not in delta:
                            continue
                        delta_tensor = delta[key].to(param.device, dtype=param.dtype)
                        weighted_delta += (weight / total_weight) * delta_tensor
                    cluster_state[key] = param + weighted_delta

                self._load_state(self.cluster_models[cluster_id], cluster_state)

