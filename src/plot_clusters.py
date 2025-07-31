import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def plot(embeddings, labels, path, dataset):
    # Convert embeddings and labels to NumPy
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Plot and save
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_np, cmap='tab10', s=10)
    plt.colorbar(scatter, label="True Labels")
    plt.title(f"t-SNE visualization of node embeddings on {dataset} dataset after {path}")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(f"/content/Graph_clustering_with_modularity_sampling_and_maximization/plots/clusters_tsne/embedding_visualization_for_{dataset}_{path}.png", dpi=300)
    plt.show()
