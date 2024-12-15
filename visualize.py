import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation

def load_embeddings(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_embeddings(embeddings: dict) -> np.ndarray:
    flattened_embeddings = [emb[0] for emb in embeddings.values()]  
    return np.array(flattened_embeddings)

def reduce_dimensions(embeddings: dict, n_components=2) -> np.ndarray:
    # Flatten embeddings
    flattened_embeddings = flatten_embeddings(embeddings)
    print(f"Number of samples: {flattened_embeddings.shape[0]}")
    print(f"Number of features per sample: {flattened_embeddings.shape[1]}")
    
    print(f"Shape of flattened embeddings: {flattened_embeddings.shape}") 
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(flattened_embeddings)
    
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings: np.ndarray, titles: list):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    def update(frame):
        sc.set_offsets(np.column_stack([reduced_embeddings[:, 0] + frame * 0.1, reduced_embeddings[:, 1]]))
        return sc,

    ani = FuncAnimation(fig, update, frames=50, interval=100)
    plt.show()


if __name__ == "__main__":
    embeddings_file = "dream_embeddings.json"
    embeddings = load_embeddings(embeddings_file)
    
    reduced_embeddings = reduce_dimensions(embeddings)
    
    titles = list(embeddings.keys())
    
    visualize_embeddings(reduced_embeddings, titles)