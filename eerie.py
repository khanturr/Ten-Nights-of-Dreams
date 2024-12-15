import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_embeddings(file_path):
    with open("/Users/work/Desktop/Projects/Eerie/dream_embeddings.json", 'r') as f:
        embeddings_dict = json.load(f)
    
    embeddings = []
    labels = []
    for header, emb in embeddings_dict.items():
        if isinstance(emb[0], list):
            flat_emb = [item for sublist in emb for item in sublist]
        else:
            flat_emb = [float(item) for item in emb]
        
        embeddings.append(flat_emb)
        labels.append(header)
    
    return np.array(embeddings), labels

def visualize_dream_embeddings(embeddings, labels):
    n_neighbors = min(10, len(embeddings) - 1)
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=n_neighbors
    )
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(12, 8), facecolor='black', edgecolor='none')
    plt.style.use('dark_background')
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=np.linspace(0, 1, len(embedding_2d)),
        cmap='cool',
        alpha=0.7
    )
    
    for i, label in enumerate(labels):
        plt.annotate(
            label, 
            (embedding_2d[i, 0], embedding_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            color='white',
            fontsize=8
        )
    
    plt.title("Dream Embedding Visualization", color='white')
    plt.xlabel("UMAP Dimension 1", color='white')
    plt.ylabel("UMAP Dimension 2", color='white')
    plt.colorbar(scatter, label='Dream Progression')
    plt.tight_layout()
    plt.savefig('dream_embeddings_visualization.png', facecolor='black', edgecolor='none')
    plt.show()

embeddings, labels = load_embeddings('dream_embeddings.json')
visualize_dream_embeddings(embeddings, labels)

print("Embedding shape:", embeddings.shape)
print("Labels:", labels)