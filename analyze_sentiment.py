import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import noise  

class FluidHeightmapVisualizer:
    def __init__(self, embedding_file):
        with open(embedding_file, 'r') as f:
            self.embeddings = json.load(f)
    
    def generate_heightmap_animation(self):
        embedding_list = [np.array(embedding).reshape(-1, 2) for embedding in self.embeddings.values()]
        all_embeddings = np.vstack(embedding_list)
        
        heights = np.linalg.norm(all_embeddings, axis=1)
        
        x = all_embeddings[:, 0]
        y = all_embeddings[:, 1]
        z = heights
        
        z_min, z_max = z.min(), z.max()
        z_normalized = (z - z_min) / (z_max - z_min)
        
        grid_size_x = 60
        grid_size_y = 64
        
        heatmap_data = z_normalized[:grid_size_x * grid_size_y].reshape(grid_size_x, grid_size_y)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        heatmap = ax.imshow(heatmap_data, cmap='gray', origin='lower', aspect='auto')
        
        def update(frame):
            dynamic_heights = z_normalized + 0.1 * noise.pnoise2(x + frame / 20.0, y + frame / 20.0, octaves=6)
            dynamic_heights = (dynamic_heights - dynamic_heights.min()) / (dynamic_heights.max() - dynamic_heights.min())
            dynamic_heatmap_data = dynamic_heights[:grid_size_x * grid_size_y].reshape(grid_size_x, grid_size_y)
            
            heatmap.set_data(dynamic_heatmap_data)
            return heatmap,

        anim = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
        writer = animation.PillowWriter(fps=20)
        anim.save('fluid_heightmap_animation.gif', writer=writer)
        
        plt.show()

visualizer = FluidHeightmapVisualizer('/Users/work/Desktop/Projects/Eerie/dream_embeddings.json')
visualizer.generate_heightmap_animation()
