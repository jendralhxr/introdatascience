import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Generate Synthetic Height & Weight Data ---
n_samples = 50

# S, M, L height and weight distributions
height_S, weight_S = np.random.normal(150, 4, n_samples), np.random.normal(52, 4, n_samples)
height_M, weight_M = np.random.normal(160, 4, n_samples), np.random.normal(60, 4, n_samples)
height_L, weight_L = np.random.normal(170, 4, n_samples), np.random.normal(65, 4, n_samples)

heights = np.concatenate([height_S, height_M, height_L])
weights = np.concatenate([weight_S, weight_M, weight_L])
X = np.column_stack((heights, weights))

# --- 2. Custom K-Means Implementation to Track Iterations ---
def run_kmeans_with_history(X, k=3, max_iters=10):
    initial_idx = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_idx]
    
    history = []
    
    for iteration in range(max_iters):
        # Calculate Euclidean distance using np.linalg.norm
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        history.append({
            'iteration': iteration + 1,
            'centroids': centroids.copy(),
            'labels': labels.copy()
        })
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return history

history = run_kmeans_with_history(X, k=3)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# --- Plot 1: Distinct Figure - Before Clustering ---
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, edgecolors='none')
plt.title("Before Clustering (Unlabeled Data)", fontsize=14, fontweight='bold')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Plot 2+: Distinct Figures - Every Iteration ---
for state in history:
    labels = state['labels']
    centroids = state['centroids']
    
    plt.figure(figsize=(8, 6))
    
    # Sort centroids by height to map S, M, L consistently
    sorted_order = np.argsort(centroids[:, 0])
    size_names = ['S', 'M', 'L']
    label_map = {sorted_order[i]: size_names[i] for i in range(3)}
    
    # Plot points by cluster
    for c_idx in range(3):
        mask = (labels == c_idx)
        plt.scatter(
            X[mask, 0], X[mask, 1], 
            c=colors[c_idx], alpha=0.6, 
            label=f'Size {label_map[c_idx]}'
        )
        
    # Plot centroids
    for c_idx in range(3):
        cx, cy = centroids[c_idx]
        plt.scatter(cx, cy, c='black', s=200, marker='X', zorder=5)
        plt.annotate(
            f"Center {label_map[c_idx]}\n({cx:.1f}cm, {cy:.1f}kg)", 
            (cx, cy), textcoords="offset points", xytext=(0, 10), 
            ha='center', fontweight='bold', fontsize=9
        )
        
    plt.title(f"K-Means Iteration {state['iteration']}", fontsize=14, fontweight='bold')
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()