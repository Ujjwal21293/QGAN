import numpy as np
import matplotlib.pyplot as plt
import os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

def generate_2d_gaussian_mixture(n_samples=1000):
    """Generate a 2D mixture of Gaussians centered at four locations"""
    centers = [
        [2, 2],
        [-2, 2],
        [-2, -2],
        [2, -2]
    ]
    cluster_std = 0.5
    samples_per_center = n_samples // len(centers)

    data = []
    for center in centers:
        x = np.random.normal(loc=center[0], scale=cluster_std, size=(samples_per_center,))
        y = np.random.normal(loc=center[1], scale=cluster_std, size=(samples_per_center,))
        cluster = np.vstack((x, y)).T
        data.append(cluster)

    full_data = np.vstack(data)
    return full_data

# Generate and save
real_data = generate_2d_gaussian_mixture(n_samples=2000)
np.save("data/real_distribution.npy", real_data)

# Optional: visualize
plt.figure(figsize=(6,6))
plt.scatter(real_data[:,0], real_data[:,1], alpha=0.5, s=10)
plt.title("2D Gaussian Mixture (Real Data)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.savefig("data/real_distribution_plot.png")
plt.show()
