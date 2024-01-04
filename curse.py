import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers

num_samples_lst = [50, 500, 5000, 50000]

for idx, num_samples in enumerate(num_samples_lst):
    # TODO: Fill in a valid feature range
    feature_range = np.arange(1,101)
    ratios = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X,_ = make_classification(n_samples=num_samples, n_features=num_features, n_informative=num_features, n_redundant=0, n_repeated=0, n_classes=1, n_clusters_per_class=1, random_state=42)
        
        # TODO: Choose random query point from X
        query_point_idx = np.random.randint(num_samples-1) # X is array of shape (num_samples, num_features), so choose random row
        query_point = X[query_point_idx]

        # TODO: remove query pt from X so it isn't used in distance calculations
        X = np.delete(X, query_point_idx, axis=0)

        # TODO: Calculate distances
        distances = np.sqrt(np.sum((X - query_point)**2, axis=1))
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')\

plt.legend()
plt.tight_layout()
plt.grid(True)
#plt.savefig(f'Curse.png')
plt.show()

#Use sklearn to generate a dataset from a standard Gaussian distribution. Then select a random query point
#and calculate the DMAX/DMIN ratio (the ratio of the distance of furthest point from the query to distance of
#nearest point to the query)