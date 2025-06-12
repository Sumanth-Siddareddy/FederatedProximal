import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# NON-IID Data Distribution: Train and test class-wise counts
model_train = {
    "Model 1": [1021, 100, 100, 100],
    "Model 2": [100, 1039, 100, 100],
    "Model 3": [100, 100, 1295, 100],
    "Model 4": [100, 100, 100, 1157],
}

model_test = {
    "Model 1": [210, 32, 50, 50],
    "Model 2": [30, 210, 50, 50],
    "Model 3": [30, 32, 255, 50],
    "Model 4": [30, 32, 50, 150],
}

# Combine and normalize distributions
def normalize(dist):
    return np.array(dist) / np.sum(dist)

combined_probs = {}
for key in model_train:
    combined = np.array(model_train[key]) + np.array(model_test[key])
    combined_probs[key] = normalize(combined)

# Compute JSD matrix
model_names = list(combined_probs.keys())
n = len(model_names)
jsd_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        jsd = jensenshannon(combined_probs[model_names[i]], combined_probs[model_names[j]]) ** 2
        jsd_matrix[i, j] = jsd

# Plot with high-resolution and enhanced fonts
with PdfPages("combined_jsd_matrix_blue.pdf") as pdf:
    plt.figure(figsize=(9, 7), dpi=1000)
    sns.set_style("white")
    sns.set_context("notebook")

    ax = sns.heatmap(jsd_matrix,
                     xticklabels=model_names,
                     yticklabels=model_names,
                     annot=True,
                     fmt=".4f",
                     cmap="Blues",
                     linewidths=0.5,
                     linecolor='gray',
                     square=True,
                     cbar=False,
                     annot_kws={"size": 13, "weight": "bold"})

    # Set font styles
    ax.set_title("Combined Jensen-Shannon Divergence Matrix (Train + Test)",
                 fontsize=18, weight='bold', pad=20)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, weight='bold', rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, weight='bold', rotation=0)

    plt.tight_layout()
    pdf.savefig(dpi=1000)
    plt.close()

print("High-quality, bold-labeled PDF saved as 'combined_jsd_matrix_blue.pdf'")
