import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix data
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
confusion_matrix = np.array([
    [235, 57, 1, 7],
    [16, 245, 28, 17],
    [6, 7, 388, 4],
    [7, 16, 2, 275]
])

# Create the heatmap
plt.figure(figsize=(30, 26))  # Increased vertical size
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 40, "weight": "bold"})
plt.xlabel("Predicted Labels", fontsize=40, weight="bold")
plt.ylabel("True Labels", fontsize=40, weight="bold")
plt.xticks(fontsize=40, weight="bold")
plt.yticks(fontsize=40, weight="bold")

# Adjust font size of the colorbar scale
colorbar = plt.gca().collections[0].colorbar
colorbar.ax.tick_params(labelsize=18, width=1)

# Remove top and bottom margins
plt.subplots_adjust(top=0.95, bottom=0.05)

# Save the plot as a PDF
output_path = "confusion_matrix_mu=1.pdf"
plt.savefig(output_path, format="pdf", bbox_inches='tight')
plt.show()

print(f"Confusion matrix saved as {output_path}")
