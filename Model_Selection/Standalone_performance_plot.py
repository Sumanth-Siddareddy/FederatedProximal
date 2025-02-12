# ---------------------- Standalone performance ---------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)

# Model names
models = ['ResNet50+CBAM', 'ResNet18+CBAM', 'VGG19+CBAM']

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss']

# Average evaluation metrics for each model
data = [
    [np.mean([85.09/100.0, 80.00/100.0, 70.84/100.0, 57.25/100.0]),  # Accuracy
     np.mean([0.8224, 0.8077, 0.9011, 0.3278]),  # Precision
     np.mean([0.8509, 0.8000, 0.7084, 0.5725]),  # Recall
     np.mean([0.8135, 0.7831, 0.7283, 0.4169]),  # F1 Score
     np.mean([0.8917, 0.9584, 1.0370, 1.3799]),],# Loss

    [np.mean([70.47/100.0, 61.76/100.0, 69.48/100.0, 76.72/100.0]),
     np.mean([0.7066, 0.3815, 0.4828, 0.8597]),
     np.mean([0.7047, 0.6176, 0.6948, 0.7672]),
     np.mean([0.6316, 0.4717, 0.5697, 0.7624]),
     np.mean([1.0412, 1.2082, 1.0639, 0.9549])],

    [np.mean([88.60/100.0, 85.00/100.0, 82.02/100.0, 85.88/100.0]),
     np.mean([0.8858, 0.8489, 0.7992, 0.8655]),
     np.mean([0.8860, 0.8500, 0.8202, 0.8588]),
     np.mean([0.8623, 0.8465, 0.7852, 0.8422]),
     np.mean([0.8991, 0.9239, 0.9489, 0.8756])]
]

# Convert data to numpy array and transpose it for correct indexing
data = np.array(data).T  # Shape becomes (5,3) for (metrics, models)

# Plot settings
bar_width = 0.2
x = np.arange(len(metrics))
colors = [rgb(135, 206, 250),
          rgb(144, 238, 144),
          rgb(240, 128, 128)]

plt.figure(figsize=(10, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    plt.bar(x + i * bar_width, data[:, i], width=bar_width, label=model, color=color)

    # Display values on bars
    for j in range(len(metrics)):
        plt.text(x[j] + i * bar_width, data[j, i] + 0.02, f'{data[j, i]:.2f}', 
                 ha='center', fontsize=10, fontweight='bold')

# Labels and formatting
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.title("Comparison of Standalone Average Performance Metrics Across Models", fontsize=14)
plt.xticks(x + bar_width, metrics, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.savefig("standalone_model_comparison_plot.pdf", dpi=500, bbox_inches='tight')
plt.show()

