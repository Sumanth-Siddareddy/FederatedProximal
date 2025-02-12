import matplotlib.pyplot as plt
import numpy as np

# Function to convert RGB to a matplotlib color
def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)  # Normalize to [0, 1] range

# Data for the models
metrics = {
    'mu=0': {
        'Accuracy': [64.91, 73.68, 76.51, 79.33, 80.85, 80.70, 83.22, 83.75, 84.13, 85.51],
        'Loss': [1.0776, 1.0031, 0.9802, 0.9512, 0.9365, 0.9324, 0.9121, 0.9087, 0.8982, 0.8901],
        'Precision': [0.6586, 0.7439, 0.7754, 0.7911, 0.8057, 0.8231, 0.8395, 0.8345, 0.8385, 0.8523],
        'Recall': [0.6491, 0.7368, 0.7651, 0.7933, 0.8085, 0.8070, 0.8322, 0.8375, 0.8413, 0.8551],
        'F1 Score': [0.6518, 0.7325, 0.7656, 0.7886, 0.8064, 0.8052, 0.8312, 0.8336, 0.8377, 0.8518],
    },
    'mu=0.1': {
        'Accuracy': [69.41, 70.33, 78.87, 79.33, 80.93, 81.62, 81.92, 83.52, 83.37, 72.77],
        'Loss': [1.0394, 1.0294, 0.9582, 0.9451, 0.9325, 0.9321, 0.9223, 0.9049, 0.9071, 1.0097],
        'Precision': [0.6913, 0.7410, 0.7841, 0.7899, 0.8038, 0.8135, 0.8173, 0.8391, 0.8456, 0.7866],
        'Recall': [0.6941, 0.7033, 0.7887, 0.7933, 0.8093, 0.8162, 0.8192, 0.8352, 0.8337, 0.7277],
        'F1 Score': [0.6922, 0.7094, 0.7840, 0.7902, 0.8008, 0.8091, 0.8139, 0.8342, 0.8301, 0.7043],
    },
    'mu=0.4': {
        'Accuracy': [69.57, 73.76, 75.44, 78.79, 78.79, 81.24, 82.84, 83.07, 85.13, 84.52],
        'Loss': [1.0490, 1.0014, 0.9819, 0.9595, 0.9621, 0.9256, 0.9166, 0.9132, 0.8993, 0.8944],
        'Precision': [0.6924, 0.7368, 0.7612, 0.7815, 0.7882, 0.8079, 0.8251, 0.8268, 0.8488, 0.8454],
        'Recall': [0.6957, 0.7376, 0.7544, 0.7879, 0.7879, 0.8124, 0.8284, 0.8307, 0.8513, 0.8452],
        'F1 Score': [0.6927, 0.7315, 0.7509, 0.7748, 0.7787, 0.8074, 0.8231, 0.8212, 0.8480, 0.8451],
    },
    'mu=0.7': {
        'Accuracy': [70.33, 70.10, 74.98, 78.11, 79.41, 79.86, 82.38, 83.37, 82.23, 84.44],
        'Loss': [1.0394, 1.0272, 0.9849, 0.9580, 0.9532, 0.9414, 0.9259, 0.9132, 0.9157, 0.9039],
        'Precision': [0.7064, 0.7172, 0.7519, 0.7823, 0.7901, 0.8006, 0.8202, 0.8311, 0.8182,0.8415],
        'Recall': [0.7033, 0.7010, 0.7498, 0.7811, 0.7941, 0.7986, 0.8238, 0.8337, 0.8223,0.8444],
        'F1 Score': [0.6981, 0.7064, 0.7488, 0.7793, 0.7880, 0.7988, 0.8158, 0.8274, 0.8116,0.8420],
    },
    'mu=1.0': {
    'Accuracy': [69.26, 74.37, 79.48, 81.16, 82.07, 83.30, 84.13, 85.58, 83.60, 87.19],
    'Loss': [1.0405, 0.9980, 0.9492, 0.9336, 0.9227, 0.9102, 0.8999, 0.8904, 0.9020, 0.8774],
    'Precision': [0.7011, 0.7615, 0.7903, 0.8130, 0.8167, 0.8321, 0.8402, 0.8554, 0.8390, 0.8734],
    'Recall': [0.6926, 0.7437, 0.7948, 0.8116, 0.8207, 0.8330, 0.8413, 0.8558, 0.8360, 0.8719],
    'F1 Score': [0.6950, 0.7441, 0.7916, 0.8098, 0.8154, 0.8323, 0.8398, 0.8543, 0.8255, 0.8716],
    }
}




# Metrics to plot
metrics_names = ['Accuracy', 'Loss', 'Precision', 'Recall', 'F1 Score']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'thistle', 'peachpuff']

# Training rounds
rounds = [i * 10 for i in range(1, 11)]  
mu_values = list(metrics.keys())
x = np.arange(len(rounds))  

# Plot grouped bar charts for each metric
for metric in metrics_names:
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
    width = 0.13 

    for i, model in enumerate(mu_values):
        data = metrics[model][metric]
        ax.bar(x + i * width, data, width, label=model, color=colors[i])

    # Set properties with bigger fonts
    ax.set_title(f'Comparison of {metric} Across Models', fontsize=20, fontweight='bold')
    ax.set_xlabel('Training Rounds', fontsize=20, fontweight='bold', labelpad=12)
    ax.set_ylabel(metric, fontsize=18, fontweight='bold')
    ax.set_xticks(x + width / 2)  
    ax.set_xticklabels(rounds, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)  # Set Y-axis number font size to 14


    # Improve legend visibility
    ax.legend(title='', fontsize=19, title_fontsize=20, loc='center', bbox_to_anchor=(0.5, 1.12), ncol=len(mu_values))

    # Grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent cutting
    plt.subplots_adjust(bottom=0.11)  # Adds space below for labels

    # Save without cutting labels
    plt.savefig(f"{metric}_comparison_plot.pdf", dpi=2000, pad_inches=0)
    plt.show()
