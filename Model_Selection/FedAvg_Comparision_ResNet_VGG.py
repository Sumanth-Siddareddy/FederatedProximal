
# ------------------------ FedAvg of ResNet50 VS VGG19 --------------------------------
import numpy as np
import matplotlib.pyplot as plt

def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)

# ResNet 50 + CBAM Test Accuracies
resnet_accuracies = np.array([
    [29.642857, 25.952381, 36.372549, 26.833333],
    [29.761905, 30.833333, 38.333333, 37.666667],
    [27.380952, 32.261905, 30.098039, 31.333333],
    [30.238095, 29.642857, 25.686275, 25.0],
    [44.166667, 50.0, 58.137255, 53.833333],
    [41.547619, 44.285714, 42.45098, 46.166667],
    [42.261905, 42.142857, 38.921569, 45.0],
    [47.02381, 50.357143, 47.45098, 52.166667],
    [49.166667, 54.047619, 50.980392, 55.0],
    [49.642857, 52.380952, 53.039216, 56.166667]
])

# VGG 19 + CBAM Test Accuracies
vgg_accuracies = np.array([
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0],
    [25.0, 25.0, 25.0, 25.0]
])

# Compute Average Accuracy
resnet_avg_accuracy = np.mean(resnet_accuracies)
vgg_avg_accuracy = np.mean(vgg_accuracies)

# Bar Graph
models = ["ResNet 50 + CBAM", "VGG 19 + CBAM"]
avg_accuracies = [resnet_avg_accuracy, vgg_avg_accuracy]

plt.figure(figsize=(10, 7))
plt.bar(models, avg_accuracies, color=[rgb(135, 206, 250), rgb(144, 238, 144)])
plt.ylabel("Average Test Accuracy (%)")
plt.title("Comparison of ResNet 50 + CBAM and VGG 19 + CBAM on FedAvg")
plt.ylim(20, 60)
plt.savefig("FedAvg_Comparision.pdf", dpi=500, bbox_inches='tight')
plt.show()
