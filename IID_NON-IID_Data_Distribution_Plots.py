import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# Tumor categories
categories = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
x = np.arange(len(categories))  # X-axis positions

# IID data (update here if needed)
model_1_train = [1021, 1039, 1295, 1157]
model_1_test = [210, 210, 255, 150]

model_2_train = [1021, 1039, 1295, 1157]
model_2_test = [210, 210, 255, 150]

model_3_train = [1021, 1039, 1295, 1157]
model_3_test = [210, 210, 255, 150]

model_4_train = [1021, 1039, 1295, 1157]
model_4_test = [210, 210, 255, 150]

# NON_IID_Data for each model
# model_1_train = [1021, 100, 100, 100]
# model_1_test = [210, 32, 50, 50]

# model_2_train = [100, 1039, 100, 100]
# model_2_test = [30, 210, 50, 50]

# model_3_train = [100, 100, 1295, 100]
# model_3_test = [30, 32, 255, 50]

# model_4_train = [100, 100, 100, 1157]
# model_4_test = [30, 32, 50, 150]

# Bar width
bar_width = 0.2

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 10), dpi=1000)

# Plot bars
ax.bar(x - 1.5 * bar_width, model_1_train, bar_width, label="Model 1 - Training", color='#1f77b4')
ax.bar(x - 1.5 * bar_width, model_1_test, bar_width, bottom=model_1_train, label="Model 1 - Testing", color='#aec7e8')

ax.bar(x - 0.5 * bar_width, model_2_train, bar_width, label="Model 2 - Training", color='#9467bd')
ax.bar(x - 0.5 * bar_width, model_2_test, bar_width, bottom=model_2_train, label="Model 2 - Testing", color='#c5b0d5')

ax.bar(x + 0.5 * bar_width, model_3_train, bar_width, label="Model 3 - Training", color='#2ca02c')
ax.bar(x + 0.5 * bar_width, model_3_test, bar_width, bottom=model_3_train, label="Model 3 - Testing", color='#98df8a')

ax.bar(x + 1.5 * bar_width, model_4_train, bar_width, label="Model 4 - Training", color='#d62728')
ax.bar(x + 1.5 * bar_width, model_4_test, bar_width, bottom=model_4_train, label="Model 4 - Testing", color='#ff9896')

# Title and axis labels
ax.set_title("Combined Sample Distribution for Models 1, 2, 3, and 4", fontsize=28, weight='bold', pad=20)
ax.set_xlabel("Tumor Type", fontsize=22, weight='bold', labelpad=15)
ax.set_ylabel("Number of Samples", fontsize=22, weight='bold', labelpad=15)

# Tick labels
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=20, weight='bold')
ax.tick_params(axis='y', labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Define bold font properties for legend
legend_font = FontProperties()
legend_font.set_size(22)
legend_font.set_weight('bold')

legend_title_font = FontProperties()
legend_title_font.set_size(24)
legend_title_font.set_weight('bold')

# Enhanced legend with bold, clear text
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=4,
    prop=legend_font,  # Apply bold font to items
    title="Training and Testing Models",
    title_fontproperties=legend_title_font,  # Apply bold font to title
    frameon=False
)

# Clean layout
plt.tight_layout()

# Save high-quality PDF without legend cropping
# Change name of pdf file based on NON-IID or IID data
fig.savefig("IID_DATA_REPRESENTATION.pdf", format="pdf", dpi=900, bbox_inches='tight')
plt.show()

