
# Title: Federated Proximal Algorithm for Brain Tumor Classification : Addressing Non-IID Challenges with Data Augmentation

## Description
This project implements an AI-based application using deep learning ResNet50 architecture to classify brain MRI images into categories such as Glioma, Meningioma, Pituitary, and No Tumor. The primary objective is to explore the real-world utility of deep learning for tumor detection and classification in a medical imaging context.

## Dataset Information
- **Source:** [Mention dataset name/source, e.g., Brain Tumor Dataset from Kaggle or custom collected]
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Size:** [e.g., 3,264 training images and 800 test images]
- **Format:** JPEG/PNG grayscale or RGB images
- **Preprocessing:** Resizing to 224x224, normalization, data augmentation (if any)

## Code Information
- **Language:** Python
- **Framework:** TensorFlow/Keras (or PyTorch)
- **Model:** VGG19 fine-tuned for multi-class classification
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (or other, specify)
- **Regularization:** Dropout, L2, etc.
- **Training Strategy:** Custom training loop or `model.fit()`

## Usage Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset:
   Place dataset in `./dataset/train` and `./dataset/test` following subdirectory structure by class.

4. Run training:
   ```bash
   python train.py
   ```

5. Evaluate model:
   ```bash
   python evaluate.py
   ```

## Requirements
- Python 3.8+
- TensorFlow >= 2.9.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenCV
- Seaborn

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## Methodology
- Data augmentation using Keras' `ImageDataGenerator` (rotation, shift, zoom).
- A pre-trained VGG19 model was fine-tuned on the MRI dataset.
- Evaluation includes hold-out validation and cross-validation.
- Model hyperparameters were tuned via grid search.

## Evaluation Method
- **Validation Strategy:** 80/20 Train/Test split, followed by 5-fold Cross-Validation
- **Ablation Study:** E.g., model without augmentation, model without transfer learning
- **Cross-dataset Validation:** [Briefly describe if applicable]

## Assessment Metrics
- **Accuracy:** Overall classification accuracy.
- **Precision, Recall, F1-Score:** Evaluated per class to handle class imbalance.
- **Confusion Matrix:** Visual analysis of classification performance.
- **Justification:** Given the clinical significance, recall and F1-score are emphasized to reduce false negatives.

## Computing Infrastructure
- **OS:** Ubuntu 20.04 / Windows 11
- **GPU:** NVIDIA Tesla T4 / RTX 3060 (12GB VRAM)
- **RAM:** 16GB+
- **Software:** Python 3.8, TensorFlow 2.9, CUDA 11.2

## Limitations
- Model performance may degrade on images with poor contrast or noise.
- Dataset size limits generalization to real-world data.
- Further evaluation on external datasets is needed for clinical applicability.
- Computationally intensive for real-time or mobile inference.

## Citations
If you use this project or dataset, please cite:

```
@article{yourarticle2025,
  title={Deep Learning for Brain Tumor Classification: An AI Application Study},
  author={Your Name, Collaborators},
  journal={PeerJ Computer Science},
  year={2025},
  note={Submitted}
}

@ARTICLE{bib24,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial intelligence and statistics},
  pages={1273--1282},
  year={2017},
  organization={PMLR}
}

@ARTICLE{bib25,
  title={Federated optimization in heterogeneous networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  journal={Proceedings of Machine learning and systems},
  volume={2},
  pages={429--450},
  year={2020}
}
```

