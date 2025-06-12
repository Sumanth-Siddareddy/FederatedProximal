
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
   git clone https://github.com/Sumanth-Siddareddy/FederatedProximal
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
- numpy==1.23.5
- pandas==1.5.3
- scipy==1.10.0
- scikit-learn==1.3.0
- matplotlib==3.7.1
- seaborn==0.12.2
- tensorflow==2.12.0
- opencv-python==4.10.0
- torch==2.0.1
- torchvision
- torchaudio
- jupyterlab==3.5.2


## Methodology
- Data augmentation using Keras' `ImageDataGenerator` (rotation, shift, zoom).
- A pre-trained ResNet50 model was fine-tuned on the MRI dataset.
- Evaluation includes hold-out validation and cross-validation.
- Model hyperparameters were tuned via grid search.

## Evaluation Method
- **Validation Strategy:** Held-out test
- The global model was periodically evaluated every 10 rounds on a held-out centralized test set over 100 communication rounds.
- The test set remains constant, and you're not changing splits across folds.
- This is better described as periodic evaluation on a held-out test set rather than cross-validation
- Evaluation metrics include accuracy, loss, precision, recall, and F1 score. This approach enabled tracking of model convergence and generalization performance across training


## Assessment Metrics
- **Accuracy:** Overall classification accuracy.
- **Precision, Recall, F1-Score:** Evaluated per class to handle class imbalance.
- **Confusion Matrix:** Visual analysis of classification performance.
- **Justification:** Given the clinical significance, recall and F1-score are emphasized to reduce false negatives.

## Computing Infrastructure
- **OS:** Windows 11
- **GPU:** NVIDIA 
- **RAM:** 16GB
- **Software:** Python 3.8, TensorFlow, PyTorch

## Limitations
- Integrate BlockChain to improve privacy.
- Integrate user incentive mechanism for client motivation is not yet done

## Citations
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. pages 1273–1282.


