
# Title: Federated Proximal Algorithm for Brain Tumor Classification : Addressing Non-IID Challenges with Data Augmentation

## Description
The increasing use of electronic health records has transformed healthcare management, yet data sharing among institutions remains challenging due to privacy concerns. Federated learning (FL), a collaborative learning approach, enables model training across distributed data sources while preserving data privacy. However, handling non-independent and identically distributed (non-IID) data poses a significant obstacle to achieving robust global model performance. To overcome these issues, we proposed a hybrid framework to leverage the usage of the Federated Proximal (FedProx) algorithm by considering the ResNet50 architecture. We artificially partitioned IID data into non-IID subsets to simulate real-world conditions. Then, by applying data augmentation techniques, we transformed the non-IID datasets into more uniform distributions. We monitored global model performance of the hybrid framework over 100 training rounds with varying values of the regularization parameter in FedProx. Our proposed approach achieved an accuracy of up to 97.71\% with IID data and 87.19\% in an extreme case of Non-IID data, with precision, recall, and F1 scores also showing significant results. Our study highlights the significance of combining data augmentation and FedProx to reduce data imbalance issues in FL, ensuring equitable and efficient model training across distributed healthcare datasets. Our work contributes to better solutions for privacy-preserving in healthcare applications.

## Dataset Information
- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Size:** 5712 training images and 1311 test images
- **Format:** JPG images
- **Preprocessing:**
   - Data pre-processing, all images were normalized to pixel values in the range [0, 1].
   - Images were resized to a consistent input size of 224 × 224 pixels, suitable for the ResNet50 architecture.
   - Data Agumentation :
      - Shear: ±20%
      - Zoom: ±20%
      -  Rotation: ±90
      - Width and Height Shifts: ±10%
      - Horizontal and Vertical Flips: Randomly applied

## Code Information
- **Language:** Python
- **Framework:** TensorFlow and PyTorch
- **Model:** VGG19 fine-tuned for multi-class classification
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** SGD
- **Regularization:** Dropout, L2, etc.
- **Training Strategy:** Custom training loop or `model.fit()`

## Usage Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/Sumanth-Siddareddy/FederatedProximal
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
- jupyterlab==3.5.2
- os

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

- Specifically, we examined the influence of variations in the μ parameter in FedProx, and several key hyperparameters (such as learning rate and patience). By analyzing the performance metrics for each configuration, we were able to identify the relative contribution of each factor to the final outcome, providing insights into their roles in optimizing federated learning models.


## Assessment Metrics
- **Accuracy, Precision, Recall, F1-Score, Loss:** Global model was periodically evaluated every 10 rounds on a held-out centralized test set over 100 communication rounds on these metrics.
- **Line plot, Bar-graphs, Confusion Matrix:** Visual analysis of classification performance.

## Computing Infrastructure
- **OS:** Windows 11
- **GPU:** NVIDIA (DGX)
- **RAM:** 16GB

## Limitations
- Integrate BlockChain to improve privacy.
- Integrate user incentive mechanism for client motivation is not yet done

## Citations
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. pages 1273–1282.
