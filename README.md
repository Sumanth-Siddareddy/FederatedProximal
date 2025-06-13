
# Title: Improving Brain Tumor Classification with Federated Proximal: Solving Non-IID Challenges Using Data Augmentation

## Description
We developed a hybrid federated learning framework using FedProx and ResNet50 to improve model accuracy on uneven healthcare data while preserving privacy. By applying data augmentation to non-IID data, our method achieved strong performance, making training more balanced and effective.

## Dataset Information and load data
- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Size:** 5712 training images and 1311 test images
- **Format:** JPG images
- **Preprocessing:**
   - Data pre-processing, all images were normalized to pixel values in the range [0, 1].
   - Images were resized to a consistent input size of 224 × 224 pixels, suitable for the ResNet50 architecture.
   - Data Agumentation 

- **Load Data:** 
   The dataset has two folders training and testing already, and we use training, testing set to split among all the clients in NON-IID format, train the model using fedprox algorithm and we keep a copy of original testing data (1311 images) for evaluating the model 

## Code Information
- **Language:** Python
- **Frameworks:** TensorFlow and PyTorch
- **Model & Architecture:** Proposed Model Architecture - ResNet50 with Attention Head, Proposed FL Architecture
- **Loss Function:** Categorical Crossentropy, sparse_softmax_cross_entropy
- **Optimizer:** SGD, Adam
- **Regularization:** Dropout

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
### 1) Data Preparation
   - Data Splitting into NON-IID format (To simulate real world scenario).
   - Calculate Jensen-Shannon Divergence of NON-IID dataset.
### 2) Model Selection :
   - Among ResNet18, ResNet50, VGG19 select best performed model on NON-IID dataset (ResNet50 and VGG19 perform well)
   - Comparision of ResNet50 and VGG19 based on thier performance on FedAvg. (ResNet50 perform well)
### 3) Data Agumentation
### 4) Integrate Attention Head to ResNet50 (Proposed model)
### 5) Federated Proximal Implementation using proposed FL Architecture
### 6) Training the clients and update the global model with trained model weights of client models and test the client models with the data clients have.
### 7) Evaluate the global model using Hold-Out testing and record the accuracy metrics like Accuracy, Loss, F1 Score, Precision, Recall.

## Evaluation Method
- **Validation Strategy:** Hold-out test
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
- Integration of blockchain to enhance privacy has not yet been implemented.
- A user incentive mechanism to motivate client participation is also lacking.

## Citations
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. pages 1273–1282.