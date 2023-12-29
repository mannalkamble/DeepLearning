# Deep Learning Course

This repo contained the completed codes as part of a deep learning course. The focus of these was to apply deep learning techniques using Python and PyTorch to solve various problems related to image classification, generative adversarial networks (GANs), and object detection.

### 1: Improving the FashionMNIST Classifier
**Objective:** Enhance classification performance of the FashionMNIST dataset using a neural network.  
- **Approach:** Transition from a logistic regression model to a neural network with three hidden layers (256, 128, 64 neurons) and ReLU activations, culminating in a 10-dimensional output layer for class prediction.  
- **Dataset Loading:** FashionMNIST dataset loaded using PyTorch:  
  ```python
  trainingdata = torchvision.datasets.FashionMNIST('./FashionMNIST/', train=True, download=True, transform=torchvision.transforms.ToTensor())  
  testdata = torchvision.datasets.FashionMNIST('./FashionMNIST/', train=False, download=True, transform=torchvision.transforms.ToTensor())  
  ```
- **Evaluation:** Training and test loss curves were analyzed. Sample predictions and class probabilities for three images from the validation set were visualized and discussed.  

### 2: Implementing Backpropagation
**Objective:** Manually implement backpropagation for several operations.  
- **Details:** Complete the gradient calculations for addition, multiplication, power, and ReLU without using autodiff. The focus is on deepening the understanding of the gradient computation process.  

### 3: Training GANs
**Objective:** Train a Generative Adversarial Network (GAN) on the FashionMNIST dataset.  
- **Quality Measures:** Investigated and reported several measures of GAN quality, comparing the generated image quality between the FashionMNIST GAN and the MNIST Conditional GAN.  

### 4: Object Detection
**Objective:** Fine-tune a pretrained Torchvision object detection model.  
- **Tutorial Implementation:** Adapted a PyTorch tutorial for object detection to include a different backbone, comparing the performance of the original and modified models over 10 training epochs.  
- **Test:** Evaluated both models on a standard test image to compare performance differences.  

### 5: Training AlexNet
**Objective:** Build and train an AlexNet architecture.  
- **Environment:** The notebook can be accessed and edited in Google Colab or any other Python IDE of choice.  

## Additional Notes
- All assignments were implemented in Google Colab or other Python IDEs, as preferred.  
- Results, including code snippets, plots, and performance metrics, are documented within each respective Jupyter notebook.  

## Dependencies
- Python 3.8+  
- PyTorch  
- Torchvision  
- Matplotlib (for plotting)  
