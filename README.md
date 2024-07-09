

# Traffic Sign Recognition System

## Overview
This project involves training a Convolutional Neural Network (CNN) to recognize traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The trained model is then used to classify new images of traffic signs.

## Training Code

### Dependencies
- `numpy`
- `matplotlib`
- `keras`
- `opencv-python`
- `scikit-learn`
- `pandas`
- `tensorflow`

### Setup
1. **Dataset Preparation**: The dataset is expected to be organized in folders where each folder corresponds to a class of traffic signs, with images stored within these folders.
2. **Parameters**: Configure the paths to your dataset (`path`) and label file (`labelFile`). Set parameters such as `batch_size_val`, `epochs_val`, and `imageDimesions` according to your dataset characteristics.

### Data Preprocessing
- **Image Loading**: Images are loaded using OpenCV and split into training, validation, and testing sets.
- **Preprocessing**: Images undergo grayscale conversion, histogram equalization, and normalization to enhance model performance.

### Model Architecture
- **CNN Architecture**: Defined using Keras, with convolutional layers, max pooling, dropout for regularization, and dense layers for classification.

### Training
- **Data Augmentation**: Image augmentation using `ImageDataGenerator` to generate variations of training images, enhancing model robustness.
- **Training Process**: The model is compiled with Adam optimizer and categorical cross-entropy loss. Training progress is monitored through accuracy and loss metrics.

### Evaluation
- **Validation**: Model performance is evaluated on validation data during training.
- **Testing**: Final evaluation on a separate test set to measure model generalization.

### Results
- **Visualization**: Training and validation loss/accuracy curves are plotted using Matplotlib.
- **Model Persistence**: Trained model is serialized using pickle for future deployment.

## Test Code

### Dependencies
- Same dependencies as training code.

### Usage
1. **Model Loading**: Loads the trained model using pickle.
2. **Image Prediction**: Reads and preprocesses images from a specified folder (`test_data_folder`), predicts the class probabilities using the trained CNN, and overlays results onto the original images.
3. **Visualization**: Displays the original image with predicted class and probability if above a specified threshold (`threshold`).

### Output
- Displays each processed image with predicted class and probability, demonstrating the model's inference capabilities on new data.

