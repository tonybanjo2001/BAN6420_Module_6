# BAN6420_Module_6

This file provides an overview of the Python code designed and executed in Google Colab. The code builds, trains, and evaluates a Convolutional Neural Network (CNN) using the TensorFlow and Keras libraries on the Fashion MNIST dataset. This project aims to classify images of fashion items into one of ten categories using deep learning techniques.

Prerequisites
Before running the code, ensure that the following Python libraries are installed in your environment:

NumPy: For numerical computations.
Matplotlib: This is used to plot graphs and visualize data.
TensorFlow and Keras: For building and training the neural network.
The required libraries can be installed by running the following command:

Python
Copy code
!pip install numpy matplotlib tensorflow keras
Dataset
The project uses the Fashion MNIST dataset, which contains 70,000 grayscale images of 10 fashion categories. The dataset is divided into:

Training set: 60,000 images
Test set: 10,000 images
Each image is a 28x28 pixel array, and the corresponding label is an integer from 0 to 9 representing the fashion category.

Steps in the Code
1. Import Libraries
The necessary libraries (numpy, matplotlib, tensorflow) are imported to manage data, plot visualizations, and build the CNN model.

2. Load the Dataset
The Fashion MNIST dataset is loaded using TensorFlow's Keras API:

Python
Copy code
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
3. Preprocess the Data
Normalization: The pixel values of the images are normalized to a range of 0 to 1 by dividing by 255.
Reshaping: The images are reshaped to include a single color channel, making them suitable for input to the CNN model.
Categorical Conversion: The labels are converted to categorical format using to_categorical to be used in the classification process.
4. Build the CNN Model
A Convolutional Neural Network is built using the Sequential API from Keras. The architecture includes:

Convolutional Layers: For extracting features from the images.
MaxPooling Layers: For downsampling the feature maps.
Flatten Layer: For converting the 2D feature maps to 1D.
Dense Layers: For classification, with the final layer using the softmax activation function to output probabilities for each class.
5. Compile the Model
The model is compiled using the Adam optimizer, categorical crossentropy as the loss function, and accuracy as the metric:

Python
Copy code
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
6. Train the Model
The model is trained on the training dataset for 10 epochs with validation on the test dataset:

Python
Copy code
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
7. Evaluate the Model
The trained model is evaluated on the test dataset to determine its accuracy:

Python
Copy code
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
8. Make Predictions
The model makes predictions on the first two images of the test dataset. These predictions are compared with the actual labels, and the images along with their predicted and actual labels are plotted:

Python
Copy code
predictions = model.predict(test_images[:2])
for i in range(2):
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(test_labels[i])}")
    plt.show()
Results
The model achieves a test accuracy of approximately 91%, demonstrating its effectiveness in classifying the Fashion MNIST dataset.

Example Output
Predicted vs. Actual Labels: The first two images of the test set are displayed along with the predicted and actual labels, giving a visual confirmation of the model’s performance.
Conclusion
This project provides a basic yet powerful introduction to deep learning with CNNs using the Fashion MNIST dataset. It covers data preprocessing, model building, training, evaluation, and making predictions with TensorFlow and Keras. The model's accuracy demonstrates its capability to classify fashion items, providing a foundation for more complex and customized image classification tasks.

Usage
To run this project in Google Colab:

Open a new Colab notebook.
Copy and paste the provided code into the notebook.
Run the cells sequentially to install dependencies, build, train, evaluate, and test the CNN model.
Feel free to experiment with different model architectures, hyperparameters, and datasets to further improve and adapt the model to other classification tasks.
