The goal of this project is to build an image classifier that distinguishes between images of cats and dogs with high accuracy. This is achieved by extracting features from the images using a pre-trained VGG16 model and training an SVM model on these features. This method combines the power of transfer learning and traditional machine learning.

Dataset
The dataset contains images of cats and dogs in separate folders for training and testing:

Train Folder: Contains labeled images of cats and dogs.
Test Folder: Used for validation and contains images of both classes.
Directory Structure
Ensure that your dataset is organized as follows:

dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
Model Pipeline

Data Loading and Preprocessing:

Images are resized to 128x128 pixels.
Labels are assigned (0 for cats, 1 for dogs).
Images are normalized using preprocess_input from Keras.

Feature Extraction:

A pre-trained VGG16 model (without top layers) is used to extract high-level features from the images.
Model Training:

The SVM model with an RBF kernel is trained on the extracted features.
Cross-validation is used to evaluate model performance across multiple splits.
Evaluation:

Model accuracy is evaluated on both training and test datasets.
Confusion matrices and accuracy metrics are plotted for detailed performance insights.
Requirements
To run the project, you need the following libraries:

numpy
opencv-python
matplotlib
seaborn
scikit-learn
tensorflow (for VGG16 and preprocessing)
Install dependencies with:

pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow
Code
Training and Testing
The main code is located in the file svm_cats_vs_dogs.py. To run the code, place the dataset in the correct directory structure as shown above, and execute:


python svm_cats_vs_dogs.py
Key Code Sections
Data Loading: Loads and preprocesses the images.
Feature Extraction: Uses VGG16 for transfer learning.
Model Training and Evaluation: Trains the SVM model and evaluates its accuracy.
Visualizations
The code includes:

Sample images from the dataset to visualize the data.
A confusion matrix to show model performance in detail.
Accuracy vs. Data Split plot, showing how model accuracy changes over different data splits.
Results
The final model achieved approximately 85-90% accuracy on the test dataset, showing good performance in distinguishing between cats and dogs.

Repository Structure

├── svm_cats_vs_dogs.py     # Main code file
├── README.md               # Project overview and instructions
└── dataset/                # Folder containing the dataset
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
Sample Output
Training and validation accuracy plotted across multiple splits:



Acknowledgements
Thanks to Kaggle for providing the dataset.
Transfer learning is performed using the VGG16 model provided by the Keras library.
