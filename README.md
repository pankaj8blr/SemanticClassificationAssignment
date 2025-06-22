# Automated Waste Classification Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify waste images into seven categories. This project demonstrates end-to-end image classification with preprocessing, data augmentation, training, and evaluation using TensorFlow/Keras.

---

## Project Objectives

- Automate the classification of waste images into appropriate categories.
- Tackle **class imbalance** using **data augmentation**.
- Improve model generalization and evaluate performance.
- Extract key insights from training and testing metrics.

---

## Dataset Overview

The dataset contains images labeled into the following 7 classes:

1. **Cardboard**
2. **Food_Waste**
3. **Glass**
4. **Metal**
5. **Other**
6. **Paper**
7. **Plastic**

These images are resized, normalized, and split for training, validation, and testing.

---

## Technologies and Libraries

- Python 3.x
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (cv2)
- Scikit-learn
- Development Environment: **Google Colab**
- Version Control: Git

---

## Project Workflow

### 1. **Data Preprocessing**
- Loaded image dataset into NumPy arrays
- Normalized pixel values
- Label-encoded categories using `LabelEncoder`
- Converted labels to categorical using `to_categorical`

### 2. **Exploratory Data Analysis**
- Bar plot visualizations of class distribution
- Observed significant **class imbalance**

### 3. **Data Augmentation**
To enhance training data and reduce overfitting:
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

---

## Evaluation on Test Set (Before Augmentation)
Classification Report:
- Accuracy: 13%
- Model predicted only Food_Waste class
- All other classes had precision, recall, f1-score = 0.00
- Severe bias due to class imbalance

---

## Key Insights
- Dataset Analysis
    - Heavy class imbalance
    - Visually similar classes (Paper vs. Cardboard, Plastic vs. Other)
- Model Training
    - Without augmentation, model overfits and predicts dominant class only.
    - With augmentation, better generalization and balanced accuracy.

---

## Acknowledgements
    Developed as part of an assignment to cover the topic of convolutional neural networks (CNNs) and their applications in image processing. This assignment is part of Executive Post Graduate Program in Machine Learning and AI - IIIT,Bangalore.
    
---

## Contact
- Pankaj Kumar Agrawal  pankaj8blr@gmail.com


