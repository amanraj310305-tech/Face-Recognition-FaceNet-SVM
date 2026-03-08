# Face Recognition using FaceNet Embeddings and SVM Classification

## Overview

This project implements an end-to-end face recognition system that combines deep learning based feature extraction with classical machine learning classification.

The system uses **MTCNN** for robust face detection and **FaceNet** for generating high-dimensional facial embeddings. These embeddings are then used to train a **Support Vector Machine (SVM)** classifier capable of identifying individuals from facial images.

The goal of the project is to demonstrate a practical pipeline for face recognition that separates **deep feature extraction** from **classification**, enabling efficient and scalable identity recognition.

---

## Key Features

* Automatic **face detection using MTCNN**
* **Deep feature extraction using FaceNet embeddings (512-dimensional)**
* **SVM based identity classification**
* Automated dataset loading and preprocessing
* Model training and evaluation pipeline
* Confusion matrix visualization for performance analysis
* Saved trained model and label encoder for future inference

---

## System Architecture

The pipeline follows four major stages:

### 1. Face Detection

Faces are detected using **MTCNN (Multi-Task Cascaded Convolutional Networks)**.
This network detects bounding boxes around faces and extracts the face region from the original image.

### 2. Face Preprocessing

Detected faces are:

* Cropped from the image
* Resized to **160×160 pixels**
* Normalized before embedding generation

This step ensures consistent input for the FaceNet model.

### 3. Face Embedding Generation

The system uses **FaceNet** to convert each detected face into a **512-dimensional embedding vector**.

These embeddings represent unique facial features and serve as compact numerical representations of identities.

Advantages of embeddings:

* Robust to lighting variations
* Robust to minor pose changes
* Efficient for machine learning models

### 4. Classification

Embeddings are used to train a **Support Vector Machine (SVM)** classifier with an RBF kernel.

The classifier learns decision boundaries that separate identities in the embedding space.

---

## Dataset Structure

The dataset is organized as a directory containing subfolders for each individual.

Example:

dataset/

person1/

image1.jpg

image2.jpg

person2/

image1.jpg

image2.jpg

Each folder represents one class label.

---

## Model Training Pipeline

The training pipeline performs the following steps:

1. Load images from dataset
2. Detect faces using MTCNN
3. Crop and resize detected faces
4. Generate FaceNet embeddings
5. Encode labels using LabelEncoder
6. Split dataset into training and testing sets
7. Train SVM classifier
8. Evaluate model accuracy
9. Generate confusion matrix visualization
10. Save trained model and label encoder

Saved artifacts:

* face_recognition_model.pkl
* label_encoder.pkl

---

## Evaluation

The model performance is evaluated using:

* **Training Accuracy**
* **Testing Accuracy**
* **Confusion Matrix Visualization**

The confusion matrix provides insight into classification performance across different identities.

---

## Technologies Used

* Python
* OpenCV
* TensorFlow
* MTCNN
* FaceNet (keras-facenet)
* Scikit-learn
* NumPy
* Matplotlib

---

## Project Structure

face-recognition

face_recognition.py

dataset/

face_recognition_model.pkl

label_encoder.pkl

README.md

---

## Applications

Face recognition systems like this can be applied to:

* Identity verification
* Smart surveillance systems
* Secure authentication
* Attendance systems
* Human-computer interaction

---

## Future Improvements

Possible extensions of this system include:

* Real-time webcam face recognition
* Unknown face detection
* Incremental learning for new identities
* Deployment as a web application
* Mobile edge inference optimization



This project demonstrates practical understanding of **computer vision pipelines, deep feature embeddings, and machine learning based classification for face recognition tasks.**

