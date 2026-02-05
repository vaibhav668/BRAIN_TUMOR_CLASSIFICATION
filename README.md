📌 Overview

This project focuses on automatic brain tumor classification from MRI images using Deep Learning.
I implemented and compared custom Convolutional Neural Networks (CNNs) with transfer learning models to analyze performance, generalization, and training efficiency in a medical imaging setting.

The goal was not just prediction, but understanding trade-offs between handcrafted architectures and pre-trained models when working with limited medical data.

🧠 Problem Statement

Manual analysis of MRI scans is:

Time-consuming

Expertise-dependent

Prone to human error under high workload

This project aims to:

Assist radiologists by providing automated tumor classification

Evaluate how deep learning models perform on medical imaging data

Compare custom CNNs vs transfer learning approaches

🛠️ Approach & Methodology
1️⃣ Data Preparation

MRI brain scan images were preprocessed using:

Resizing & normalization

Label encoding

Train–validation split

Care was taken to reduce overfitting due to limited data.

2️⃣ Model Architectures
🔹 Custom CNN (from scratch)

Designed CNN architectures with:

Convolutional layers

MaxPooling

Dropout for regularization

Fully connected layers

Used to study:

Learning behavior from raw data

Overfitting vs generalization

🔹 Transfer Learning

Implemented pre-trained models (e.g., ResNet / MobileNet)

Fine-tuned higher layers for tumor classification

Compared against custom CNNs in terms of:

Accuracy

Training time

Generalization performance

3️⃣ Model Training & Evaluation

Framework: PyTorch

Loss function: Cross-Entropy Loss

Optimizer: Adam

Evaluation metrics:

Accuracy

Validation loss trends

Performance comparison conducted across models.

📊 Key Observations

Transfer learning models achieved higher accuracy with faster convergence

Custom CNNs provided better insight into:

Feature extraction

Overfitting behavior

Highlighted the importance of pre-trained representations in medical imaging tasks with limited data.

🖥️ Deployment

Built an interactive Streamlit web application

Users can:

Upload MRI images

Receive real-time tumor classification results

Demonstrates end-to-end ML workflow from model → inference → UI

🧰 Tech Stack

Programming: Python

Deep Learning: PyTorch

Computer Vision: OpenCV

ML & Data: NumPy, Pandas, Matplotlib

Web App: Streamlit
