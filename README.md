# Machine Learning & Deep Learning Portfolio
This repository contains a collection of complete ML and DL projects, each built with an end-to-end workflow that includes data preprocessing, exploratory analysis, model development, tuning, and performance evaluation. The projects cover real-world datasets across computer vision and scientific domains.

---

## Repository Structure

### 1. Convolutional Autoencoders – CIFAR-10
Unsupervised learning workflow for image reconstruction using a custom convolutional autoencoder.
Key components:
- Normalization and dataset exploration
- Encoder–decoder architecture using Conv2D, MaxPooling, and Conv2DTranspose
- 40-epoch training with stable convergence (MSE ~ 0.0018, PSNR ~ 27.67 dB)
- Original vs reconstructed image comparisons
- Analysis of latent features and color versus grayscale complexity

Skills: CNNs, Autoencoders, Keras, image preprocessing, GPU training

---

### 2. Digit Classification – SVHN (PyTorch)
Deep learning classification pipeline using the Street View House Numbers dataset.
Key components:
- Data cleaning, normalization, and PyTorch tensor/datamodule pipelines
- Custom augmentation (rotation, jitter, affine transforms)
- Two model architectures:
  - Simple CNN
  - Modified AlexNet for 32×32 RGB inputs
- Performance:
  - Simple CNN: ~86% accuracy
  - AlexNet: ~92% accuracy
- Full evaluation: precision, recall, F1, confusion matrices, runtime per sample

Skills: PyTorch, CNN design, augmentation, performance analysis

---

### 3. Stellar Object Classification – SDSS (Galaxy, Star, QSO)
Machine learning pipeline for classifying astronomical objects using 100,000 structured samples.
Key components:
- Comprehensive preprocessing including outlier capping, scaling, encoding
- Feature correlation, redundancy checks, and Random Forest importance ranking
- Visualizations: PairGrid relationships, boxplots, histograms
- Models trained and compared:
  - Logistic Regression
  - KNN
  - SVM
  - Random Forest (best model)
- Hyperparameter tuning using Grid Search
- Performance:
  - Random Forest: ~98% accuracy, macro-F1 ~ 0.97

Skills: ML pipelines, feature engineering, hyperparameter tuning, scientific data analysis

---

## What This Repository Demonstrates
- Ability to handle both image-based and structured datasets  
- Strong understanding of preprocessing, outlier handling, scaling, and class balance  
- Proficiency in implementing ML models and deep neural networks  
- Clear architectural reasoning and reproducible experiment design  
- Use of advanced evaluation metrics (macro-F1, PSNR, MSE, confusion matrices)  
- Experience working with scientific and real-world noisy datasets  

---

## Technologies Used
- Python  
- PyTorch  
- TensorFlow / Keras  
- scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- GPU acceleration (T4 via Google Colab)  

---

## Navigation
Each project directory contains:
- A project-specific README  
- Structured notebooks or scripts  
- Plots, tables, and evaluation outputs  
- Notes explaining reasoning, design choices, and reflections  

---

## Contact
For any inquiries or discussion about this repository or individual projects, feel free to reach out.
