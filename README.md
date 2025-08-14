# Music Genre Classification Using Convolutional Neural Networks
DS 6050: Deep Learning Final Project
Tools and Libraries Used:  Python, PyTorch, Torchaudio, Librosa, Scikit-learn, Matplotlib, Seaborn  


# Project Overview
This project implements a deep learning model to automatically classify music into one of ten genres using Convolutional Neural Networks (CNNs) trained on audio spectrograms. The goal is to explore how CNNs can extract meaningful frequency and temporal features from raw audio and apply them to music information retrieval (MIR) tasks.

Music genre classification plays a key role in digital audio indexing, recommendation systems, and intelligent content curation. By framing audio signals as 2D Mel-spectrograms, this project applies proven techniques from computer vision to solve a classic MIR challenge.


**Key Features Include:** 
- Custom CNN Architecture: Three convolutional blocks with batch normalization, ReLU activations, and dropout regularization to prevent overfitting.
- Audio Preprocessing Pipeline: WAV files are converted to mono, downsampled to 22.05kHz, and transformed into log-scaled Mel-spectrograms using `torchaudio` and `librosa`.
- Performance Metrics:  
  - Accuracy: 77.5% on the test set  
  - Detailed classification report (precision, recall, F1-score)  
  - Confusion matrix visualization
- Evaluation Strategy: Stratified train/val/test split (80/10/10) to maintain balanced class distribution and evaluate model generalization.

