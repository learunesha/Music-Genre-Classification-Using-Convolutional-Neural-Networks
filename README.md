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


**Dataset:**
- Name: GTZAN Genre Collection
- Source: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- Content:
  - genres original: A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
  - images original: A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
  - 2 CSV files: Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models).
