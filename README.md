import numpy as np
import pandas as pd
import librosa
import resampy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
def load_data(data_dir):
    labels, features = [], []
    for label_dir in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, label_dir)):
            filepath = os.path.join(data_dir, label_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)
            audio = resampy.resample(audio, sr, 16000)
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            features.append(mfcc)
            labels.append(label_dir)
    return np.array(features), np.array(labels)

# Preprocess data
def preprocess_data(features, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    return features, labels_encoded

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    X_train_res = X_train_res.reshape(X_train_res.shape[0], features.shape[1], features.shape[2], 1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])
    return history

# Plot results
def plot_results(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main execution
data_dir = 'data'
features, labels = load_data(data_dir)
features, labels = preprocess_data(features, labels)
model = build_model(input_shape=(features.shape[1], features.shape[2], 1))
history = train_model(model, features, labels)
plot_results(history)
