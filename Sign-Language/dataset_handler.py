import pandas as pd
import os
import cv2
import numpy as np

def load_dataset():
    """Load and preprocess the dataset from CSV files and PNG images."""
    train_data = pd.read_csv('Dataset/sign_mnist_train/sign_mnist_train.csv')
    test_data = pd.read_csv('Dataset/sign_mnist_test/sign_mnist_test.csv')

    X_train = train_data.iloc[:, 1:].values / 255.0
    Y_train = train_data.iloc[:, 0].values

    X_test = test_data.iloc[:, 1:].values / 255.0
    Y_test = test_data.iloc[:, 0].values

    png_images = []
    png_labels = []
    for file_name in os.listdir('Dataset'):
        if file_name.endswith('.png'):
            img = cv2.imread(os.path.join('Dataset', file_name), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28)).flatten() / 255.0
            png_images.append(img_resized)

            label = int(file_name.split('sign')[1].split('.')[0])
            png_labels.append(label)

    X_train = np.vstack([X_train, png_images])
    Y_train = np.hstack([Y_train, png_labels])

    return X_train, Y_train, X_test, Y_test

def load_dataset_with_folders():
    """Load and preprocess the dataset from CSV files, PNG images, and A-Z folders."""
    train_data = pd.read_csv('Dataset/sign_mnist_train/sign_mnist_train.csv')
    test_data = pd.read_csv('Dataset/sign_mnist_test/sign_mnist_test.csv')

    X_train = train_data.iloc[:, 1:].values / 255.0 
    Y_train = train_data.iloc[:, 0].values

    X_test = test_data.iloc[:, 1:].values / 255.0
    Y_test = test_data.iloc[:, 0].values

    # png to dataset
    png_images = []
    png_labels = []
    for file_name in os.listdir('Dataset'):
        if file_name.endswith('.png'):
            img = cv2.imread(os.path.join('Dataset', file_name), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28)).flatten() / 255.0
            png_images.append(img_resized)

            label = int(file_name.split('sign')[1].split('.')[0])
            png_labels.append(label)

    # png to training
    X_train = np.vstack([X_train, png_images])
    Y_train = np.hstack([Y_train, png_labels])

    # a-z in test and train folder
    def load_images_from_folders(base_path):
        images = []
        labels = []
        for label_folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, label_folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (28, 28)).flatten() / 255.0
                    images.append(img_resized)
                    labels.append(ord(label_folder) - ord('A'))
        return images, labels

    train_images, train_labels = load_images_from_folders('Dataset/train')
    test_images, test_labels = load_images_from_folders('Dataset/test')

    X_train = np.vstack([X_train, train_images])
    Y_train = np.hstack([Y_train, train_labels])

    X_test = np.vstack([X_test, test_images])
    Y_test = np.hstack([Y_test, test_labels])

    return X_train, Y_train, X_test, Y_test