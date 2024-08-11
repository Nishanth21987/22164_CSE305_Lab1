import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def load_data(filepath):
    """Loads data from an Excel file into a Pandas DataFrame."""
    return pd.read_excel(filepath)


def calculate_accuracy_for_k(X_train, y_train, X_test, y_test, k):
    """Trains a kNN classifier with a given k and returns the accuracy."""
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    accuracy = knn_classifier.score(X_test, y_test)
    return accuracy


def plot_accuracy_vs_k(k_values, accuracies):
    """Plots accuracy against the value of k."""
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k in kNN Classifier')
    plt.grid(True)
    plt.show()


def main():
    # Filepaths for the two datasets using raw strings to handle backslashes
    filepath_class1 = r"C:\Users\nagas\Downloads\English_Extractive_Embeddings_Fasttext.xlsx"
    filepath_class2 = r"C:\Users\nagas\Downloads\English_Abstractive_Embeddings_Fasttext.xlsx"

    # Load the datasets
    class1_data = load_data(filepath_class1)
    class2_data = load_data(filepath_class2)

    # Add a label column to distinguish between the two classes
    class1_data['label'] = 0  # Assuming 0 for class 1
    class2_data['label'] = 1  # Assuming 1 for class 2

    # Combine the two datasets
    combined_data = pd.concat([class1_data, class2_data], axis=0)

    # Separate features (X) and labels (y)
    X = combined_data.drop(columns=['label']).values  # Feature vectors
    y = combined_data['label'].values  # Class labels

    # Split the dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vary k from 1 to 11 and compute accuracy for each k
    k_values = list(range(1, 299))
    accuracies = []

    for k in k_values:
        accuracy = calculate_accuracy_for_k(X_train, y_train, X_test, y_test, k)
        accuracies.append(accuracy)
        print(f"Accuracy for k={k}: {accuracy:.2f}")

    # Plot the accuracy vs. k
    plot_accuracy_vs_k(k_values, accuracies)


if __name__ == "__main__":
    main()
