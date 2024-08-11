import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_data(filepath):
    """Loads data from an Excel file into a Pandas DataFrame."""
    return pd.read_excel(filepath)


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

    # Initialize the kNN classifier with k=3 (as an example)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier
    knn_classifier.fit(X_train, y_train)

    # Predict the labels for both the training and test sets
    y_train_pred = knn_classifier.predict(X_train)
    y_test_pred = knn_classifier.predict(X_test)

    # Evaluate the confusion matrix and classification report for training data
    print("Confusion Matrix for Training Data:")
    print(confusion_matrix(y_train, y_train_pred))

    print("Classification Report for Training Data:")
    print(classification_report(y_train, y_train_pred))

    # Evaluate the confusion matrix and classification report for test data
    print("Confusion Matrix for Test Data:")
    print(confusion_matrix(y_test, y_test_pred))

    print("Classification Report for Test Data:")
    print(classification_report(y_test, y_test_pred))


if __name__ == "__main__":
    main()
