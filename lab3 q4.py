import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    # Print the sizes of the splits to verify
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Optionally: Inspect the first few rows of the training data
    print("First few rows of X_train:")
    print(X_train[:5])


if __name__ == "__main__":
    main()
