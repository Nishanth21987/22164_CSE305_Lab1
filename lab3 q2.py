import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    """Loads data from an Excel file into a Pandas DataFrame."""
    return pd.read_excel(filepath)


def calculate_mean_variance(feature_data):
    """Calculates the mean and variance of the feature data."""
    mean_value = np.mean(feature_data)
    variance_value = np.var(feature_data)
    return mean_value, variance_value


def plot_histogram(feature_data, bins=10):
    """Plots a histogram of the feature data."""
    plt.hist(feature_data, bins=bins, edgecolor='black')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Feature')
    plt.show()


def main():
    # Filepaths for the two datasets using raw strings to handle backslashes
    filepath_class1 = r"C:\Users\nagas\Downloads\English_Extractive_Embeddings_Fasttext.xlsx"
    filepath_class2 = r"C:\Users\nagas\Downloads\English_Abstractive_Embeddings_Fasttext.xlsx"

    # Load the datasets
    class1_data = load_data(filepath_class1)
    class2_data = load_data(filepath_class2)

    # Choose a feature (column) from the first dataset, for example, the first numerical column
    feature_name = class1_data.columns[0]  # Selecting the first column as an example
    feature_data = class1_data[feature_name].values

    # Calculate mean and variance
    mean_value, variance_value = calculate_mean_variance(feature_data)

    # Print mean and variance
    print(f"Mean of the feature '{feature_name}':", mean_value)
    print(f"Variance of the feature '{feature_name}':", variance_value)

    # Plot histogram
    plot_histogram(feature_data, bins=10)


if __name__ == "__main__":
    main()