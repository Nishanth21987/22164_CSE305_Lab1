import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance


def load_data(filepath):
    """Loads data from an Excel file into a Pandas DataFrame."""
    return pd.read_excel(filepath)


def calculate_minkowski_distance(vector1, vector2, r):
    """Calculates the Minkowski distance between two vectors for a given r."""
    return distance.minkowski(vector1, vector2, r)


def plot_minkowski_distances(distances, r_values):
    """Plots the Minkowski distances against r values."""
    plt.plot(r_values, distances, marker='o')
    plt.xlabel('r value')
    plt.ylabel('Minkowski Distance')
    plt.title('Minkowski Distance between Two Vectors for Different r Values')
    plt.grid(True)
    plt.show()


def main():
    # Filepaths for the two datasets using raw strings to handle backslashes
    filepath_class1 = r"C:\Users\nagas\Downloads\English_Extractive_Embeddings_Fasttext.xlsx"
    filepath_class2 = r"C:\Users\nagas\Downloads\English_Abstractive_Embeddings_Fasttext.xlsx"

    # Load the datasets
    class1_data = load_data(filepath_class1)
    class2_data = load_data(filepath_class2)

    # Select two feature vectors (columns) from the datasets
    # Assuming you want to compare the same feature between two datasets
    feature_name_class1 = class1_data.columns[0]  # For example, selecting the first column
    feature_name_class2 = class2_data.columns[0]  # Similarly, selecting the first column

    vector1 = class1_data[feature_name_class1].values
    vector2 = class2_data[feature_name_class2].values

    # Initialize lists to store distances and r values
    distances = []
    r_values = list(range(1, 11))  # r from 1 to 10

    # Calculate Minkowski distances for each r
    for r in r_values:
        dist = calculate_minkowski_distance(vector1, vector2, r)
        distances.append(dist)

    # Print calculated distances
    print(f"Minkowski distances for r from 1 to 10: {distances}")

    # Plot the results
    plot_minkowski_distances(distances, r_values)


if __name__ == "__main__":
    main()
