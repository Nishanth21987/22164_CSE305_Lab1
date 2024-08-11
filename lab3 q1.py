import numpy as np
import pandas as pd


def load_data(filepath):
    return pd.read_excel(filepath)


def calculate_centroid(class_data):
    return np.mean(class_data, axis=0)


def calculate_spread(class_data):
    return np.std(class_data, axis=0)


def calculate_interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)


def main():
    class1_data = load_data("C:/Users/nagas/Downloads/English_Extractive_Embeddings_Fasttext.xlsx")
    class2_data = load_data("C:/Users/nagas/Downloads/English_Abstractive_Embeddings_Fasttext.xlsx")

    centroid1 = calculate_centroid(class1_data)
    centroid2 = calculate_centroid(class2_data)

    spread1 = calculate_spread(class1_data)
    spread2 = calculate_spread(class2_data)

    interclass_distance = calculate_interclass_distance(centroid1, centroid2)

    print("Centroid of Class 1:", centroid1)
    print("Centroid of Class 2:", centroid2)
    print("Spread of Class 1:", spread1)
    print("Spread of Class 2:", spread2)
    print("Interclass Distance between Class 1 and Class 2:", interclass_distance)


if __name__ == "__main__":
    main()