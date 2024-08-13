import numpy as np
import pandas as pd


def load_data(filepath):
    # Load the CSV file and select only columns 0 and 1
    return pd.read_csv(filepath).iloc[:, [0, 1]]


def calculate_centroid(class_data):
    return np.mean(class_data, axis=0)


def calculate_spread(class_data):
    return np.std(class_data, axis=0)


def calculate_interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)


def main():
    filepath = "C:/Users/NISHANTH/Downloads/DCT_withoutduplicate 4.csv"

    class1_data = load_data(filepath)
    class2_data = load_data(filepath)

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
