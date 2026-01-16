import numpy as np

def get_data():
    data_path = "data_part_B.csv"
    data = np.genfromtxt(fname = data_path, delimiter = ",")
    X_noisy = data[:, 0]
    Y = data[:, 1]
    Delta = data[:, 2]

    return X_noisy, Y, Delta