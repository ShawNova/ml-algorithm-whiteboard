import numpy as np

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed=seed)
    idx = [x for x in range(y.shape[0])]
    np.random.shuffle(idx)
    return X[idx], y[idx]


if __name__ == '__main__':
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])
    y = np.array([1, 2, 3, 4])
    output = (np.array([[5, 6],
                    [1, 2],
                    [7, 8],
                    [3, 4]]),
             np.array([3, 1, 4, 2]))
    X_shuffle, y_shuffle = shuffle_data(X, y)


