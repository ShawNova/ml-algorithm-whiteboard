import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	x, y = np.array(X), np.array(y)
	# Ax + b = y
	# min  (Ax - y)
	A = np.linalg.inv(x.T @ x) @ x.T @ y
	# b = A * x - y
	theta = np.round(A, 4).tolist()
	return theta

if __name__ == '__main__':
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    output = [0.0, 1.0]
    pred = linear_regression_normal_equation(X, y)
    assert pred == output