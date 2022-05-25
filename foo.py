import numpy as np


class PolynomialRegression:
    def __init__(self, degree, epoch, lr=0.01):
        self.degree = degree
        self.epoch = epoch
        self.lr = lr

    def transform(self, X):
        # initialize X_transform
        X_transformed = np.hstack((np.ones((X.shape[0], 1)), X))
        for i in range(1, self.degree):
            X_transformed = np.hstack((X_transformed, np.power(X, i + 1)))
        return X_transformed

    def normalize(self, X):
        return (X - X.min(axis=0, keepdims=True)) / X.max(axis=0, keepdims=True)

    def fit(self, X, y):
        y = y.reshape(-1, 1)

        # initialize weights
        X_normalized = self.normalize(X)
        X_transformed = self.transform(X_normalized)

        # solve normal equation
        if np.linalg.det(X_transformed.T @ X_transformed) != 0:
            self.W = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ y
            print("solve analytically")

        # solve with gradient descent
        else:
            self.W = np.random.rand(X_transformed.shape[1], 1)
            for _ in range(self.epoch):
                y_hat = self.predict(X_transformed, is_train=True)
                error = y_hat - y
                self.W -= self.lr * np.dot(X_transformed.T, error)
            print("solve numerically")

    def predict(self, X, is_train=False):
        if not is_train:
            X_normalized = self.normalize(X)
            X = self.transform(X_normalized)
        return np.dot(X, self.W)


def RMSE(predict, gt):
    return np.log(np.sum(np.power(predict - gt, 2)) / gt.shape[0])


# class PolynomialRegression():
#     def __init__(self, degree, epoch, lr=0.01):
#         self.degree = degree
#         self.epoch = epoch
#         self.lr = lr

#     def transform(self, X):
#         # initialize X_transform
#         X_transformed = np.hstack((np.ones((X.shape[0], 1)), X))
#         for i in range(1, self.degree):
#             X_transformed = np.hstack((X_transformed, np.power(X, i+1)))
#         return X_transformed

#     def normalize(self, X):
#         return (X - X.min(axis=0, keepdims=True)) / X.max(axis=0, keepdims=True)

#     def fit(self, X, y):
#         y = y.reshape(-1, 1)

#         # initialize weights
#         X_transformed = self.transform(X)
#         X_normalized = self.normalize(X_transformed)

#         # solve normal equation
#         if np.linalg.det(X_normalized.T @ X_normalized) != 0:
#             self.W = np.linalg.inv(X_normalized.T @ X_normalized) @ X_normalized.T @ y
#             print("solve analytically")

#         # solve with gradient descent
#         else:
#             self.W = np.random.rand(X_transformed.shape[1], 1)
#             for i in range(self.epoch):
#                 y_hat = self.predict(X_transformed, is_train=True)
#                 error = y_hat - y
#                 self.W -= self.lr * np.dot(X_normalized.T, error)
#             print("solve numerically")

#     def predict(self, X, is_train=False):
#         if not is_train:
#             X = self.transform(X)
#         X_normalized = self.normalize(X)
#         return np.dot(X_normalized, self.W)

# def RMSE(predict, gt):
#     return np.log(np.sum(np.power(predict - gt, 2)) / gt.shape[0])
