import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from numpy import ndarray


def gauss(X: ndarray, Mu: ndarray, Sigma: ndarray) -> ndarray:
    """gaussian distribution

    Args:
        X (ndarray): input data (N x D)
        Mu (ndarray): mean vectors (K x D)
        Sigma (ndarray): (K x D x D)

    Returns:
        ndarray: N x K
    """
    X_centered = X[:, None, :, None] - Mu[None, :, :, None]  # (N x K x D x 1)
    X_centered_T = X_centered.transpose(0, 1, 3, 2)
    dim = X.shape[1]
    coeff = np.sqrt(2 * np.pi) ** dim * np.sqrt(LA.det(Sigma))
    return np.exp(-0.5 * X_centered_T @ LA.inv(Sigma) @ X_centered).squeeze() / coeff


def E(X: ndarray, ratio: ndarray, Mu: ndarray, Sigma: ndarray) -> tuple[ndarray, ndarray]:
    """calculate burden

    Args:
        X (ndarray): input data (N x D)
        ratio (ndarray): mixing ratio (K)
        Mu (ndarray): mean vectors (K x D)
        Sigma (ndarray): (K x (D x D))

    Returns:
        gmm: N x K
        burden: N x K
    """
    dist = gauss(X, Mu, Sigma)
    gmm = ratio * dist  # probs
    return gmm, gmm / np.sum(gmm, axis=1)[:, None]


def M(Burden: ndarray, X: ndarray, Mu: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """M step

    Args:
        Burden (ndarray): N x K
        X (ndarray): N x D
        Mu (ndarray): K x D

    Returns:
        ratio: K, Mu: K x D, Sigma: K x D x D
    """
    # Sk_1 = Burden.sum(axis=0)
    # Sk_x = np.sum(Burden[:, :, None] * X[:, None, :], axis=0)  # N x K x D -> K x D
    X_centered = X[:, None, :, None] - Mu[None, :, :, None]  # (N x K x D x 1)
    X_centered_T = X_centered.transpose(0, 1, 3, 2)

    N_k = Burden.sum(axis=0)  # 1 x K
    ratio = N_k / N_k.sum()
    Mu = np.sum(Burden[:, :, None] * X[:, None, :], axis=0) / N_k[:, None]  # K x D
    Sigma = np.sum(Burden[:, :, None, None] * (X_centered @ X_centered_T), axis=0) / N_k[:, None, None]  # K x D x D
    return ratio, Mu, Sigma


def main(x_path: Path, z_path: Path, params_path: Path, fig_path: Path | None, K: int = 4) -> None:
    data = np.loadtxt(x_path, delimiter=",")
    dim = data.shape[1]

    # initialize params
    Mu = np.random.randn(K, dim)
    Sigma = np.tile(np.eye(dim), (K, 1, 1))
    ratio = np.ones(K) / K

    th = 0.1
    max_loop = 50

    log_likelihood_prev = 0
    log_likelihood = -1
    cnt = 0
    while np.abs(log_likelihood_prev - log_likelihood) > th and cnt < max_loop:
        log_likelihood_prev = log_likelihood
        cnt += 1

        gmm, Burden = E(data, ratio, Mu, Sigma)
        ratio, Mu, Sigma = M(Burden, data, Mu)
        log_likelihood = np.log(gmm.sum(axis=1)).sum()
        print(f"log likelihood: {log_likelihood} ({cnt} loop)")
    np.savetxt(z_path, Burden, delimiter=",")
    params = {"ratio": ratio, "Mu": Mu, "Sigma": Sigma}
    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    if fig_path is not None:
        visualize(data, Burden, fig_path, log_likelihood, K)


def visualize(X: ndarray, Burden: ndarray, fig_path: Path, log_likelihood: float, K: int):
    labels = np.argmax(Burden, axis=1)
    cmap = plt.get_cmap("tab10")
    fig = plt.figure()
    ax = Axes3D(fig)
    N = X.shape[0]
    for n in range(N):
        ax.plot([X[n][0]], [X[n][1]], [X[n][2]], "o", color=cmap(labels[n]), ms=1.5)
    ax.set_title(f"K = {K}, log likelihood = {log_likelihood:.1f}", size=20, y=1)
    ax.view_init(elev=30, azim=45)
    fig.savefig(fig_path, format="png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":

    K = 4
    x_path = Path(f"{sys.argv[1]}")
    z_path = Path(f"{sys.argv[2]}")
    params_path = Path(f"{sys.argv[3]}")
    main(x_path, z_path, params_path, None, K)
