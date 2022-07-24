import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from numpy import ndarray
from scipy.special import digamma


def gauss(X: ndarray, M: ndarray, Sigma: ndarray) -> ndarray:
    """gaussian distribution

    Args:
        X (ndarray): input data (N x D)
        Mu (ndarray): mean vectors (K x D)
        Sigma (ndarray): (K x D x D)

    Returns:
        ndarray: N x K
    """
    X_center = X[:, None, :, None] - M[None, :, :, None]  # (N x K x D x 1)
    X_center_T = X_center.transpose(0, 1, 3, 2)
    dim = X.shape[1]
    coeff = np.sqrt(2 * np.pi) ** dim * np.sqrt(LA.det(Sigma))
    return np.exp(-0.5 * X_center_T @ LA.inv(Sigma) @ X_center).squeeze() / coeff


def VB_E(X: ndarray, alpha: ndarray, beta: ndarray, nu: ndarray, W: ndarray, M: ndarray, dim: int) -> ndarray:
    """calculate burden

    Args:
        X (ndarray): N x D
        alpha (ndarray): 1 x K
        beta (ndarray): 1 x K
        nu (ndarray): 1 x K
        W (ndarray): N x K x D
        M (ndarray): K x D
        dim (int): dimension

    Returns:
        N x K
    """
    ln_ratio = digamma(alpha) - digamma(alpha.sum())
    ln_Sigma = np.array([digamma((nu + 1 - i) / 2) for i in range(dim)]).sum(axis=0) + dim * np.log(2) + np.log(LA.det(W))

    X_centered = X[:, None, :, None] - M[None, :, :, None]  # (N x K x D x 1)
    X_centered_T = X_centered.transpose(0, 1, 3, 2)

    ln_rho = ln_ratio + 0.5 * ln_Sigma - dim / (2 * beta) - nu / 2 * (X_centered_T @ W[None] @ X_centered).squeeze()  # N x K
    rho = np.exp(ln_rho)
    return rho / rho.sum(axis=1, keepdims=True)  # N x K


def VB_M(
    Burden: ndarray, X: ndarray, alpha_0: float, beta_0: float, nu_0: float, m_0: ndarray, W_0: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """M step (update parameters)

    Args:
        Burden (ndarray): N x K
        X (ndarray): N x D
        alpha_0 (float):
        beta_0 (float):
        nu_0 (float):
        m_0 (ndarray): 1 x D
        W_0 (ndarray): D x D

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
    """
    N_k = Burden.sum(axis=0)  # 1 x K

    X_weighted = np.sum(Burden[:, :, None] * X[:, None, :], axis=0) / N_k[:, None]  # N x K x D -> K x D
    X_weighted_centered = X[:, None, :, None] - X_weighted[None, :, :, None]  # (N x K x D x 1)
    X_weighted_centered_T = X_weighted_centered.transpose(0, 1, 3, 2)
    S = np.sum(Burden[:, :, None, None] * (X_weighted_centered @ X_weighted_centered_T), axis=0) / N_k[:, None, None]  # K x D x D

    alpha = alpha_0 + N_k
    beta = beta_0 + N_k
    nu = nu_0 + N_k
    M = (beta_0 * m_0 + N_k[:, None] * X_weighted) / beta[:, None]

    X_weighted_m0 = (X_weighted - m_0)[:, :, None]
    X_weighted_m0_T = X_weighted_m0.transpose(0, 2, 1)
    W_inv = LA.inv(W_0) + N_k[:, None, None] * S + (beta_0 * N_k[:, None] / (beta_0 + N_k[:, None]))[..., None] * X_weighted_m0 @ X_weighted_m0_T
    W = LA.inv(W_inv)

    return alpha, beta, nu, M, W


def main(x_path: Path, z_path: Path, params_path: Path, fig_path: Path | None, K: int = 4) -> None:
    data = np.loadtxt(x_path, delimiter=",")
    dim = data.shape[1]

    # initialize params
    alpha_0 = 0.1
    beta_0 = 1.0
    m_0 = np.random.randn(dim)
    nu_0 = dim
    W_0 = np.eye(dim)

    alpha = alpha_0 * np.ones(K)
    beta = beta_0 * np.ones(K)
    M = np.random.randn(K, dim)
    nu = dim * np.ones(K)
    W = np.tile(W_0, (K, 1, 1))
    ratio = np.ones(K) / K

    th = 0.1
    max_loop = 50

    log_likelihood_prev = 0
    log_likelihood = -1
    cnt = 0
    while np.abs(log_likelihood_prev - log_likelihood) > th and cnt < max_loop:
        log_likelihood_prev = log_likelihood
        cnt += 1

        Burden = VB_E(data, alpha, beta, nu, W, M, dim)
        alpha, beta, nu, M, W = VB_M(Burden, data, alpha_0, beta_0, nu_0, m_0, W_0)

        ratio = alpha / np.sum(alpha, keepdims=True)
        Sigma = nu[:, None, None] * W
        gmm = ratio * gauss(data, M, Sigma)

        log_likelihood = np.log(gmm.sum(axis=1)).sum()
        print(f"log likelihood: {log_likelihood} ({cnt} loop)")

    np.savetxt(z_path, Burden, delimiter=",")
    params = {"ratio": ratio, "M": M, "Sigma": Sigma}
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
