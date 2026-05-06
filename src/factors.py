import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    raw_returns: np.ndarray,
    max_factors: int = 25,
    target_var=0.55,
) -> tuple[np.ndarray, np.ndarray, PCA, int]:
    """
    Standardize returns and extract PCA factors with dynamic factor selection.

    Fits PCA with max_factors components, then selects the minimum number
    needed to explain target_var of total variance (Section 5.4).

    Parameters
    ----------
    raw_returns : shape (M, N)
        Raw simple returns for N stocks over M days.
    max_factors : int
        Maximum number of principal components to fit.
    target_var : float
        Target cumulative explained variance ratio (e.g. 0.55 = 55%).

    Returns
    -------
    factor_returns : np.ndarray, shape (M, n_factors)
        Eigenportfolio returns (eq. 9).
    weights : np.ndarray, shape (N, n_factors)
        Eigenportfolio weights per stock.
    pca : PCA
        Fitted PCA object for diagnostics.
    n_factors : int
        Number of factors selected to meet target_var.
    """
    # Step 1: Standardize returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(raw_returns)  # Shape (M, N)

    # Step 2: Run PCA
    pca = PCA(n_components=max_factors)
    pca.fit(scaled_returns)

    # Step 3: Choose # of factors to model target variance
    total_var = np.cumsum(pca.explained_variance_ratio_)
    n_factors = np.searchsorted(total_var, target_var) + 1

    # Step 4: Scale components by volatility (eq. 8)
    returns_std = scaler.scale_

    # Step 5: Compute weights for each stock for each eigenvector
    weights = pca.components_[:n_factors].T / returns_std.reshape(
        -1, 1
    )  # Shape (N, n_factors)

    # Step 6: Compute Eigenportfolio Returns (eq. 9)
    factor_returns = raw_returns @ weights  # Shape (M, n_factors)

    return factor_returns, weights, pca, n_factors


def fit_factors(
    returns: np.ndarray, factor_returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regress each stock's returns on the eigenportfolio factors.
    Extracts β loadings, α drift(dropped), and idiosyncratic residuals.

    This implements eq. 10:
        dS_i/S_i = α_i dt + Σ β_ij dI_j/I_j + dX_i
    """
    M = returns.shape[0]

    F = np.column_stack([np.ones(M), factor_returns])

    coefs, _, _, _ = np.linalg.lstsq(F, returns, rcond=None)

    factor_loadings = coefs[1:, :].T  # (N, n_factors)
    residuals = returns - (F @ coefs)  # (M, N)

    return factor_loadings, residuals
