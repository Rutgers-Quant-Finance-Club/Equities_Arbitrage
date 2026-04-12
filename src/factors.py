import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import constants


def run_pca(
    raw_returns: np.ndarray, n_factors: int = None
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Standardize returns and extract PCA factors.

    Parameters
    ----------
    returns : shape (M, N)
        Raw returns for N stocks over M days
    n_factors : int
        Number of principal components to extract

    Returns
    -------
    factors_returns : np.ndarray, shape (M, n_factors)
        Return F (F_{ij} is factor i's returns on day j)
    weights : np.ndarray, shape (N, n_factors)
        Return Q (Q_{ij} is factor i's jth stock value)
    pca : fitted PCA object
        Kept so we can transform new data later
    """
    if n_factors is None:
        n_factors = constants.N_FACTORS

    # Step 1: Standardize returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(raw_returns)  # Shape (M, N)

    # Step 2: Run PCA
    pca = PCA(n_components=n_factors)
    pca.fit(scaled_returns)

    # Step 3: Scale components by volatility (eq. 8)
    returns_std = scaler.scale_

    # Step 4: Compute weights for each stock for each eigenvector
    weights = pca.components_.T / returns_std.reshape(-1, 1)  # Shape (N, n_factors)

    # Step 5: Compute Eigenportfolio Returns (eq. 9)
    factor_returns = raw_returns @ weights  # Shape (M, n_factors)

    return factor_returns, weights, pca


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
