import numpy as np

from src import constants


def compute_s_scores(residuals: np.ndarray) -> np.ndarray:
    """
    Compute centred s-scores for a universe of stocks.

    Takes idiosyncratic residuals from a 60-day factor regression,
    fits AR(1) to the cumulative residual process for each stock,
    and returns the cross-sectionally centred s-score (eq. 15, 18, A2).

    Parameters
    ----------
    residuals : shape (60, N)
        Idiosyncratic daily return residuals from factor regression.

    Returns
    -------
    s_scores : np.ndarray, shape (N,)
        Centred s-scores. NaN for stocks rejected by the mean-reversion filter.
    """
    X = np.cumsum(residuals, axis=0)  # (60, N)

    X_cur = X[1:]  # (59, N)
    X_lag = X[:-1]  # (59, N)

    X_cur_avg = X_cur.mean(axis=0)  # (N,)
    X_lag_avg = X_lag.mean(axis=0)  # (N,)

    var = (X_lag**2).mean(axis=0) - X_lag_avg**2  # (N,)
    co_var = (X_lag * X_cur).mean(axis=0) - X_cur_avg * X_lag_avg  # (N,)

    # Compute slope, intercept with OLS formula
    slope = co_var / var  # (N,)
    slope = np.clip(slope, -10, 0.9999)
    intercept = X_cur_avg - slope * X_lag_avg  # (N,)

    # Compute equilibrium (m)
    equilibriums = intercept / (1 - slope)  # (N,)

    # Residuals from AR(1) / OU
    ar_resids = X_cur - (intercept + slope * X_lag)  # (59, N)
    var_ar_resids = np.var(ar_resids, axis=0, ddof=1)  # (N,)

    # Standard Deviation of Equilibriums
    equilibrium_std = np.sqrt(var_ar_resids / (1 - slope**2))  # (N,)

    # slope <= 0 (Random Walk/ Explosive), kappa > 8.4 == slope < 0.967
    invalid = (slope <= 0) | (slope >= constants.MAX_SLOPE)
    equilibriums[invalid] = np.nan
    equilibrium_std[invalid] = np.nan

    # Center equilibriums to remove systematic estimation bias
    equilibriums_centered = equilibriums - np.nanmean(equilibriums)  # (N,)
    s_scores = -equilibriums_centered / equilibrium_std  # (N,)

    return s_scores


def update_positions(
    s_scores: np.ndarray, valid: np.ndarray, cur_pos: np.ndarray, equity: float
) -> np.ndarray:
    """
    Determine tomorrow's positions from today's s-scores and current holdings.

    Implements the trading rules (eq. 16):
    - Open long/short when |s| > 1.25
    - Close long when s > -0.50
    - Close short when s < 0.75
    - Close any position if the stock was rejected (NaN s-score)

    Parameters
    ----------
    s_scores : shape (N,)
        Centred s-scores. NaN for rejected stocks.
    cur_pos : shape (N,)
        Current dollar positions per stock.
    equity : float
        Current portfolio equity.

    Returns
    -------
    np.ndarray, shape (N,)
        Updated positions for next trade day.
    """
    N = len(s_scores)
    updated_pos = np.zeros(N)

    allocation = (equity * constants.LEVERAGE) / N

    flat = cur_pos == 0
    long = cur_pos > 0
    short = cur_pos < 0

    # Open Long Position
    updated_pos[flat & valid & (s_scores < -constants.OPEN_THRESHOLD)] = allocation
    # Open Short Position
    updated_pos[flat & valid & (s_scores > constants.OPEN_THRESHOLD)] = -allocation

    # Hold Long Position
    hold_long = long & valid & (s_scores <= -constants.CLOSE_LONG_THRESHOLD)
    updated_pos[hold_long] = cur_pos[hold_long]
    # Hold Short Position
    hold_short = short & valid & (s_scores >= constants.CLOSE_SHORT_THRESHOLD)
    updated_pos[hold_short] = cur_pos[hold_short]

    return updated_pos
