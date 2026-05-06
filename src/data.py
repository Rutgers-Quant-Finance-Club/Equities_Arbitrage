import numpy as np
import pandas as pd

from src import constants


def load_data():

    stock_returns = pd.read_csv(
        "../data/raw/stock_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    stock_volume = pd.read_csv(
        "../data/raw/stock_volume.csv",
        index_col=0,
        parse_dates=True,
    )
    stock_meta = pd.read_csv(
        "../data/raw/stock_meta.csv",
        index_col=0,
    )
    spy_returns = pd.read_csv(
        "../data/raw/spy_returns.csv",
        index_col=0,
        parse_dates=True,
    ).squeeze()
    return stock_returns, stock_volume, stock_meta, spy_returns


def adjust_for_volume(
    returns: np.ndarray,
    volume: np.ndarray,
    window: int = constants.VOLUME_WINDOW,
) -> np.ndarray:
    """
    Convert returns from calendar time to trading time (eq. 20).

    Scales returns inversely to relative daily volume. Low-volume moves
    get amplified, high-volume moves get dampened.

    Parameters
    ----------
    returns : shape (T, N)
        Daily stock returns.
    volume : shape (T, N)
        Daily share volume.
    window : int
        Trailing window for average volume. Paper uses 10 days.

    Returns
    -------
    np.ndarray, shape (T, N)
        Volume-adjusted returns.
    """
    avg_volume = pd.DataFrame(volume).rolling(window).mean().values
    adj_returns = returns * (avg_volume / volume)

    return adj_returns
