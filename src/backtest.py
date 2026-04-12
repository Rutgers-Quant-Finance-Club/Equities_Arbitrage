import numpy as np
import pandas as pd

from src import constants

from .data import adjust_for_volume
from .factors import fit_factors, run_pca
from .signals import compute_s_scores, update_positions


def run_backtest(
    returns: pd.DataFrame,
    volume: pd.DataFrame = None,
    debug_interval: int = None,
) -> pd.DataFrame:
    """
    Run the back test.

    Walks forward daily, estimating signals from a trailing 60-day window
    and updating positions according to the trading rules (eq. 16).

    Parameters
    ----------
    returns : shape (T, N)
        Daily stock returns for the full period.
    volume : shape (T, N), optional
        Daily share volume. If provided, returns are adjusted to trading time (eq. 20).

    Returns
    -------
    pd.DataFrame
        Daily backtest results: equity, PNL, costs, position counts.
    """
    T, N = returns.shape
    equity = 100.0
    positions = np.zeros(N)
    if volume is not None:
        adj_returns = adjust_for_volume(returns.values, volume.values)
    else:
        adj_returns = returns.values
    results = []

    for t in range(constants.PCA_WINDOW, T):
        cur_returns = returns.values[t]
        pnl = positions @ cur_returns

        pca_window = adj_returns[t - constants.PCA_WINDOW : t]
        factor_returns, _, _ = run_pca(pca_window)

        returns_window = pca_window[-constants.ESTIMATION_WINDOW :]
        factor_returns_window = factor_returns[-constants.ESTIMATION_WINDOW :]

        _, residuals = fit_factors(
            returns=returns_window,
            factor_returns=factor_returns_window,
        )

        s_scores = compute_s_scores(residuals)

        new_positions = update_positions(s_scores, positions, equity)

        trades = new_positions - positions
        slippage_cost = constants.SLIPPAGE_PER_TRADE * np.abs(trades).sum()

        net = pnl - slippage_cost

        daily_returns = net / equity
        equity += net

        if debug_interval and (t - constants.ESTIMATION_WINDOW) % debug_interval == 0:
            valid = ~np.isnan(s_scores)
            print(f"\n--- Day {t}: {returns.index[t].date()} ---")
            print(f"Equity:        {equity:.2f}")
            print(f"Valid stocks:  {valid.sum()}")
            print(
                f"S-score range: {np.nanmin(s_scores):.2f} to {np.nanmax(s_scores):.2f}"
            )
            print(
                f"|s| > 1.25:    {(np.abs(s_scores[valid]) > constants.OPEN_THRESHOLD).sum()}"
            )
            print(f"Longs:         {(new_positions > 0).sum()}")
            print(f"Shorts:        {(new_positions < 0).sum()}")
            print(f"Trades today:  {(trades != 0).sum()}")
            print(f"Trade cost:    {slippage_cost:.4f}")
            print(f"Daily PNL:     {pnl:.4f}")

        positions = new_positions

        results.append(
            {
                "date": returns.index[t],
                "equity": equity,
                "daily_pnl": pnl,
                "slippage_cost": slippage_cost,
                "net": net,
                "n_long": (positions > 0).sum(),
                "n_short": (positions < 0).sum(),
                "daily_returns": daily_returns,
            }
        )

    return pd.DataFrame(results).set_index("date")
