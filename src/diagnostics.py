import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def print_diagnostics(results: pd.DataFrame):
    total_days = len(results)
    avg_factors = results["n_factors"].mean()
    total_return = results["equity"].iloc[-1] / results["equity"].iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / total_days) - 1
    annual_vol = results["daily_returns"].std() * np.sqrt(252)
    avg_beta = results["portfolio_beta"].mean()
    sharpe = annual_return / annual_vol
    max_dd = (results["equity"] / results["equity"].cummax() - 1).min()
    total_cost = results["slippage_cost"].sum()
    total_gross_pnl = results["daily_pnl"].sum()
    avg_long = results["n_long"].mean()
    avg_short = results["n_short"].mean()
    cost_drag = total_cost / total_gross_pnl if total_gross_pnl != 0 else np.inf

    print(f"Period:          {results.index[0]} to {results.index[-1]}")
    print(f"Total return:    {total_return:.2%}")
    print(f"Annual return:   {annual_return:.2%}")
    print(f"Annual vol:      {annual_vol:.2%}")
    print(f"Sharpe ratio:    {sharpe:.2f}")
    print(f"Avg portfolio beta: {avg_beta:.4f}")
    print(f"Max drawdown:    {max_dd:.2%}")
    print(f"Avg longs:       {avg_long:.0f}")
    print(f"Avg shorts:      {avg_short:.0f}")
    print(f"Gross PNL:       {total_gross_pnl:.2f}")
    print(f"Total costs:     {total_cost:.2f}")
    print(f"Cost drag:       {cost_drag:.2%}")
    print(f"Avg factors:     {avg_factors:.1f}")


def graph_weights(weights: np.ndarray, tickers: np.ndarray, factor_num: int):
    weight = weights[:, factor_num]
    df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weight,
        }
    )
    df = df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
    if (df["Weight"] < 0).any():
        zero_idx = df["Weight"].abs().idxmin()
        df = df.iloc[zero_idx - 20 : zero_idx + 20]
    else:
        df = df.tail(40)

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#e6ecf5"})
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=90, fontsize=8)
    colors = ["#2ca02c" if w > 0 else "#d62728" for w in df["Weight"]]

    ax = sns.barplot(
        data=df, x="Ticker", y="Weight", palette=colors, hue="Ticker", legend=False
    )
    ax.set_title(f"Factor Weights: {factor_num}")
    plt.show()


def top_factors(var_ratio: np.ndarray):
    df = pd.DataFrame(
        {
            "Principal Component": np.arange(1, len(var_ratio) + 1),
            "Explained Variance": var_ratio,
        }
    )

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#e6ecf5"})
    plt.figure(figsize=(8, 4))

    ax = sns.barplot(
        data=df, x="Principal Component", y="Explained Variance", color="darkblue"
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title(f"Top {len(var_ratio)} Eigenvalues (Explained Variances)")

    plt.show()
