import numpy as np
from statsmodels.tsa.stattools import adfuller


def test_stationary(residuals: np.ndarray):
    X = np.cumsum(residuals, axis=0)

    n_rejected = 0
    total = X.shape[1]
    p_values = np.zeros(total)

    for i in range(total):
        result = adfuller(
            X[:, i],
            autolag='AIC',
            regression="c",
        )

        p_values[i] = result[1]
        if p_values[i] < 0.05:
            n_rejected += 1

    print(f"Stocks tested:    {total}")
    print(f"Stationary (p<0.05): {n_rejected} ({n_rejected / total:.1%})")
    print(
        f"Non-stationary:      {total - n_rejected} ({(total - n_rejected) / total:.1%})"
    )

    return p_values
