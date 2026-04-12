# PCA eigenportfolios retained as risk factors (Section 2.1)
N_FACTORS = 5

# Trailing window for average volume in trading-time adjustment (Section 6)
VOLUME_WINDOW = 10

# Trailing window for correlation matrix / PCA estimation (Section 2.1)
PCA_WINDOW = 252

# Trailing window for factor regression and OU estimation (Section 4)
ESTIMATION_WINDOW = 75

# S-score thresholds for entry and exit (eq. 16)
OPEN_THRESHOLD = 1.25
CLOSE_LONG_THRESHOLD = 0.65
CLOSE_SHORT_THRESHOLD = 0.8

# Slippage per side, 2 bps round trip (Section 5)
SLIPPAGE_PER_TRADE = 0.0001

# Dollars long and short per dollar of equity (Section 5)
LEVERAGE = 2
