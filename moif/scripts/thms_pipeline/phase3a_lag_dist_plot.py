import pandas as pd
import matplotlib.pyplot as plt

REPORT_PATH = "phase3a_results/case_lag_correlation_report.csv"
df = pd.read_csv(REPORT_PATH)

# Histogram of best_lag
for feat in df['feat'].unique():
    for label in df['label'].unique():
        subset = df[(df['feat'] == feat) & (df['label'] == label)]
        plt.figure()
        plt.hist(subset['best_lag'], bins=25, range=(-60, 60))
        plt.title(f"Best Lag Distribution: {feat} vs {label}")
        plt.xlabel("Lag (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(f"phase3a_results/lag_hist_{feat}_{label}.png")
        print(f"Histogram saved for {feat}_{label}")

print("Lag distribution analysis complete.")
