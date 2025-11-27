# tools/plot_locust_results.py
# This script reads locust CSV 'locust_stats_history.csv' and plots median latency if present.
import pandas as pd
import matplotlib.pyplot as plt
def plot_csv(path):
    df = pd.read_csv(path)
    # user needs to adapt column names depending on locust version
    if 'Median Response Time' in df.columns:
        plt.plot(df['Timestamp'], df['Median Response Time'])
        plt.title("Median Response Time")
        plt.xlabel("Time")
        plt.ylabel("ms")
        plt.savefig("locust_median.png")
if __name__ == "__main__":
    print("Run plot_csv with the locust csv path")
