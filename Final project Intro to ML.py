"""1st doc for tests
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def workflow() -> None:
    # Get data.
    weather_data = pd.read_csv('weather_prediction_dataset.csv')

    # See histograms for distributions.
    # histograms(weather_data)

    # See missing values.
    missing_vals(weather_data)

    #


# visualize the missing values
def missing_vals(data:np.array=None) -> None:
    plt.figure(figsize=(10,6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.show()


def histograms(data:np.array=None) -> None:
    data.hist(bins=50, figsize=(20,15))
    plt.show()


if __name__ == "__main__":
    workflow()