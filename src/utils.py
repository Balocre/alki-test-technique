import numpy as np
import pandas as pd


def get_missing_dates_in_df_dt_index(dataframe):
    date_range = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max())
    missing_dates = date_range[~date_range.isin(dataframe.index)]

    return missing_dates


def build_sample_weights(series, weight=0.05):
    series_na_mask = series.pd_dataframe().isna()

    sample_weight = np.ones(series.shape)
    sample_weight[series_na_mask, 0] = weight
    sample_weight = series.with_values(sample_weight)

    return sample_weight
