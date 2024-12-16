from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.model_selection import train_test_split

from .utils import build_sample_weights


def fit(
    model,
    data_df,
    epochs,
    missing_sample_weight=0.05,
    fill_missing_values=True,
    test_size=0.2,
):
    series_group = TimeSeries.from_group_dataframe(
        data_df,
        group_cols="CUSTOMER",
        value_cols="QUANTITY",
        fill_missing_dates=True,
        freq="D",
    )

    # encode static covariates (customer names) as int
    static_covariate_transformer = StaticCovariatesTransformer()
    for i, series in enumerate(series_group):
        series_group[i] = static_covariate_transformer.fit_transform(series)

    # builds sample weights
    sample_weight_group = list()
    for series in series_group:
        sample_weight_group.append(build_sample_weights(series, missing_sample_weight))

    # fill missing values as averaged values
    if fill_missing_values:
        transformer_filler = MissingValuesFiller()
        series_group = transformer_filler.transform(series_group)

    input_size = model.model_params.get("input_chunk_length", None)
    horizon = model.model_params.get("output_chunk_length", None)

    series_train, series_val = train_test_split(
        series_group,
        test_size=test_size,
        axis=1,
        input_size=input_size,
        horizon=horizon,
        vertical_split_type="model-aware",
    )

    sample_weight_train, sample_weight_val = train_test_split(
        sample_weight_group,
        test_size=test_size,
        axis=1,
        input_size=input_size,
        horizon=horizon,
        vertical_split_type="model-aware",
    )

    # can be trained on multiple timeseries at once
    if isinstance(model, GlobalForecastingModel):
        pass

    model.fit(
        series_train,
        val_series=series_val,
        verbose=True,
        sample_weight=sample_weight_train,
        val_sample_weight=sample_weight_val,
    )

    model.fit(series_group, epochs=epochs)
