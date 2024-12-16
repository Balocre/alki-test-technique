import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from darts.metrics.metrics import mape
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.model_selection import train_test_split

from .utils import build_sample_weights


def fit(
    model,
    series_group,
    epochs,
    missing_sample_weight=0.05,
    fill_missing_values=True,
    test_size=0.2,
):
    # encode static covariates (customer names) as int
    static_covariate_transformer = StaticCovariatesTransformer()
    series_group = static_covariate_transformer.fit_transform(series_group)

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
        epochs=epochs,
        sample_weight=sample_weight_train,
        val_sample_weight=sample_weight_val,
    )


def eval(model, series, series_val, n, num_samples):
    # encode static covariates (customer names) as int
    static_covariate_transformer = StaticCovariatesTransformer()

    series = static_covariate_transformer.fit_transform(series)
    series_val = static_covariate_transformer.fit_transform(series_val)

    pred_series = model.predict(
        n=n,
        num_samples=num_samples,
    )

    figsize = (16, 6)
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

    # plot actual series
    plt.figure(figsize=figsize)
    series_val.plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(series_val, pred_series)))
    plt.legend()
