from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller

from .utils import build_sample_weights


def fit(
    model,
    data_df,
    epochs,
    checkpoint_path=None,
    missing_sample_weight=0.05,
    fill_missing_values=True,
    training_cutoff=0.8,
):
    if checkpoint_path is not None:
        model.load(checkpoint_path)

    series_group = TimeSeries.from_group_dataframe(
        data_df,
        group_cols="CUSTOMER",
        value_cols="QUANTITY",
        fill_missing_dates=True,
        freq="D",
    )

    static_covariate_transformer = StaticCovariatesTransformer()
    for i, series in enumerate(series_group):
        series_group[i] = static_covariate_transformer.fit_transform(series)

    if fill_missing_values:
        transformer_filler = MissingValuesFiller()
        series_group = transformer_filler.transform(series_group)

    sample_weight_group = list()
    for series in series_group:
        sample_weight_group.append(build_sample_weights(series, missing_sample_weight))

    series_train = list()
    series_val = list()
    for series in series_group:
        t_train, t_val = series.split_after(training_cutoff)
        series_train.append(t_train)
        series_val.append(t_val)

    sample_weight_train = list()
    sample_weight_val = list()
    for sample_weight in sample_weight_group:
        t_train, t_val = sample_weight.split_after(training_cutoff)
        sample_weight_train.append(t_train)
        sample_weight_val.append(t_val)

    model.fit(
        series_train,
        val_series=series_val,
        verbose=True,
        sample_weight=sample_weight_train,
        val_sample_weight=sample_weight_val,
    )

    model.fit(series_group, epochs=epochs)
