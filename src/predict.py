import pandas as pd
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from darts.models.forecasting.forecasting_model import GlobalForecastingModel


def infer(model, series, n, save_to_csv=True):
    # encode static covariates (customer names) as int
    static_covariate_transformer = StaticCovariatesTransformer()
    # series = static_covariate_transformer.fit_transform(series)
    static_covariate_transformer.fit(series)

    # need to fill missing values or model can't work
    transformer_filler = MissingValuesFiller()
    series = transformer_filler.transform(series)

    # can be trained on multiple timeseries at once
    if isinstance(model, GlobalForecastingModel):
        pass

    series_pred = model.predict(
        n=n,
        series=static_covariate_transformer.transform(series),
        verbose=True,
    )

    temp = list()
    for (
        s,
        sp,
    ) in zip(series, series_pred):
        series_df = sp.pd_dataframe()
        customer_name = s.static_covariates_values()[0][0]
        series_df["CUSTOMER"] = customer_name
        series_df["QUANTITY"] = series_df["QUANTITY"].astype(int)
        temp.append(series_df)

    test_df = pd.concat(temp)
    test_df.to_csv("test.csv", sep=";", index_label="DATE", date_format="%d/%m/%y")
