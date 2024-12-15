import darts
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from influxdb_client import InfluxDBClient


def build_piped_flux_filter(key: str, values: str | list[str]):
    flux = " |> filter(fn: (r) => "

    flux += f'r["{key}"] == "{values.pop(0)}" '
    for v in values:
        flux += f'or r["{key}"] == "{v}" '
    flux += ")"

    return flux


def build_piped_flux_range(start, stop=None):
    flux = f" |> range(start: {start} "
    if stop is not None:
        flux += f", stop: {stop} "
    flux += ")"

    return flux


def fill_series_missing_dates(series: darts.TimeSeries):
    transformer_filler = MissingValuesFiller()

    return transformer_filler.transform(series)


def split_series(series: darts.TimeSeries, cutoff: float):
    series_train, series_val = series.split_after(cutoff)

    return series_train, series_val


def get_df_from_influxdb(
    bucket,
    influxdb_host,
    influxdb_token,
    influxdb_org,
    flux_query_filters,
    start=None,
    stop=None,
):
    with InfluxDBClient(influxdb_host, influxdb_token, org=influxdb_org) as client:
        flux = f'from(bucket: "{bucket}")' + build_piped_flux_range(start, stop)

        for filter_ in flux_query_filters:
            flux += filter_

        flux += (
            ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        )

        query_api = client.query_api()
        df = query_api.query_data_frame(flux, data_frame_index="_time")

    return df
