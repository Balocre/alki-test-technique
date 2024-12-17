import os

import hydra
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data import build_piped_flux_filter, get_df_from_influxdb
from src.predict import infer
from src.train import eval, fit

CLIENT_HOST = os.getenv("INFLUXDB_V2_URL")
CLIENT_ORG = os.getenv("INFLUXDB_V2_ORG")
CLIENT_TOKEN = os.getenv("INFLUXDB_V2_TOKEN")

BUCKET = "technical-test"


def _load_model_from_cfg(cfg: DictConfig):
    model_cls = hydra.utils.get_class(cfg.model._target_)
    # load model from checkpoint for finetuning
    model_name = cfg.checkpoint.model_name
    work_dir = cfg.checkpoint.work_dir
    file_name = cfg.checkpoint.file_name
    if model_name is not None:
        try:
            model = model_cls.load_from_checkpoint(model_name, work_dir, file_name)
        except ValueError:
            print("Using untrained model")
            model = instantiate(cfg.model)

    return model


def _build_flux_filter_from_cfg(cfg: DictConfig):
    flux_filters = list()
    for key, values in cfg.data.filters.builder_arguments.items():
        flux_filters.append(build_piped_flux_filter(key, values))

    return flux_filters


def train(cfg: DictConfig):
    model = _load_model_from_cfg(cfg)

    flux_filters = _build_flux_filter_from_cfg(cfg)

    data_df = get_df_from_influxdb(
        bucket=BUCKET,
        influxdb_host=CLIENT_HOST,
        influxdb_org=CLIENT_ORG,
        influxdb_token=CLIENT_TOKEN,
        flux_query_filters=flux_filters,
        start="-8y",
        stop="now()",
    )

    series = TimeSeries.from_group_dataframe(
        data_df,
        group_cols="CUSTOMER",
        value_cols="QUANTITY",
        fill_missing_dates=True,
        freq="D",
    )

    fit(model, series, epochs=cfg.train_parameters.epochs)


def test(cfg):
    model = _load_model_from_cfg(cfg)

    flux_filters = _build_flux_filter_from_cfg(cfg)

    data_df = get_df_from_influxdb(
        bucket=BUCKET,
        influxdb_host=CLIENT_HOST,
        influxdb_org=CLIENT_ORG,
        influxdb_token=CLIENT_TOKEN,
        flux_query_filters=flux_filters,
        start="-8y",
        stop="now()",
    )

    series = TimeSeries.from_group_dataframe(
        data_df,
        group_cols="CUSTOMER",
        value_cols="QUANTITY",
        fill_missing_dates=True,
        freq="D",
    )

    input_size = model.model_params.get("input_chunk_length", None)
    horizon = model.model_params.get("output_chunk_length", None)

    series, series_val = train_test_split(
        series,
        test_size=cfg.test_parameters.test_size,
        axis=1,
        input_size=input_size,
        horizon=horizon,
        vertical_split_type="model-aware",
    )

    n = cfg.test_parameters.n
    num_samples = cfg.test_parameters.num_samples
    eval(model, series, series_val, n, num_samples)


def predict(cfg: DictConfig):
    model = _load_model_from_cfg(cfg)

    flux_filters = _build_flux_filter_from_cfg(cfg)

    data_df = get_df_from_influxdb(
        bucket=BUCKET,
        influxdb_host=CLIENT_HOST,
        influxdb_org=CLIENT_ORG,
        influxdb_token=CLIENT_TOKEN,
        flux_query_filters=flux_filters,
        start="-8y",
        stop="now()",
    )

    series = TimeSeries.from_group_dataframe(
        data_df,
        group_cols="CUSTOMER",
        value_cols="QUANTITY",
        fill_missing_dates=True,
        freq="D",
    )

    infer(model, series, n=23)


@hydra.main(config_path="./conf/", config_name="run", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        print("Launching train")
        train(cfg)
    elif cfg.mode == "test":
        print("Testing model")
        test(cfg)
    elif cfg.mode == "predict":
        print("Predicting values")
        predict(cfg)


if __name__ == "__main__":
    main()
