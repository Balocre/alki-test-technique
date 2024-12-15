import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data import build_piped_flux_filter, get_df_from_influxdb
from src.train import fit

CLIENT_HOST = os.getenv("INFLUXDB_V2_URL")
CLIENT_ORG = os.getenv("INFLUXDB_V2_ORG")
CLIENT_TOKEN = os.getenv("INFLUXDB_V2_TOKEN")

BUCKET = "technical-test"


def train(cfg: DictConfig):
    model = instantiate(cfg.model)

    flux_query_filters = list()
    for key, values in cfg.data.filters.items():
        flux_query_filters.append(build_piped_flux_filter(key, values))

    data_df = get_df_from_influxdb(
        bucket=BUCKET,
        influxdb_host=CLIENT_HOST,
        influxdb_org=CLIENT_ORG,
        influxdb_token=CLIENT_TOKEN,
        flux_query_filters=flux_query_filters,
        start="-8y",
        stop="now()",
    )

    fit(model, data_df, epochs=cfg.train.epochs)


@hydra.main(config_path="./conf/", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
