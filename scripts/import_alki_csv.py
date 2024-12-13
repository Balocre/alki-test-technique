#!/usr/bin/env python

import os
import sys

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client_3.write_client.client.exceptions import InfluxDBError

CLIENT_HOST = os.getenv("INFLUXDB_V2_URL")
CLIENT_ORG = os.getenv("INFLUXDB_V2_ORG")
CLIENT_TOKEN = os.getenv("INFLUXDB_V2_TOKEN")

BUCKET = "technical-test"


# TODO: paramatrize batchsize
def main(data_fp):
    print(CLIENT_HOST, CLIENT_ORG, CLIENT_TOKEN)
    with InfluxDBClient(
        CLIENT_HOST, CLIENT_TOKEN, debug=True, org=CLIENT_ORG
    ) as client:
        for df in pd.read_csv(
            data_fp,
            sep=";",
            dtype={"DATE": str, "CUSTOMER": str, "QUANTITY": int},
            index_col="DATE",
            parse_dates=["DATE"],
            chunksize=1_000,
        ):
            with client.write_api() as write_api:
                try:
                    write_api.write(
                        bucket=BUCKET,
                        record=df,
                        data_frame_measurement_name="customer_quantity",
                        data_frame_tag_columns=["CUSTOMER"],
                    )
                except InfluxDBError as e:
                    print(e)


if __name__ == "__main__":
    data_fp = sys.argv[1]
    main(data_fp)
