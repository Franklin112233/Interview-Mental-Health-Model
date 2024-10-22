import logging
from pathlib import Path

import pandas as pd

from src.ml_src.utils import DATA_DIR, config


class IngestData:
    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(
            Path(DATA_DIR, config["ml_utils"]["sample_data_file"]), index_col=0
        )


def ingest_data() -> pd.DataFrame:
    try:
        ingest_data = IngestData()
        return ingest_data.get_data()
    except Exception:
        logging.exception("An error occurred during data processing")
        raise
