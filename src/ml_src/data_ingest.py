import logging
from pathlib import Path

import pandas as pd

from src.ml_src.utils import DATA_DIR, config


class IngestData:
    """Ingest data from a CSV file."""

    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(
            Path(DATA_DIR, config["ml_utils"]["sample_data_file"]), index_col=0
        )


def ingest_data() -> pd.DataFrame:
    """Run the data ingestion process."""
    try:
        ingest_data = IngestData()
        return ingest_data.get_data()
    except Exception:
        logging.exception("An error occurred during data processing")
        raise
