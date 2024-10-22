import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
REPORTS_DIR = PROJ_ROOT / "reports"


with Path(f"{Path(__file__).resolve().parents[2]}/config.yml").open() as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception()


def create_sample_data(
    data_file: pd.DataFrame = config["ml_utils"]["raw_data_file"],
    fraction: float = config["ml_utils"]["sample_fraction"],
    random_state: int = config["ml_utils"]["random_state"],
) -> tuple:
    return (
        pd.read_csv(Path(DATA_DIR, data_file))
        .sample(frac=fraction, random_state=random_state)
        .reset_index(drop=True)
        .to_csv(Path(DATA_DIR, "depression_data_sample.csv", index=False)),
        logger.info(f"Sample data created with fraction {fraction}"),
        logger.info(f"Sample data saved: {DATA_DIR}/depression_data_sample.csv"),
    )


def create_data_profile(
    data_file: pd.DataFrame = config["ml_utils"]["sample_data_file"],
) -> None:
    from ydata_profiling import ProfileReport

    df_to_profile = pd.read_csv(Path(DATA_DIR, data_file), index_col=0)
    data_profile = ProfileReport(
        df_to_profile, title="Data Profiling Report", explorative=True
    )
    data_profile.to_file(Path(REPORTS_DIR, "depression_data_profile.html"))
    logger.info(
        f"Data profile created and saved: {REPORTS_DIR}/depression_data_profile.html"
    )


class ModelBase(ABC):
    @abstractmethod
    def table_fetch(self):
        pass

    @abstractmethod
    def data_split(self):
        pass

    @abstractmethod
    def feature_select(self):
        pass

    @abstractmethod
    def get_clf(self):
        pass

    @abstractmethod
    def pipeline_fit(self):
        pass

    @abstractmethod
    def model_comparison(self):
        pass

    @abstractmethod
    def model_run(self):
        pass

if __name__ == "__main__":
    create_sample_data()
    create_data_profile()
