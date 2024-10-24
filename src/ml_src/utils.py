import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
REPORTS_DIR = PROJ_ROOT / "reports"
MODEL_DIR = PROJ_ROOT / "model_files"

with Path(f"{PROJ_ROOT}/config.yml").open() as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception()


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
