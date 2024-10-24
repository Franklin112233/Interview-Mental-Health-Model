import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJ_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJ_ROOT / "data"
REPORTS_DIR: Path = PROJ_ROOT / "reports"
MODEL_DIR: Path = PROJ_ROOT / "model_files"

with Path(f"{PROJ_ROOT}/config.yml").open() as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception()


class ModelBase(ABC):
    """Base class for the model pipeline."""

    @abstractmethod
    def table_fetch(self) -> None:
        pass

    @abstractmethod
    def data_split(self) -> None:
        pass

    @abstractmethod
    def feature_select(self) -> None:
        pass

    @abstractmethod
    def get_clf(self) -> None:
        pass

    @abstractmethod
    def pipeline_fit(self) -> None:
        pass

    @abstractmethod
    def model_comparison(self) -> None:
        pass

    @abstractmethod
    def model_run(self) -> None:
        pass
