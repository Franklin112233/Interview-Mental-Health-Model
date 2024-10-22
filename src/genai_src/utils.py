import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_v3"
REPORTS_DIR = PROJ_ROOT / "reports"


with Path(f"{PROJ_ROOT}/config.yml").open() as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception()
