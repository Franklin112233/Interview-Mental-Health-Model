from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml_src.data_clean import table_clean
from src.ml_src.data_ingest import ingest_data
from src.ml_src.utils import PROJ_ROOT


def load_model(proj_root: str = PROJ_ROOT) -> Pipeline:
    model_path = Path(proj_root) / "model_file" / "best_pipeline.pkl"
    with model_path.open("rb") as f:
        return joblib.load(f)


def load_test_data(n: int = 20) -> pd.DataFrame:
    return (
        table_clean(ingest_data())
        .sample(n=n)
        .drop(columns=["history_of_mental_illness"])
        .reset_index(drop=True)
    )
