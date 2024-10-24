from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml_src.data_clean import table_clean
from src.ml_src.data_ingest import ingest_data
from src.ml_src.utils import MODEL_DIR


def load_model(model_dir: Path = MODEL_DIR) -> Pipeline:
    """Load the best model from the model directory."""
    model_path: Path = Path(model_dir) / "best_pipeline.pkl"
    with model_path.open("rb") as f:
        return joblib.load(f)


def load_test_data(n: int = 20) -> pd.DataFrame:
    """Load a sample of the test data to predict."""
    return (
        table_clean(ingest_data())
        .sample(n=n)
        .drop(columns=["history_of_mental_illness"])
        .reset_index(drop=True)
    )
