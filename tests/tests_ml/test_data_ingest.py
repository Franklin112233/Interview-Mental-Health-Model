import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ml_src.data_ingest import IngestData, ingest_data

sample_data = pd.DataFrame({"column1": [1, 2, 3], "column2": ["A", "B", "C"]})


@patch("src.ml_src.data_ingest.pd.read_csv")
@patch("src.ml_src.data_ingest.config")
@patch("src.ml_src.data_ingest.Path")
def test_ingest_data(mock_path, mock_config, mock_read_csv) -> None:
    mock_config["ml_utils"]["sample_data_file"] = "sample_data.csv"
    mock_read_csv.return_value = sample_data
    result = ingest_data()
    mock_read_csv.assert_called_once_with(
        mock_path(mock_path.DATA_DIR, "sample_data.csv"), index_col=0
    )
    pd.testing.assert_frame_equal(result, sample_data)


@patch("src.ml_src.data_ingest.logging")
@patch.object(IngestData, "get_data", side_effect=Exception("Test Exception"))
def test_ingest_data_exception(mock_get_data, mock_logging) -> None:
    result = ingest_data()
    mock_get_data.assert_called_once()
    mock_logging.exception.assert_called_once_with(
        "An error occurred during data processing"
    )
    assert result is None
