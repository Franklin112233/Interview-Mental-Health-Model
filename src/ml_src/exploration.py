import os
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

from src.ml_src.data_clean import table_clean
from src.ml_src.predict import load_model
from src.ml_src.utils import DATA_DIR, REPORTS_DIR, config, logger


def create_data_profile(
    data_file: pd.DataFrame = config["ml_utils"]["sample_data_file"],
) -> None:
    from ydata_profiling import ProfileReport

    df_loaded: pd.DataFrame = pd.read_csv(Path(DATA_DIR, data_file), index_col=0)
    data_profile = ProfileReport(
        df_loaded, title="Data Profiling Report", explorative=True
    )
    data_profile.to_file(Path(REPORTS_DIR, "depression_data_profile.html"))
    logger.info(
        f"Data profile created and saved: {REPORTS_DIR}/depression_data_profile.html"
    )


def create_drift_report(
    data_file: pd.DataFrame = config["ml_utils"]["sample_data_file"],
) -> None:
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    df_loaded: pd.DataFrame = pd.read_csv(Path(DATA_DIR, data_file), index_col=0)
    df_sample: pd.DataFrame = table_clean(df_loaded)
    df_mental_illness: pd.DataFrame = df_sample[
        df_sample["history_of_mental_illness"] == 1
    ]
    df_no_mental_illness: pd.DataFrame = df_sample[
        df_sample["history_of_mental_illness"] == 0
    ]
    data_drift_report.run(
        reference_data=df_no_mental_illness, current_data=df_mental_illness
    )
    output_file = os.path.join(REPORTS_DIR, "data_drift_report.html")
    data_drift_report.save_html(output_file)
    logger.info(f"Data drift report saved: {output_file}")


def create_diagnosis(
    data_file: pd.DataFrame = config["ml_utils"]["sample_data_file"],
) -> None:
    df_loaded: pd.DataFrame = pd.read_csv(Path(DATA_DIR, data_file), index_col=0)
    df_sample: pd.DataFrame = table_clean(df_loaded).sample(n=10000)
    x_test: pd.DataFrame = df_sample.drop(["history_of_mental_illness"], axis=1)
    y_test: pd.DataFrame = df_sample[["history_of_mental_illness"]]
    model = load_model()
    explainer = ClassifierExplainer(model, x_test, y_test)
    dignosis_board = ExplainerDashboard(
        explainer,
        title="Model Diagnosis",
        # whatif=False,
        # shap_interaction=False,
        # decision_trees=False
    )
    dignosis_board.run(host="0.0.0.0", port=8508, use_waitress=True)  # noqa: S104


if __name__ == "__main__":
    report_type = "diagnosis"
    match report_type:
        case "profile":
            create_data_profile()
        case "drift":
            create_drift_report()
        case "diagnosis":
            create_diagnosis()
        case _:
            logger.error("Invalid report type")
