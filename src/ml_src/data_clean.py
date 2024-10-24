from pathlib import Path

import pandas as pd
from icecream import ic
from sklearn.base import BaseEstimator, TransformerMixin

from src.ml_src.utils import DATA_DIR, config, logger


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


def remove_outliers(sample_df: pd.DataFrame, column: pd.Series) -> pd.DataFrame:
    q1 = sample_df[column].quantile(0.25)
    q3 = sample_df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return sample_df[
        (sample_df[column] >= lower_bound) & (sample_df[column] <= upper_bound)
    ]


def table_clean(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df = sample_df.loc[:, ~sample_df.columns.str.contains("^Unnamed")]
    sample_df_renamed = sample_df.rename(
        columns={
            # "Unnamed: 0": "index",
            "Name": "name",
            "Age": "age",
            "Marital Status": "marital_status",
            "Education Level": "education_level",
            "Number of Children": "num_of_children",
            "Smoking Status": "smoking_status",
            "Physical Activity Level": "physical_activity_level",
            "Employment Status": "employment_status",
            "Income": "income",
            "Alcohol Consumption": "alcohol_consumption",
            "Dietary Habits": "dietary_habits",
            "Sleep Patterns": "sleeping_patterns",
            "History of Mental Illness": "history_of_mental_illness",
            "History of Substance Abuse": "history_of_substance_abuse",
            "Family History of Depression": "family_history_of_depression",
            "Chronic Medical Conditions": "chronic_medical_conditions",
        }
    ).drop(columns=["name", "employment_status"], inplace=False)

    sample_df_renamed["history_of_mental_illness"] = sample_df_renamed[
        "history_of_mental_illness"
    ].replace(("Yes", "No"), (1, 0))
    sample_df_renamed["marital_status"] = sample_df_renamed["marital_status"].replace(
        ("Married", "Single", "Widowed", "Divorced"), (1, 2, 3, 3)
    )
    sample_df_renamed["education_level"] = sample_df_renamed["education_level"].replace(
        (
            "Bachelor's Degree",
            "High School",
            "Associate Degree",
            "Master's Degree",
            "PhD",
        ),
        (1, 2, 3, 4, 4),
    )
    sample_df_renamed["num_of_children"] = (
        sample_df_renamed["num_of_children"].replace("4", "3").astype(int)
    )
    sample_df_renamed["smoking_status"] = sample_df_renamed["smoking_status"].replace(
        ("Non-smoker", "Former", "Current"), (1, 2, 3)
    )
    sample_df_renamed["physical_activity_level"] = sample_df_renamed[
        "physical_activity_level"
    ].replace(("Sedentary", "Moderate", "Active"), (1, 2, 3))
    sample_df_renamed["alcohol_consumption"] = sample_df_renamed[
        "alcohol_consumption"
    ].replace(("Low", "Moderate", "High"), (1, 2, 3))
    sample_df_renamed["dietary_habits"] = (
        sample_df_renamed["dietary_habits"]
        .replace(("Unhealthy", "Moderate", "Healthy"), (1, 2, 3))
        .astype(int)
    )
    sample_df_renamed["sleeping_patterns"] = sample_df_renamed[
        "sleeping_patterns"
    ].replace(("Poor", "Fair", "Good"), (1, 2, 3))

    sample_df_renamed.pipe(remove_outliers, "income").pipe(remove_outliers, "age")

    return sample_df_renamed.reset_index(drop=True)


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return table_clean(X)


if __name__ == "__main__":
    create_sample_data()
    ic(table_clean(pd.read_csv(Path(DATA_DIR, "depression_data_sample.csv"))))
    table_clean(pd.read_csv(Path(DATA_DIR, "depression_data_sample.csv"))).info()
