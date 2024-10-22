import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.ml_src.utils import config


def table_clean(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df_cleaned = sample_df.rename(
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
            "Sleeping Patterns": "sleeping_patterns",
            "History of Mental Illness": "history_of_mental_illness",
            "History of Substance Abuse": "history_of_substance_abuse",
            "Family History of Depression": "family_history_of_depression",
            "Chronic Medical Conditions": "chronic_medical_conditions",
        }
    ).drop(columns=["name", "employment_status"], inplace=False)
    sample_df_cleaned["history_of_mental_illness"] = sample_df_cleaned[
        "history_of_mental_illness"
    ].replace(("Yes", "No"), (1, 0))
    return sample_df_cleaned


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return table_clean(X)
