import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import mlflow
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.ml_src.data_clean import table_clean
from src.ml_src.data_ingest import ingest_data
from src.ml_src.utils import PROJ_ROOT, ModelBase, logger


class PipelineCreator(ModelBase):
    def __init__(self, config: dict):
        self.config_info = config
        self.target = self.config_info["ml_train"]["target"]
        self.test_size = self.config_info["ml_train"]["test_size"]
        self.random_seed = self.config_info["ml_train"]["random_seed"]
        self.sample_df = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model_names = self.config_info["ml_train"]["model_names"]
        self.model_params = self.config_info["ml_train"]["model_params"]
        self.cv = self.config_info["ml_train"]["cv"]
        self.scoring = self.config_info["ml_train"]["scoring"]
        self.n_jobs = self.config_info["ml_train"]["n_jobs"]
        self.res_list = None
        self.best_model = None
        self.best_acc = None
        self.best_each_list = None
        self.trained_clf = None
        self.pipeline_save = self.config_info["ml_train"]["model_save"]
        self.training_res = None

    def table_fetch(self):
        self.sample_df = table_clean(ingest_data())
        logger.info("Data Loaded and Cleaned")

    def data_split(self):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.sample_df.drop(self.target, axis=1),
            self.sample_df[self.target],
            test_size=self.test_size,
            random_state=self.random_seed,
        )
        logger.info("Data Split")

    def feature_select(self):
        self.features = self.sample_df.drop(self.target, axis=1).columns
        self.features_cat = (
            self.sample_df.drop(self.target, axis=1)
            .select_dtypes(include="object")
            .columns
        )
        self.features_num = (
            self.sample_df.drop(self.target, axis=1)
            .select_dtypes(include=["int64", "float64"])
            .columns
        )

    @staticmethod
    def get_clf(model_name: str):
        match model_name:
            case "GLM":
                return LogisticRegression()
            case "RF":
                return RandomForestClassifier()
            case "XGB":
                return XGBClassifier()
            case "LGBM":
                return LGBMClassifier()
            case _:
                return "Model not found"

    def pipeline_fit(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.features_num),
                ("cat", categorical_transformer, self.features_cat),
            ]
        )
        res_list = []
        for n in self.model_names:
            clf = self.get_clf(n)
            params = self.model_params[n]
            pipeline_obj = Pipeline(
                steps=[
                    # ("custom", CustomTransformer()),
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )
            gs = GridSearchCV(
                pipeline_obj,
                param_grid=params,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            gs.fit(self.X_train, self.y_train)
            best_clf = gs.best_estimator_
            res = {n: best_clf}
            res_list.append(res)
            self.res_list = res_list

    def model_comparison(self, models: list):
        best_acc = 0.0
        best_model = None
        best_each_list = []
        for _, model in enumerate(models):
            pred_train = model.predict(self.X_train)
            pred_valid = model.predict(self.X_valid)
            train_acc = accuracy_score(self.y_train, pred_train)
            valid_acc = accuracy_score(self.y_valid, pred_valid)
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = model
            precision, recall, fscore, _ = precision_recall_fscore_support(
                self.y_valid, pred_valid
            )
            best_each_dict = {
                "model": str(model),
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
            }
            self.best_model = best_model
            self.best_acc = best_acc
            best_each_list.append(best_each_dict)
            self.best_each_list = best_each_list

    def model_run(self):
        try:
            mlflow.autolog()
            with mlflow.start_run():
                trained_clfs = []
                self.pipeline_fit()
                for model_info in self.res_list:
                    best_clf = list(model_info.values())[0]
                    trained_clfs.append(best_clf)
                    self.model_comparison(trained_clfs)
                logger.info("Model Run")
                logger.info(f"best pipeline is: {self.best_model}")
                logger.info(f"best accuracy is: {self.best_acc}")
                self.trained_clfs = trained_clfs
                if self.pipeline_save:
                    model_path = Path(PROJ_ROOT) / "model_file" / "best_pipeline.pkl"
                    with model_path.open("wb") as f:
                        pickle.dump(self.best_model, f)
                logger.info(f"Model is trained and saved at {model_path}")
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                training_res = {
                    "time": str(dt_string),
                    "trained_model": str(self.model_names),
                    "best_each": str(self.best_each_list),
                    "best_model": str(self.best_model),
                    "best_acc": str(self.best_acc),
                }
                training_log_path = Path(PROJ_ROOT) / "model_file" / "training_log.json"
                with training_log_path.open("a") as f:
                    json.dump(training_res, f, indent=4)
                training_info_path = (
                    Path(PROJ_ROOT) / "model_file" / "training_info.json"
                )
                with training_info_path.open("w") as f:
                    json.dump(training_res, f, indent=4)
                self.training_res = training_res
        except (ValueError, KeyError, TypeError) as e:
            logger.error("An error occurred: %s", str(e))
        else:
            logger.info("Model Train Completed")
            return self.trained_clf, self.best_model, self.best_acc
