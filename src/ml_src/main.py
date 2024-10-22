from src.ml_src.predict import load_model, load_test_data
from src.ml_src.train_pipeline import PipelineCreator
from src.ml_src.utils import config, logger
import click


def model_train(config=config):
    try:
        pl = PipelineCreator(config)
        pl.table_fetch()
        pl.data_split()
        pl.feature_select()
        pl.model_run()
    except Exception:
        logger.exception("An error occurred during model training")
        raise


def model_predict():
    try:
        pl = load_model()
        test_df = load_test_data()
        pred = pl.predict(test_df)
        prob = pl.predict_proba(test_df)
        logger.info(pred, prob)
    except Exception:
        logger.exception("An error occurred during model prediction")
        raise


@click.command()
@click.option("--predict", is_flag=True, help="Using the model to make predictions")
def main(predict):
    if predict:
        model_predict()
    else:
        model_train()


if __name__ == "__main__":
    main()
