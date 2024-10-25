import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.genai_src.utils import DATA_DIR, REPORTS_DIR, logger


def create_word_cloud(sentiment_result_df) -> None:
    """Create a word cloud from the member text."""
    text = sentiment_result_df["Member_Text"][0]
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white"
    ).generate(text)
    wordcloud.to_file(Path(REPORTS_DIR, "word_cloud.png"))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def sentiment_score_vs_outcome_barplot(sentiment_result_df) -> None:
    """Create a bar plot of the average sentiment score by outcome."""
    plt.figure(figsize=(10, 6))
    sentiment_result_df.groupby("Outcome")["Sentiment_Score"].mean().plot(
        kind="bar", color="skyblue"
    )
    plt.xlabel("Outcome")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=0)
    plt.title("Average Sentiment Score by Outcome")
    plt.savefig(Path(REPORTS_DIR, "sentiment_score_vs_outcome.png"))
    plt.show()


def sentiment_score_vs_team_barplot(sentiment_result_df) -> None:
    """Create a bar plot of the average sentiment score by team."""
    plt.figure(figsize=(10, 6))
    sentiment_result_df.groupby("Team")["Sentiment_Score"].mean().plot(
        kind="bar", color="skyblue"
    )
    plt.xlabel("Team")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=0)
    plt.title("Average Sentiment Score by Team")
    plt.savefig(Path(REPORTS_DIR, "sentiment_score_vs_team.png"))
    plt.show()


def sentiment_proportion_by_team_barplot(sentiment_result_df) -> None:
    """Create a bar plot of the proportion of sentiments by team."""
    sentiment_proportion = sentiment_result_df.pivot_table(
        index="Team", columns="Sentiment", aggfunc="size", fill_value=0
    )
    sentiment_proportion = sentiment_proportion.div(
        sentiment_proportion.sum(axis=1), axis=0
    )
    sentiment_proportion = sentiment_proportion.reset_index()
    sentiment_proportion_melted = sentiment_proportion.melt(
        id_vars="Team",
        value_vars=["positive", "neutral", "negative"],
        var_name="Sentiment",
        value_name="Proportion",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Team", y="Proportion", hue="Sentiment", data=sentiment_proportion_melted
    )
    plt.xlabel("Team")
    plt.ylabel("Proportion")
    plt.title("Proportion of Sentiments by Team")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(Path(REPORTS_DIR, "sentiment_proportion_vs_team.png"))
    plt.show()


def sentiment_vs_outcome_barplot(sentiment_result_df) -> None:
    """Create a bar plot of the count of outcomes by sentiment."""
    sentiment_outcome_count = (
        sentiment_result_df.groupby(["Sentiment", "Outcome"])  # noqa: PD010
        .size()
        .unstack()
        .fillna(0)
    )
    sentiment_outcome_count = sentiment_outcome_count.reset_index()
    sentiment_outcome_count_melted = sentiment_outcome_count.melt(
        id_vars="Sentiment",
        value_vars=sentiment_outcome_count.columns[1:],
        var_name="Outcome",
        value_name="Count",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Sentiment", y="Count", hue="Outcome", data=sentiment_outcome_count_melted
    )
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Count of Outcomes by Sentiment")
    plt.xticks(rotation=0)
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig(Path(REPORTS_DIR, "sentiment_vs_outcome.png"))
    plt.show()


def plot_run() -> None:
    """Run the visualization functions."""
    data_file = "sentiment_result.csv"
    sentiment_result_df: pd.DataFrame = pd.read_csv(Path(DATA_DIR, data_file))
    create_word_cloud(sentiment_result_df)
    sentiment_score_vs_outcome_barplot(sentiment_result_df)
    sentiment_score_vs_team_barplot(sentiment_result_df)
    sentiment_proportion_by_team_barplot(sentiment_result_df)
    sentiment_vs_outcome_barplot(sentiment_result_df)
    logger.info(f"Visualization plots saved in the {REPORTS_DIR}.")


if __name__ == "__main__":
    try:
        plot_run()
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Visualization completed.")
        sys.exit(0)
