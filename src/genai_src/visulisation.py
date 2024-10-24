from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src.genai_src.utils import DATA_DIR

data_file = "sentiment_result.csv"
text = pd.read_csv(Path(DATA_DIR, data_file))["Member_Text"][0]


def create_word_cloud(text: str) -> None:
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white"
    ).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def run_word_cloud() -> None:
    data_file = "sentiment_result.csv"
    text = pd.read_csv(Path(DATA_DIR, data_file))["Member_Text"][0]
    create_word_cloud(text)


if __name__ == "__main__":
    run_word_cloud()
