import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

text = pd.read_csv("data/member_transcripts.csv")["Member_Text"][0]

print(text)
wordcloud = WordCloud(
    max_font_size=50, max_words=100, background_color="white"
).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
