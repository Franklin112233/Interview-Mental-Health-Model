import json
import sys

import pandas as pd
from dotenv import load_dotenv
from icecream import ic
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from src.genai_src.prompt_template import template
from src.genai_src.utils import DATA_DIR, logger

load_dotenv()


def sentiment_llm(text_review: str, team: str) -> json:
    """_Create a sentiment analysis model using GPT and langchain.

    Args:
        text_review (str): member text
        team (str): team in "Customer Support", "PA Agent", "Technical Support", or "Unknown"

    Returns:
        Resonse (json) from langchain llm model

    """
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0,
    )
    prompt = PromptTemplate(template=template, input_variables=["text_review", "team"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    if prompt:
        response = llm_chain.run({"text_review": text_review, "team": team})
    return json.loads(response)


def sentment_run() -> pd.DataFrame:
    """Run sentiment analysis and save the result in the DATA_DIR.

    Returns:
        Response dataframe with the fields of Outcome, Sentiment, Sentiment_Score, and Summary

    """
    member_transcript_df: pd.DataFrame = pd.read_csv(DATA_DIR / "member_transcripts.csv")[
        ["Member_Text", "Team"]
    ]
    outcome_list = []
    sentiment_list = []
    sentiment_score_list = []
    summary_list = []
    for _, row in member_transcript_df.iterrows():
        text_review = row["Member_Text"]
        team = row["Team"]
        response = sentiment_llm(text_review, team)
        outcome = response["outcome"]
        sentiment = response["sentiment"]
        sentiment_score = response["sentiment_score"]
        summary = response["summary"]
        outcome_list.append(outcome)
        sentiment_list.append(sentiment)
        sentiment_score_list.append(sentiment_score)
        summary_list.append(summary)
    response_df = pd.DataFrame(
        {
            "Outcome": outcome_list,
            "Sentiment": sentiment_list,
            "Sentiment_Score": sentiment_score_list,
            "Summary": summary_list,
        }
    )
    result_df: pd.DataFrame = pd.concat([member_transcript_df, response_df], axis=1)
    result_df.to_csv(DATA_DIR / "sentiment_result.csv", index=False)
    logger.info(f"Sentiment analysis result saved in {DATA_DIR}'/sentiment_result.csv")
    return result_df


if __name__ == "__main__":
    try:
        result_df: pd.DataFrame = sentment_run()
        ic(result_df)
    except json.JSONDecodeError as e:
        logger.error(f"An json decode error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Sentiment analysis completed")
        sys.exit(0)
