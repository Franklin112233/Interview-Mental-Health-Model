import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.genai_src.utils import DATA_DIR, TRANSCRIPTS_DIR, logger


def parse_transcripts(transcripts_dir: Path = TRANSCRIPTS_DIR) -> pd.DataFrame:
    """Parse the transcripts of the members and return a DataFrame.

    Args:
        transcripts_dir (Path, optional): Description. Defaults to TRANSCRIPTS_DIR.

    Returns:
        pd.DataFrame: Table contain the information of filename, member text, and team.

    """
    member_transcripts_data = []
    try:
        for file_name in os.listdir(path=transcripts_dir):
            if file_name.endswith(".txt"):
                file_path: Path = transcripts_dir / file_name
                with file_path.open() as file:
                    transcript: str = file.read()
                member_texts: list[Any] = re.findall(
                    r"Member: (.*?)\n", transcript, re.DOTALL
                )
                member_paragraph = " ".join(member_texts)
            if "Customer Support" in transcript:
                team = "Customer_Support"
            elif "PA Agent" in transcript:
                team = "PA_Agent"
            elif "Technical Support" in transcript:
                team = "Technical_Support"
            else:
                team = "Unknown"
            member_transcripts_data.append(
                {
                    "File_Name": file_name,
                    "Member_Text": member_paragraph,
                    "Team": team,
                }
            )
    except FileNotFoundError as e:
        logger.error(f"File system error occurred: {e}")
    except OSError as e:
        logger.error(f"File error occurred: {e}")
    except re.error as e:
        logger.error(f"Regex error occurred: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Pandas error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    else:
        logger.info("Transcripts parsed successfully")
    return pd.DataFrame(member_transcripts_data)


def prepare_transcripts(
    transcripts_dir: Path = TRANSCRIPTS_DIR, data_dir: Path = DATA_DIR
) -> None:
    """Extract the member text and team from the transcripts and save it as a csv file.

    Args:
        transcripts_dir (Path, optional): _description_. Defaults to TRANSCRIPTS_DIR.
        data_dir (Path, optional): _description_. Defaults to DATA_DIR.

    """
    member_transcripts_df: pd.DataFrame = parse_transcripts(
        transcripts_dir=transcripts_dir
    )
    member_transcripts_df.to_csv(Path(data_dir) / "member_transcripts.csv", index=False)


if __name__ == "__main__":
    try:
        prepare_transcripts()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
