"""
File: preprocessing.py
Authors: Andrei Medesan, Janan Jahed, and Alexandru Cernat
Description:

Drop missing data: Essential, as LLMs can't process incomplete data.
Lowercase text: Helpful for consistency in training.
Tokenization: Required to break text into smaller units for the model.
Remove punctuations: Simplifies the text.
"""
import pandas as pd
import logging
from pathlib import Path
import contractions
import re
import nltk
from nltk.tokenize import word_tokenize


root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'preprocessing_info.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format=format_style)


class Logger:
    """Simple logger class."""
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)


class DataPreprocessing:
    def __init__(self) -> None:
        Logger.log_info("Preprocessing class is initialized.")
        self._df = self._data_loader()

    def _data_loader(self) -> pd.DataFrame:
        """Loads the data from the data/ folder."""
        data_path = root / 'data' / 'raw' / 'train-balanced-sarcasm.csv'

        try:
            data = pd.read_csv(data_path, sep=',')
            Logger.log_info("The data has been loaded successfully.")
            return data
        except Exception as e:
            Logger.log_error(f"Error while loading the data: {str(e)}")
            return pd.DataFrame()

    def handle_missing_values(self) -> None:
        """Drops the rows where there are missing columns (55)."""
        try:
            missing_values = self._df['comment'].isnull().sum()
            Logger.log_info(f"Found {missing_values} missing comments " +
                            "in the data. Dropping them...")

            self._df.dropna(subset=['comment', 'parent_comment'], inplace=True)
            Logger.log_info("Successfully handled missing values.")
        except Exception as e:
            Logger.log_error(f"Error while handing missing values: {str(e)}")
            return pd.DataFrame()

    def expand_contractions(self) -> None:
        """Expands contractions such as 'I'm' -> 'I am'."""
        try:
            self._df['comment'] = self._df['comment'].apply(
                lambda x: contractions.fix(x)
            )
            self._df['parent_comment'] = self._df['parent_comment'].apply(
                lambda x: contractions.fix(x)
            )
            Logger.log_info("Successfully expanded the contractions in " +
                            "both 'comment' and 'parent_comment' columns.")

        except Exception as e:
            Logger.log_error(f"Error while expanding contractions: {str(e)}")
            return pd.DataFrame()

    def lowercase_text(self) -> None:
        """Changes the text to all lowercase for consistency."""
        try:
            self._df['comment'] = self._df['comment'].str.lower()
            self._df['parent_comment'] = self._df['parent_comment'].str.lower()
            Logger.log_info("Successfully lowered the text from both " +
                            "'comment' and 'parent_comment' columns.")
        except Exception as e:
            Logger.log_error(f"Error while lowercasing the texts: {str(e)}")
            return pd.DataFrame()

    def remove_punctuation(self) -> None:
        """Removes all punctuation for simplicity."""
        try:
            # add punctuation column for loging the charcaters then drop it
            self._df['removed_punctuation'] = self._df['comment'].apply(
                lambda x: re.findall(r'[^a-z\s]', x)
            )

            removed_chars = set(
                char for sublist in self._df['removed_punctuation']
                for char in sublist
            )
            Logger.log_info(f"Removed punctuation characters: {removed_chars}")

            # remove punctuation in comment column
            self._df['comment'] = (
                self._df['comment'].str.replace(r'[^a-z\s]', '', regex=True)
            )

            # remove punctuation in the parent_comment column
            self._df['parent_comment'] = (
                self._df['parent_comment'].str.replace(r'[^a-z\s]', '',
                                                       regex=True)
            )
            Logger.log_info("Successfully removed all punctuation special " +
                            "characters from both 'comment' and " +
                            "'parent_comment' columns.")

            # remove the column as it is unnecessary
            self._df.drop(columns=['removed_punctuation'], inplace=True)

        except Exception as e:
            Logger.log_error(f"Error while removing punctuation: {str(e)}")
            return pd.DataFrame()

    def tokenization(self) -> None:
        """Tokenize the comments with the nltk library."""
        try:
            # donwload the necessary resources
            nltk.download('punkt_tab')
            self._df['comment_tokenized'] = (
                self._df['comment'].apply(word_tokenize)
            )
            self._df['parent_tokenized'] = (
                self._df['parent_comment'].apply(word_tokenize)
            )
            Logger.log_info("Successfully tokenized the comments.")

            self._df = self._df[self._df['comment'].str.strip() != '']
            self._df = self._df[self._df['parent_comment'].str.strip() != '']
            Logger.log_info("Dropped rows with empty comments.")

        except Exception as e:
            Logger.log_error(f"Error while tokenizing the comments: {str(e)}")
            return pd.DataFrame()

    def _save_data(self) -> None:
        """Saves the data in the designated folder."""
        # check the directories and create the file
        processed_dir = root / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_file = processed_dir / 'processed_sarcasm.csv'
        # save the file in the folder
        self._df.to_csv(processed_file, index=False)
        Logger.log_info(f"Processed file saved at {processed_file}.")


if __name__ == "__main__":
    """Preprocess the sarcasm data."""
    preprocessor = DataPreprocessing()
    preprocessor.handle_missing_values()
    preprocessor.expand_contractions()
    preprocessor.lowercase_text()
    preprocessor.remove_punctuation()
    preprocessor.tokenization()
    preprocessor._save_data()