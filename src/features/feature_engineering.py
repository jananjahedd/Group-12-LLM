"""
File: feature_engineering.py
Description:
"""
import pandas as pd
import logging
from pathlib import Path
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logs_path = log_dir / 'feature_engineering.log'

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


class FeatureEngineering:
    def __init__(self, data_path: str, suffix: str) -> None:
        Logger.log_info("Feature engineering class is initialized.")
        self._df = self._data_loader(data_path).copy(deep=True)
        self.suffix = suffix

    def _data_loader(self, path: str) -> pd.DataFrame:
        """Loads the data from the data/processed folder."""
        try:
            data = pd.read_csv(path, sep=',')
            Logger.log_info("The data has been loaded successfully.")
            return data
        except Exception as e:
            Logger.log_error(f"Error while loading the data: {str(e)}")
            return pd.DataFrame()

    def add_comment_length(self) -> None:
        """Add feature for the length of the comment."""
        try:
            self._df['comment_length'] = (
                self._df['comment_tokenized'].apply(len)
            )
            self._df['parent_comment_length'] = (
                self._df['parent_tokenized'].apply(len)
            )
            Logger.log_info("Successfully added the comments length.")
        except Exception as e:
            Logger.log_error(f"Error while calculationg lengths: {str(e)}")
            return

    def add_sentiment(self) -> None:
        """Add sentiment scores using TextBlob."""
        try:
            self._df['comment_sentiment'] = (
                self._df['comment'].apply(
                    lambda x: TextBlob(x).sentiment.polarity
                    )
            )
            self._df['parent_sentiment'] = (
                self._df['parent_comment'].apply(
                    lambda x: TextBlob(x).sentiment.polarity
                    )
            )
            Logger.log_info("Successfully added the sentiment scores.")
        except Exception as e:
            Logger.log_error("Error while evaluating the sentiment " +
                             f"scores: {str(e)}")
            return

    def scale_data(self, train_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> None:
        """Scale the numeric values in the data."""
        try:
            scaler = MinMaxScaler()

            # Scale only the training data
            train_df[['score', 'ups', 'downs']] = (
                scaler.fit_transform(train_df[['score', 'ups', 'downs']])
            )
            # Use the same scaler to transform the test data
            test_df[['score', 'ups', 'downs']] = (
                scaler.transform(test_df[['score', 'ups', 'downs']])
            )
            Logger.log_info("Successfully scaled the numeric data in " +
                            "train and test sets.")

            splits_dir = root / 'data' / 'splits'
            splits_dir.mkdir(parents=True, exist_ok=True)

            # Save the datasets
            train_df.to_csv(splits_dir / f'train_data_{self.suffix}.csv',
                            index=False)
            test_df.to_csv(splits_dir / f'test_data_{self.suffix}.csv',
                           index=False)
            Logger.log_info(f"The datasets were saved at {splits_dir}")

            return train_df, test_df
        except Exception as e:
            Logger.log_error(f"Error while scaling the data: {str(e)}")
            return

    def _splitting(self) -> None:
        """Split the data into training and testing sets."""
        try:
            train_df, test_df = train_test_split(self._df, test_size=0.3,
                                                 random_state=42)
            Logger.log_info("Successfully split the data into " +
                            "training and testing.")

            return train_df, test_df

        except Exception as e:
            Logger.log_error(f"Error while splitting the data: {str(e)}")
            return

    def _save_data(self) -> None:
        """Saves the changes in a new file."""
        try:
            output_dir = root / 'data' / 'processed'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'normalized_sarcasm_{self.suffix}.csv'
            self._df.to_csv(output_path, index=False)
            Logger.log_info("Successfully saved the modified " +
                            f"data at {output_path}")
        except Exception as e:
            Logger.log_error(f"Error while saving the data: {str(e)}")
            return


if __name__ == "__main__":
    """Executer feature engineering on the preprocessed data."""
    processed_dir = root / 'data' / 'processed'
    bert_path = processed_dir / 'processed_sarcasm_bert.csv'
    distilbert_path = processed_dir / 'processed_sarcasm_distilbert.csv'
    paths = [(bert_path, 'bert'), (distilbert_path, 'distilbert')]

    for data_path, suffix in paths:
        engineer = FeatureEngineering(data_path, suffix)
        engineer.add_comment_length()
        engineer.add_sentiment()
        engineer.add_time()
        engineer._save_data()
        train, test = engineer._splitting()
        engineer.scale_data(train, test)
