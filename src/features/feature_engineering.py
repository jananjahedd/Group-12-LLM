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
    def __init__(self) -> None:
        Logger.log_info("Feature engineering class is initialized.")
        self._df = self._data_loader().copy(deep=True)

    def _data_loader(self) -> pd.DataFrame:
        """Loads the data from the data/processed folder."""
        data_path = root / 'data' / 'processed' / 'processed_sarcasm.csv'

        try:
            data = pd.read_csv(data_path, sep=',')
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

    def add_time(self) -> None:
        """Extrcat time based features from date or created_utc columns."""
        try:
            self._df['day_of_week'] = (
                pd.to_datetime(self._df['created_utc']).dt.day_of_week
            )
            Logger.log_info("Successfully added day of the week.")
        except Exception as e:
            Logger.log_error(f"Error while adding days: {str(e)}")
            return

    def scale_data(self) -> None:
        """Scale the numeric values in the data."""
        try:
            scaler = MinMaxScaler()
            self._df[['score', 'ups', 'downs']] = (
                scaler.fit_transform(self._df[['score', 'ups', 'downs']])
            )
            Logger.log_info("Successfully scaled the numeric data.")
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

            splits_dir = root / 'data' / 'processed'
            splits_dir.mkdir(parents=True, exist_ok=True)

            # save the datasets
            train_df.to_csv(splits_dir / 'train_data.csv', index=False)
            test_df.to_csv(splits_dir / 'test_data.csv', index=False)
            Logger.log_info(f"The datasets were saved at {splits_dir}")

            return train_df, test_df

        except Exception as e:
            Logger.log_error(f"Error while splitting the data: {str(e)}")
            return

    def _save_data(self) -> None:
        """Saves the changes in a new file."""
        try:
            output_dir = root / 'data' / 'processed'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'normalized_sarcasm.csv'
            self._df.to_csv(output_path, index=False)
            Logger.log_info("Successfully saved the modified " +
                            f"data at {output_path}")
        except Exception as e:
            Logger.log_error(f"Error while saving the data: {str(e)}")
            return


if __name__ == "__main__":
    """Executer feature engineering on the preprocessed data."""
    engineer = FeatureEngineering()
    engineer.add_comment_length()
    engineer.add_sentiment()
    engineer.add_time()
    engineer.scale_data()
    engineer._save_data()
    engineer._splitting()