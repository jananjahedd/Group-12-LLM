"""
File: preprocessing.py
Authors: Andrei Medesan, Janan Jahed, and Alexandru Cernat
Description:
"""
import os
import kaggle
import zipfile
import logging
from pathlib import Path


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

    @staticmethod
    def log_warning(message):
        logging.warnining(message)


class DatasetsLoader:
    def __init__(self, data_name: str, output_dir: str = 'data/', unzip:
                 bool = True) -> None:
        """
        Initializes the DatasetsLoader class.

        :param data_name: name of the Kaggle data.
        :param output_dir: the directory for the data.
        :param unzip: boolean to decide whether to open the data.
        """
        self.data_name = data_name
        self.output_dir = output_dir
        self.unzip = unzip
        self.output_path = os.path.join(output_dir, data_name.split('/')[-1] +
                                        '.zip')
        Logger.log_info("Initializing class for downloading the aggle data.")

        # ensure the directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._check_credentials()

    def _check_credentials(self) -> None:
        """Checks the credentials for downloading the data."""
        kaggle_file = "~/.kaggle/kaggle.json"
        if not os.path.exists(os.path.expanduser(kaggle_file)):
            raise FileNotFoundError("Kaggle API key not found. Please" +
                                    "follow the setup instructions in" +
                                    "the README.")

    def download_data(self) -> None:
        """Downloads the datasets."""
        Logger.log_info(f"Downloading the datasets {self.data_name} to" +
                        f"{self.output_dir}")
        kaggle.api.dataset_download_files(self.data_name, path=self.output_dir,
                                          unzip=False)

        if self.unzip:
            self._unzip_data()
        Logger.log_info("Data downloaded successfully.")

    def _unzip_data(self) -> None:
        """Unzips the data in the designated directory."""
        Logger.log_info(f"Unzipping the data in '{self.output_path}'...")
        with zipfile.ZipFile(self.output_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        Logger.log_info(f"Data extracted to {self.output_path}")


if __name__ == "__main__":
    dataset_loader = DatasetsLoader(data_name='danofer/sarcasm')
    dataset_loader.download_data()
