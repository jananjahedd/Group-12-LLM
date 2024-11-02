# Large Language Models Project

## Janan Jahed (S5107318), Alexandru Cernat (S5190584), Andrei Medesan (S5130727)

## Group 12

The repository contains the necessary files to investigate our research involving Large Language Models (LLMs). The study involves comparing the performances of the existing models - BERT and Distilbert - after fine-tuning them with Low Rank Adaptation (LoRA) and prompt engineering method on sarcasm detection.


## Dataset Download via Kaggle API

This repository uses the Kaggle API to download datasets directly from Kaggle. We were interested in the Sarcasm on Reddit datasets (containing 1.3 million labelled comments) introduced by Dan Ofer. To overview the dataset, [click here](https://www.kaggle.com/datasets/danofer/sarcasm/data). To download the datasets, follow these steps:

### Step 1: Install Kaggle API

Make sure you have `kaggle` installed by running:

```sh
!pip install kaggle
```

Following, create a Kaggle account and navigate to your Account -> Settings -> API -> `Create New Token`. This will download a file called `kaggle.json`. This file should be in the following directories on the user computer:

```sh
- On Windows: C:\Users\<Your-Username>\.kaggle\
- On Mac/Linux: /home/<Your-Username>/.kaggle/
```

If the directory cannot be found manually, open Terminal or Command Prompt and run the following lines to make sure that the folder `.kaggle` exists and to move the JSON file in the respective folder.

On Windows:
```sh
mkdir C:\Users\<Your-Username>\.kaggle\
mv \path\to\kaggle.json C:\Users\<Your-Username>\.kaggle\kaggle.json
```

On Mac/Linux:
```sh
mkdir /Users/<Your-Username>/.kaggle/
mv /path/to/kaggle.json /Users/<Your-Username>/.kaggle/kaggle.json
```

Executing the `preprocessin.py` file will download the `sarcasm.zip` file directly in the `data/` folder.

To run the code:
- first run the `data_donwloader.py` file which will download the SARC dataset from kaggle.
- Then run `preprocessing.py` to preprocess the raw dataset.
- After the processed data is saved, run `feature_engineering.py` to get the data splits and more information about the dataset's features.
- Lastly, you can run each model separately (bert.py, distilbert.py, prompt_bert.py, prompt_distilbert.py) to observe the performance.

**It is recommended to run the models using a computer cluster (such as Habrok from the University of Groningen) or on a cloud computing platform since they are large models and require a strong GPU to run.**
