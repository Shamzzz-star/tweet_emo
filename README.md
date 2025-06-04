# Tweet_emo

A machine learning project for emotion classification in tweets using Support Vector Machines (SVM).

## Overview

This repository contains code for training and evaluating SVM classifiers on a dataset of tweets labeled with emotions. The project leverages natural language processing techniques to preprocess and vectorize text data, then trains SVM models with various kernels to determine the best-performing configuration.

## Features

- Preprocessing of tweet text (cleaning, stopword removal, lemmatization)
- TF-IDF vectorization of text data
- SVM classification with multiple kernels (`linear`, `poly`, `rbf`, `sigmoid`)
- Evaluation of models (accuracy, classification report, confusion matrix)
- Saves the best performing model for future use

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- seaborn
- matplotlib
- joblib

Install dependencies with:

```bash
pip install -r requirements.txt
```

You may need to download some NLTK data as well (stopwords, wordnet).

## Usage

1. Place your labeled dataset CSV as `emotions.csv` in the root directory.
2. Run the main script:

```bash
python emo.py
```

The script will:
- Preprocess and clean your text data.
- Vectorize tweets using TF-IDF.
- Train and evaluate SVM classifiers with different kernels.
- Output accuracy and reports, and save each trained model as `svm_model_<kernel>.joblib`.

## Results

The script prints the accuracy for each kernel and highlights the best-performing kernel on your dataset.

## Notes

- Make sure your dataset CSV has columns named `text` (tweet text) and `label` (emotion category).
- The sample size per label is balanced to 7000 entries if enough data is available.

## License

[MIT](LICENSE)
