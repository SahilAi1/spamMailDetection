# Spam Email Dataset Documentation

This directory contains the dataset used for training and evaluating the Spam Mail Detection System.

## Dataset Overview

- **Source**: Kaggle Spam Mails Dataset (Enron Corpus by Venky73)
  - Kaggle URL: [https://www.kaggle.com/datasets/venky73/spam-mails-dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
  - Public Mirror: [https://raw.githubusercontent.com/kairess/toy-datasets/master/spam_ham_dataset.csv](https://raw.githubusercontent.com/kairess/toy-datasets/master/spam_ham_dataset.csv)
- **Primary File**: `spam_ham_dataset.csv`
- **Total Records**: 5,171 emails (4,993 unique messages after deduplication)
- **Class Distribution**:
  - **Ham (Legitimate / Not Spam)**: 3,672 emails (~71.0%)
  - **Spam**: 1,499 emails (~29.0%)

## Schema & Column Definitions

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Unnamed: 0` | Integer | Original record index from source corpus |
| `label` | String | Categorical label (`ham` or `spam`) |
| `text` | String | Raw email message body (including subject line) |
| `label_num` | Integer | Binary encoded target (`0` = Ham / Not Spam, `1` = Spam) |

## Supported Alternative Datasets

The preprocessing module (`src/data_preprocessing.py`) supports automatic schema detection and column mapping for common alternative spam datasets, including:

1. **Kaggle SMS Spam Collection (`spam.csv`)**:
   - `v1` (Label: `ham`/`spam`)
   - `v2` (Message text)
2. **Generic Email Datasets**:
   - Label columns: `label`, `v1`, `Category`, `target`, `class`, `label_num`
   - Text columns: `text`, `message`, `email`, `sms`, `v2`, `body`, `content`

## Manual Download Instructions

If the dataset is missing or you want to download a fresh copy:

1. Download `spam_ham_dataset.csv` from Kaggle or the public repository:
   ```bash
   curl -fsSL -o data/spam_ham_dataset.csv https://raw.githubusercontent.com/kairess/toy-datasets/master/spam_ham_dataset.csv
   ```
2. Or download directly from [Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset), extract the archive, and place `spam_ham_dataset.csv` in this `data/` directory.
3. Verify file integrity by running:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/spam_ham_dataset.csv'); print('Rows:', len(df))"
   ```
   Expected output: `Rows: 5171`.
