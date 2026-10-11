# News Topic Classification Using Machine Learning and Deep Learning

A comparative Natural Language Processing (NLP) project exploring how **text preprocessing choices** affect **news topic classification** using traditional machine learning and deep learning models. The experiments compare **TF-IDF + Logistic Regression**, **TF-IDF + SVD + Deep Neural Network (DNN)**, and **Word2Vec Skip-gram + recurrent neural networks (LSTM/GRU)**.

The project was developed as a **CSE440 NLP** notebook.

## Project Overview

The goal is to automatically assign news text to one of four topics:

| Topic | Example subject matter |
| --- | --- |
| Business | Companies, markets, oil prices, stocks |
| Science & Tech | Software, computers, space, technology |
| Sports | Matches, teams, leagues, tournaments |
| World News | International events, politics, governments |

The notebook describes **105,710 training records**, each consisting of a news text (`headline`) and its topic label (`topic`). Although identified as *headlines*, many entries are longer than typical headlines (approximately 38 words on average), making the task closer to short-document classification.

## Objectives

- Explore the dataset's structure, label distribution, text lengths and vocabulary.
- Investigate the impact of **no**, **extreme** and **optimal** preprocessing.
- Compare traditional machine learning with neural network approaches.
- Select model configurations using validation **macro F1-score**.
- Evaluate held-out test performance using **accuracy, macro F1-score, classification reports and confusion matrices**.

## Main Features

### Exploratory Data Analysis (EDA)

- Missing-value and duplicate inspection; duplicate removal for the EDA dataset.
- Label normalization and HTML/prefix cleaning.
- News-topic frequency and class-imbalance analysis.
- Word-count/character-count statistics and outlier analysis.
- Word frequency analysis, topic-specific keywords, bigrams and trigrams.
- Histograms, bar charts, box plots and word clouds.

### Text Preprocessing Experiments

| Strategy | Description |
| --- | --- |
| **No preprocessing** | Uses original news text as input. |
| **Extreme preprocessing** | Removes HTML and nonletters, lowercases, removes stopwords, and applies Lancaster stemming. |
| **Optimal preprocessing** | Lowercases, removes HTML/punctuation, and filters nonalphabetic and very short tokens without stemming. |

The EDA also examines filtering unusually short or long entries (outside **10–100 words**). The model pipelines use separate preprocessing variants as implemented in the notebook.

### Models

1. **Logistic Regression + TF-IDF**: unigram/bigram TF-IDF features, class-balanced logistic regression, and validation-based hyperparameter selection.
2. **DNN + TF-IDF + Truncated SVD**: TF-IDF features reduced to 200 or 300 dimensions and fed to a dense neural network with ReLU activations and dropout.
3. **RNN family + Word2Vec**: Skip-gram word embeddings used with GRU, bidirectional GRU, LSTM and bidirectional LSTM configurations, including different hidden-unit, epoch and batch-size settings.

## Results

The following are **approximate best test accuracies reported in the notebook's final written comparison**, not results independently reproduced here:

| Model family | Best preprocessing / setup reported | Accuracy |
| --- | --- | ---: |
| **Logistic Regression + TF-IDF** | Optimal preprocessing | **92.13%** |
| DNN + TF-IDF + SVD | No preprocessing | **90.33%** |
| RNN-based model + Word2Vec | Optimal preprocessing | **91.18%** |

**Key finding:** The simpler Logistic Regression + TF-IDF approach produced the strongest reported result. Aggressive text cleaning hurt performance; the notebook reports RNN accuracy dropping to approximately **58%** under extreme preprocessing.

> These numbers are transcribed from the notebook's concluding discussion. Exact metrics may vary on reruns, depending on environment, randomness, data version and code corrections.

## Project Files

```text
news-topic-classification/
├── CSE440_NLP.ipynb       # EDA, preprocessing, experiments, plots and comparisons
├── README.md              # Project documentation
└── data/                  # Create locally; not included with the notebook
    ├── Training_data_2.csv
    └── Test_data.csv
```

The original notebook expects the training and test CSVs in Google Drive as:

```python
'/content/drive/MyDrive/Copy of Training_data_2.csv'
'/content/drive/MyDrive/Copy of Test_data.csv'
```

**Dataset files are not included in this repository unless you add them separately.** Ensure the CSVs contain two columns: news text and topic label.

## Installation and Usage

### Option A — Google Colab (original workflow)

1. Open `CSE440_NLP.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `Copy of Training_data_2.csv` and `Copy of Test_data.csv` to your Google Drive's `MyDrive` directory, or update the file paths in the notebook.
3. Mount Google Drive when prompted.
4. Install any missing dependencies (listed below).
5. Run the notebook cells in order, reviewing the **Known notebook issue** below before attempting the RNN section.

### Option B — VS Code / Jupyter locally

1. Install Python and the VS Code **Python** and **Jupyter** extensions.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install the packages:

   ```bash
   python -m pip install jupyter ipykernel pandas numpy matplotlib seaborn scikit-learn nltk wordcloud gensim tensorflow
   ```

5. Open `CSE440_NLP.ipynb` and select the `.venv` Python kernel.
6. Replace the Colab-only `drive.mount(...)` cell and Google Drive CSV paths with local ones, for example:

   ```python
   df = pd.read_csv('data/Training_data_2.csv')
   df_test = pd.read_csv('data/Test_data.csv')
   ```

7. Execute the cells in order. On local Python, replace the Colab-specific `!pip install gensim` cell with the package installation above if needed.

**Compatibility note:** TensorFlow support varies by Python version and operating system. If installation fails, use a supported Python environment or run the notebook in Colab. Training the neural models can take significantly longer than Logistic Regression and may benefit from a GPU.

## Known Notebook Issue

The uploaded notebook contains an **incomplete/malformed code cell at the start of the RNN section**. The code uses `ensure_text(...)` and `train_w2v(...)`, but the beginning of that cell is missing their `def` declarations and has unexpected indentation. This must be repaired before running the RNN experiments successfully. The results above reflect the notebook's written report and should not be taken as proof that the current file runs end to end without modification.

## Technologies

- **Python, Jupyter Notebook / Google Colab**
- **pandas, NumPy** for data manipulation
- **Matplotlib, Seaborn, WordCloud** for visualization
- **NLTK** for n-grams and stemming
- **scikit-learn** for TF-IDF, Logistic Regression, SVD, splitting and evaluation
- **Gensim** for Word2Vec Skip-gram embeddings
- **TensorFlow / Keras** for DNN, LSTM and GRU models

## Conclusion

Text preprocessing significantly influences news topic classification. The reported experiments show that **moderate cleaning preserves useful context**, while aggressive stemming and token removal can reduce accuracy. For this dataset, well-tuned **TF-IDF + Logistic Regression** outperformed the tested neural models despite its relative simplicity.

## License

No license has been specified for this project. Add a `LICENSE` file if you wish to grant reuse permissions.
