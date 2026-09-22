# Subway Surfers Review Rating Prediction

A Text Mining and Natural Language Processing project for predicting Subway Surfers review ratings from Google Play Store reviews using TF-IDF, Word2Vec, Machine Learning, Deep Learning, and SMOTE.

## Overview

This project analyzes user reviews of the Subway Surfers mobile game collected from the Google Play Store.

The main objective is to build a text classification model that predicts user ratings from review text. The project covers the complete text mining pipeline, starting from data collection and exploratory analysis to text preprocessing, feature representation, machine learning, deep learning, and handling class imbalance.

The project compares two text representation techniques:

- TF-IDF
- Word2Vec

Several classification models are evaluated:

- Logistic Regression
- Random Forest
- Artificial Neural Network (ANN)

The effect of handling class imbalance using SMOTE is also investigated.

---

## Dataset

The reviews were collected from the Google Play Store using the `google_play_scraper` library.

### Data Collection

- Application: Subway Surfers
- Google Play Application ID: `com.kiloo.subwaysurf`
- Language: English
- Country: United States
- Sorting: Newest reviews
- Initial reviews collected: 3,500
- Reviews remaining after English-language filtering: 2,185

### Dataset Variables

| Variable | Description |
|---|---|
| `review` | Text of the user review |
| `rating` | User rating from 1 to 5 |

The final dataset contains 2,185 reviews with no missing values.

---

## Project Workflow

The analysis follows these main stages:

1. Data Collection
2. Exploratory Data Analysis
3. Text Preprocessing
4. Text Representation
5. Machine Learning
6. Deep Learning
7. Performance Evaluation
8. Handling Class Imbalance with SMOTE

---

## 1. Data Collection

Reviews were scraped from the Google Play Store using `google_play_scraper`.

The scraping process uses:

- English-language reviews
- United States as the country
- Newest reviews
- 3,500 initial reviews

After filtering the reviews to retain English-language content, 2,185 reviews remained.

The cleaned dataset was saved as:

```text
subway_surfers_reviews.csv
```

---

## 2. Exploratory Data Analysis

Several exploratory analyses were performed to understand the review dataset.

### Rating Distribution

The distribution of ratings is highly imbalanced:

| Rating | Number of Reviews |
|---:|---:|
| 1 | 165 |
| 2 | 37 |
| 3 | 86 |
| 4 | 212 |
| 5 | 1,685 |

Rating 5 accounts for the majority of the dataset, while ratings 2 and 3 have considerably fewer observations.

This imbalance becomes an important consideration during model development.

### WordCloud

WordClouds were generated separately for each rating category to identify dominant words.

The reviews with higher ratings tend to contain positive words such as:

- `good`
- `love`
- `fun`
- `best`
- `amazing`

Meanwhile, lower-rated reviews contain terms associated with complaints and problems, such as:

- `ads`
- `fix`
- `problem`
- `bad`

### Top Words

The most frequent words were also identified for each rating category using `Counter`.

The word `game` appears frequently across all rating categories because it is closely related to the main topic of the reviews.

### Non-Standard Word Detection

The project also detects words that are not found in the NLTK English word corpus.

This analysis identifies potential:

- Slang
- Abbreviations
- Typos
- Informal expressions
- Symbols and punctuation

The results showed that some detected words were actually valid English words or word variations that were not included in the dictionary, highlighting the limitations of dictionary-based detection.

---

## 3. Text Preprocessing

The review text was processed before being used for modeling.

The preprocessing pipeline includes:

- Lowercasing
- Removing stopwords
- Cleaning text
- Tokenization
- Slang normalization
- Lemmatization
- Part-of-speech tagging

A slang dictionary was also created to normalize common informal expressions while preserving their intended meaning.

The final processed text was used to generate the input features for the prediction models.

---

## 4. Text Representation

Two different text representation approaches were compared.

### TF-IDF

TF-IDF was implemented using:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
```

The model uses both unigram and bigram features to capture individual words as well as short word combinations.

### Word2Vec

Word2Vec was trained directly on the processed review tokens.

The main configuration includes:

```text
vector_size = 100
window = 5
min_count = 2
workers = 4
```

The word embeddings were then converted into sentence-level vectors using average word embeddings.

---

## 5. Machine Learning

The dataset was split into training and testing sets using an 80:20 split with stratification.

```text
Training set: 80%
Testing set: 20%
Random state: 42
```

Two machine learning algorithms were evaluated:

### Logistic Regression

Logistic Regression was tested using:

- Baseline model
- Hyperparameter tuning with GridSearchCV

The tuning process explored:

- `C`
- `solver`
- `class_weight`

### Random Forest

Random Forest was also evaluated using:

- Baseline model
- Hyperparameter tuning with GridSearchCV

The tuning process explored:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `class_weight`

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

---

## 6. Deep Learning

An Artificial Neural Network (ANN) was developed using Word2Vec sentence representations.

The architecture consists of:

```text
Input
  ↓
Fully Connected Layer (256)
  ↓
ReLU
  ↓
Dropout (0.4)
  ↓
Fully Connected Layer (128)
  ↓
ReLU
  ↓
Dropout (0.3)
  ↓
Output Layer (5 classes)
```

The model was trained using:

- Adam optimizer
- Learning rate: `0.0005`
- Cross-Entropy Loss
- Batch size: `64`
- 15 epochs

Hyperparameters were adjusted as part of the deep learning experiment.

---

## 7. Model Performance

The experiments compare the performance of different text representations and classification models.

The highest reported baseline accuracy was obtained by the TF-IDF + Random Forest model with an accuracy of approximately 78.26%.

For the tuned models, TF-IDF + Random Forest achieved an F1-score of approximately 72.03%.

### Model Comparison

| Representation | Model | Type | Accuracy | F1-Score |
|---|---|---|---:|---:|
| TF-IDF | Random Forest | Baseline | 78.26% | - |
| TF-IDF | Logistic Regression | Tuned | 68% | 68% |
| TF-IDF | Random Forest | Tuned | 73% | 72% |
| Word2Vec | Logistic Regression | Tuned | 77% | 67% |
| Word2Vec | Random Forest | Tuned | 77% | 67% |
| Word2Vec | Neural Network | Tuned | 77% | 67% |

The detailed classification reports are available in the notebook.

### Model Interpretation

Although several models achieved relatively high accuracy, the classification reports show that the models were strongly influenced by the imbalanced rating distribution.

In particular, rating 5 was predicted much better than the other rating classes.

Therefore, accuracy alone does not fully represent the model's ability to classify all rating categories.

Precision, recall, and F1-score should also be considered when evaluating the classification performance.

---

## 8. Handling Class Imbalance with SMOTE

Because rating 5 dominates the dataset, SMOTE (Synthetic Minority Oversampling Technique) was applied to the training data.

SMOTE was tested with:

- TF-IDF + Logistic Regression
- TF-IDF + Random Forest
- Word2Vec + Logistic Regression
- Word2Vec + Random Forest
- Word2Vec + Neural Network

The purpose of this experiment was to investigate whether balancing the training data could improve the model's ability to learn minority rating classes.

### SMOTE Results

Some reported results include:

| Representation | Model | Accuracy |
|---|---|---:|
| TF-IDF | Logistic Regression | 67.05% |
| TF-IDF | Random Forest | 74.60% |
| Word2Vec | Neural Network | 39.13% |

The results demonstrate that balancing the classes does not automatically lead to higher overall accuracy.

Instead, model performance needs to be evaluated using multiple metrics, particularly precision, recall, and F1-score for minority classes.

---

## Key Findings

The main findings from the project are:

1. The dataset is highly imbalanced, with rating 5 representing the majority of reviews.
2. Review content shows noticeable differences across rating categories.
3. TF-IDF provided an effective text representation for this dataset.
4. Random Forest performed well when combined with TF-IDF.
5. Word2Vec-based models tended to favor the majority class.
6. Increasing model complexity through an ANN did not automatically improve classification performance.
7. SMOTE changed the class distribution and affected model performance, but did not consistently improve overall accuracy.
8. Accuracy should be interpreted together with precision, recall, and F1-score because of the imbalanced rating distribution.

---

## Technologies & Libraries

### Programming Language

- Python

### Data Collection

- `google-play-scraper`

### Data Processing

- `pandas`
- `numpy`
- `re`

### Natural Language Processing

- `nltk`
- `langdetect`
- `wordcloud`
- `gensim`

### Machine Learning

- `scikit-learn`
- `imbalanced-learn`

### Deep Learning

- `PyTorch`

### Visualization

- `matplotlib`
- `seaborn`

---

## Repository Structure

```text
subway-surfers-review-rating-prediction/
│
├── 2702374756_AsyifaIzzatilIsma_code.ipynb
├── subway_surfers_reviews.csv
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/subway-surfers-review-rating-prediction.git
cd subway-surfers-review-rating-prediction
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn
pip install google-play-scraper langdetect nltk
pip install wordcloud gensim
pip install scikit-learn imbalanced-learn
pip install torch
```

### 3. Open the notebook

Open:

```text
2702374756_AsyifaIzzatilIsma_code.ipynb
```

Run the notebook sequentially from data collection through model evaluation.

---

## Conclusion

This project demonstrates an end-to-end Text Mining workflow for predicting user ratings from Subway Surfers reviews.

The analysis compares TF-IDF and Word2Vec representations across Logistic Regression, Random Forest, and Artificial Neural Network models. The experiments also investigate the effect of class imbalance and SMOTE.

The results show that the combination of TF-IDF and Random Forest provided strong performance for this dataset, while Word2Vec with average sentence embeddings was less effective at distinguishing the different rating classes.

The project highlights the importance of text representation and class distribution when developing text classification models.

---

## Author

**Asyifa Izzatil Isma**
