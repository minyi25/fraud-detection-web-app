# Data Dictionary

This data dictionary describes the variables in the `processed_data.csv` file.

| **Variable Name**      | **Type**        | **Description**                                             | **Possible Values**                   |
|-------------------------|-----------------|-------------------------------------------------------------|---------------------------------------|
| `text`                  | Text            | Original text data.                                         | Free text (varies by sample).         |
| `is_suspicious`         | Categorical     | Target variable indicating whether the text is fraudulent.  | `0` = Non-fraudulent, `1` = Fraudulent. |
| `datasource`            | Categorical     | The source of the data indicating how the text was collected or generated.  | Example: 2_Augmented Scam. |
| `text_length`           | Numeric         | The total number of characters in the text variable.        | Integer values.                       |
| `word_count`            | Numeric         | The total number of words in the text variable.             | Integer values.                       |
| `transformed_text1`     | Text            | The very clean version of text where stopwords and unnecessary elements are removed, and text is lemmatized.          | Free text (processed).        |
| `transformed_text2`     | Text            | A variant of text prepared for Word2Vec, Doc2Vec, or GloVe, where stopwords are optionally retained.            | Free text (processed based on choice).       |
| `transformed_text3`     | Text            | A variant of text prepared for Word2Vec, Doc2Vec, or GloVe, where stopwords are optionally retained.       | Free text (processed based on choice).      |


## Notes:
- The `transformed_msg` column contains text that has been preprocessed (e.g., stopwords removed, stemming/lemmatization applied).
- The `label` column is the target variable for classification tasks.

## Dataset Information
- **File Name**: `processed_data.csv`
- **Number of Samples**: 16,161
- **Target Variable**: `is_suspicious`
- **Purpose**: The dataset is designed to evaluate text classification models using different levels of text preprocessing.

## Transformed Text Variables
- **transformed_text1**: This is the most cleaned version of the text, adhering to strict preprocessing principles (e.g., removing stopwords and applying lemmatization).
- **transformed_text2**: This version retains stopwords based on the evaluation requirements for Word2Vec, Doc2Vec, or GloVe.
- **transformed_text3**: This version keeps or removes lemmatization based on model performance adjustments.

## Evaluation
- During model evaluation, the dataset will be used to test different preprocessing combinations (transformed_text1, transformed_text2, transformed_text3) with models like Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, AdaBoost, and BERT.
- Metrics such as Accuracy, Precision, Recall, and F1 Score will be compared to decide whether to retain stopwords or lemmatization for the embeddings.

## Usage
- This dataset is suitable for experiments with multiple vectorization techniques, including Bag of Words, TF-IDF, Word2Vec, Doc2Vec, GloVe, and BM25.
- Stopword and lemmatization decisions for transformed_text2 and transformed_text3 will be fine-tuned based on their impact on the model’s performance.
