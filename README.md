# Customer Churn Prediction using PySpark

## Project Overview

This project is part of my **Data Scientist Nanodegree** and focuses on predicting customer churn for a fictitious music streaming service using **Apache Spark**. The goal is to apply data science techniques to analyze user activity and predict the likelihood of churn. This project follows the full machine learning pipeline, including **data understanding**, **data cleaning**, **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **modeling**, and **evaluation**. 

The project leverages **PySpark** to handle large-scale data processing and machine learning, employing various algorithms to determine the best model for predicting customer churn. The best model is selected based on performance metrics, including **F1 score** and **accuracy**, with a special focus on improving predictions for churned users, who make up a smaller subset of the dataset.

*Read the article on [Medium](https://medium.com/@gracorabello/predicting-user-churn-with-spark-my-capstone-project-for-the-data-science-nanodegree-66a58e2de474)

---

## Objective

- Analyze user behavior to predict the likelihood of churn.
- Build and evaluate multiple machine learning models.
- Identify the most effective model using Cross Validation and parameter tuning.

---

## Dataset

The workspace contains a **mini subset (128MB)** of the full dataset (12GB). The data is stored in `mini_sparkify_event_data.json`. The full dataset can be processed on a Spark cluster (cloud setup instructions are provided in the Udacity Extracurricular Spark Course).

---

## Project Steps

### 1. Data Loading and Cleaning
- Load the dataset using PySpark.
- Check for missing or invalid data (e.g., records without `userId` or `sessionId`).
- Handle missing values appropriately.

### 2. Defining Churn
- **Churn is defined** by the `Cancellation Confirmation` event.
- A binary `Churn` column is created as the target label for modeling.

### 3. Exploratory Data Analysis (EDA)
- Analyze user behavior patterns.
- Compare key metrics between **churned users** (those who canceled) and **active users**.
- Visualize trends in user engagement.

### 4. Feature Engineering
Transforming and creating new features that might improve the performance of the model. Key features engineered include:

#### **Time-Based Features**
- Number of days since registration.

#### **Session-Based Features**
- Number of sessions per user.
- Average session length.
- Maximum session length.

#### **User Behavior Features**
- Total songs played.
- Total interactions (page views, song plays, etc.).
- Number of roll ads, thumbs up/down, friends added, and songs added to playlists.

---

## Models Tested

| Model                | F1 Score | Accuracy | Time     |
|----------------------|----------|----------|----------|
| Logistic Regression  | 0.674    | 0.722    | ~2m 54s  |
| Random Forest        | 0.593    | 0.694    | ~3m 49s  |
| GBTClassifier        | **0.697**| **0.708**| ~2m 9s   |
| Linear SVM           | 0.569    | 0.694    | ~3m 8s   |
| Naive Bayes          | 0.552    | 0.542    | ~1m 23s  |

---

## Evaluation Metric

Given the imbalance in the churned class, **F1 Score** was chosen as the primary metric for model selection, supported by accuracy as a secondary metric.

---

## Best Model

The **Gradient Boosted Tree Classifier (GBTClassifier)** was selected for model tuning, due to its strong performance on both F1 score and accuracy while maintaining reasonable training time.

---

## Dependencies

- **Apache Spark (PySpark)**: For distributed data processing and machine learning.
- **Python 3.x**: Primary programming language.
- **Jupyter Notebook / Local Spark Environment**: For running Spark in a local or cloud-based environment.
- **MLlib (Spark's Machine Learning Library)**: For building machine learning models.
- **Pandas, Matplotlib, Seaborn**: For data manipulation and visualization (in EDA).

---

## Future work

### Hyperparameter Tuning
While initial models were built using default parameters, further improvements can be achieved by tuning hyperparameters using Grid Search or Randomized Search over parameters such as:
maxDepth, maxIter, stepSize, subsamplingRate.

This tuning can be done using tools like CrossValidator from Spark MLlib for robust evaluation.

### Use the Full Dataset on the Cloud
To scale this solution for larger datasets: Migrate the pipeline to a cloud environment like Amazon EMR, Google Dataproc, or Databricks, which are optimized for distributed processing with Spark.

### Feature Expansion
Enrich the dataset with user demographics, listening context (e.g., device, location), or even sentiment analysis on user feedback to improve predictive performance.

---

## Acknowledgments

Udacity Data Scientist Nanodegree for the project inspiration.

---

## Contact

If you have any questions or suggestions, feel free to reach out! [Linkedin](https://www.linkedin.com/in/gracorabello/)

---
