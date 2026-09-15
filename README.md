# Bank Marketing Classification

A machine learning project for predicting whether a bank customer is likely to subscribe to a term deposit based on customer characteristics and marketing campaign information.

The project implements and compares three classification algorithms — **Decision Tree, Gaussian Naive Bayes, and K-Nearest Neighbors (KNN)** — using a complete data preprocessing, model training, and evaluation pipeline.

---

## 🚀 Project Overview

Financial institutions conduct large-scale marketing campaigns to promote products such as term deposits. Identifying customers who are more likely to respond positively can help banks improve campaign efficiency and make better data-driven decisions.

This project applies supervised machine learning to the **Bank Marketing dataset** to predict customer subscription outcomes.

The workflow covers:

**Data Exploration → Data Cleaning → Feature Engineering → Model Training → Model Evaluation → Model Comparison**

The target variable is whether a customer subscribed to a term deposit:

* **Yes** → Customer subscribed
* **No** → Customer did not subscribe

---

## 🎯 Objectives

* Explore customer and marketing campaign data.
* Identify and handle missing values and duplicate records.
* Analyse the distribution of the target variable.
* Transform categorical variables into machine-readable features.
* Scale numerical features where required.
* Build multiple classification models.
* Evaluate models using several performance metrics.
* Analyse confusion matrices and classification errors.
* Compare model performance.
* Identify the strongest-performing model for the prediction task.
* Analyse feature importance using the Decision Tree model.

---

## 📊 Dataset

The project uses the **Bank Marketing dataset**, which contains information about customers and marketing campaigns conducted by a bank.

The dataset includes customer demographic information, financial characteristics, existing products, contact details, and campaign-related variables.

### Target Variable

The prediction target is:

```text
y
```

| Value | Meaning                               |
| ----- | ------------------------------------- |
| `yes` | Customer subscribed to a term deposit |
| `no`  | Customer did not subscribe            |

### Example Features

Some of the information available in the dataset includes:

* Age
* Job
* Marital status
* Education
* Account balance
* Housing loan
* Personal loan
* Contact method
* Campaign contacts
* Previous campaign outcome
* Call duration
* Previous contacts

---

## 🧠 Machine Learning Models

### 1. Decision Tree

A Decision Tree learns a series of decision rules to classify customers into subscription and non-subscription groups.

**Why it is useful:**

* Easy to interpret
* Captures non-linear relationships
* Provides feature importance
* Suitable for both numerical and categorical information after preprocessing

---

### 2. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It provides a fast and computationally efficient baseline for binary classification.

**Why it is useful:**

* Fast training
* Computationally efficient
* Simple probabilistic approach
* Useful as a baseline model

---

### 3. K-Nearest Neighbors (KNN)

KNN classifies a customer based on the classes of its closest observations in the feature space.

The implementation uses scaled numerical representations because distance calculations are central to KNN.

**Why it is useful:**

* Simple and intuitive
* Can capture non-linear patterns
* Useful for comparing distance-based classification with tree-based and probabilistic approaches

---

# 🔄 Data Processing Pipeline

The project follows a structured preprocessing pipeline.

### Data Cleaning

* Missing-value detection
* Missing-value treatment
* Duplicate detection
* Data type inspection
* Target variable preparation

### Feature Transformation

Categorical variables are transformed using **One-Hot Encoding**.

Numerical features are standardized using **StandardScaler** where appropriate.

### Train-Test Split

The dataset is divided into training and testing subsets.

A stratified split is used to maintain the distribution of the target classes.

---

# 📈 Model Evaluation

The models are evaluated using multiple classification metrics.

### Accuracy

Measures the percentage of predictions that are classified correctly.

### Precision

Measures how many observations predicted as positive are actually positive.

### Recall

Measures how many actual positive observations are successfully identified.

### F1-Score

Combines precision and recall into a single metric and is particularly useful when the target classes are imbalanced.

### Confusion Matrix

Confusion matrices are generated to examine:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

# 📊 Results

The final performance comparison is generated automatically by the Python pipeline.

| Model                | Accuracy | Precision | Recall | F1-Score |
| -------------------- | -------: | --------: | -----: | -------: |
| Decision Tree        |        — |         — |      — |        — |
| Gaussian Naive Bayes |        — |         — |      — |        — |
| KNN                  |        — |         — |      — |        — |

> Results will be updated using the actual model outputs generated by the project.

---

# 📉 Visualizations

The project generates visual outputs for model and dataset analysis.

### Class Distribution

Shows the distribution of customers who subscribed and did not subscribe to the term deposit.

### Confusion Matrices

Separate confusion matrices are generated for:

* Decision Tree
* Gaussian Naive Bayes
* KNN

### Model Comparison

A comparison chart visualizes:

* Accuracy
* Precision
* Recall
* F1-Score

### Feature Importance

The Decision Tree model is used to analyse which features contribute most strongly to classification decisions.

---

# 📁 Project Structure

```text
Bank-Marketing-Classification/
│
├── bank_marketing_classification.py
├── README.md
├── .gitignore
│
└── outputs/
    │
    ├── preprocessed_dataset.csv
    ├── classifier_results.csv
    ├── classifier_predictions.csv
    │
    ├── class_distribution.png
    ├── decision_tree_confusion_matrix.png
    ├── naive_bayes_confusion_matrix.png
    ├── knn_confusion_matrix.png
    ├── classifier_comparison.png
    ├── decision_tree_feature_importance.png
    │
    └── model_analysis.txt
```

---

# 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Git**
* **GitHub**

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Bank-Marketing-Classification.git
```

Navigate to the project:

```bash
cd Bank-Marketing-Classification
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ Running the Project

Run the main Python script:

```bash
python bank_marketing_classification.py
```

After execution, the analysis outputs will be generated inside the:

```text
outputs/
```

directory.

---

# 💡 Key Insights

This project demonstrates how machine learning can support customer targeting in banking.

A classification model can potentially help financial institutions:

* Identify customers with a higher probability of subscribing.
* Improve marketing campaign targeting.
* Reduce unnecessary customer contacts.
* Allocate marketing resources more efficiently.
* Support data-driven customer segmentation.
* Evaluate trade-offs between precision and recall.

The comparison of multiple algorithms also demonstrates why model selection should not rely solely on accuracy. Precision, recall, F1-score, and confusion matrices provide a more complete view of classification performance.

---

# 🔍 Future Improvements

Potential improvements to the project include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Cross-validation for more robust model evaluation.
* Testing additional algorithms such as Random Forest, Logistic Regression, XGBoost, and SVM.
* Feature selection and dimensionality reduction.
* Handling class imbalance using techniques such as class weighting or SMOTE.
* Developing an interactive Streamlit dashboard.
* Adding probability-based customer targeting.
* Deploying the model as a web application or API.

---

# 📚 Skills Demonstrated

This project demonstrates practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Feature scaling
* Supervised machine learning
* Classification
* Decision Trees
* Naive Bayes
* KNN
* Model evaluation
* Confusion matrix analysis
* Data visualization
* Python
* Scikit-learn
* Git and GitHub

---

## 👨‍💻 Author

**Naveen Kumar**

BBA Finance & Marketing Analytics

Interested in **Investment Banking, Financial Analytics, Financial Modelling, Machine Learning, and Data Analytics**.
