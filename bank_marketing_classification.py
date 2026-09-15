

# SECTION 1 - IMPORT LIBRARIES
# ==============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2 - CONFIGURATION
# ==============================================================================
# NOTE: The CSV supplied for this assignment names the target column
# 'deposit' (yes/no), not 'y'. If you use a different CSV where the target
# is genuinely called 'y', simply change TARGET_COLUMN below.
CSV_FILE_PATH = "bank_marketing.csv"
TARGET_COLUMN = "deposit"          # class label column in the supplied CSV
POSITIVE_LABEL = "yes"             # the value that means "subscribed"
NEGATIVE_LABEL = "no"              # the value that means "did not subscribe"

RANDOM_STATE = 42
TEST_SIZE = 0.20
KNN_DEFAULT_K = 5
KNN_CANDIDATE_KS = [3, 5, 7, 9]

OUTPUT_DIR = "outputs"


# ==============================================================================
# SECTION 3 - CREATE OUTPUT DIRECTORY
# ==============================================================================
def create_output_directory(path):
    """Create the outputs folder if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    print(f"[OK] Output directory ready: '{path}/'")


# ==============================================================================
# SECTION 4 - LOAD DATASET
# ==============================================================================
def load_dataset(csv_path):
    """
    Load the CSV file with pandas.
    Prints a clear, actionable message and stops the script if the file
    cannot be found, instead of silently continuing.
    """
    if not os.path.exists(csv_path):
        print("=" * 70)
        print("ERROR: CSV file not found.")
        print(f"Expected file at: '{csv_path}'")
        print("Please place your Bank Marketing CSV file in the same folder")
        print("as this script, and make sure the filename matches exactly")
        print("(or update CSV_FILE_PATH in SECTION 2 of this script).")
        print("=" * 70)
        sys.exit(1)

    # Some Bank Marketing CSVs use ';' as a separator instead of ','.
    # sep=None with engine='python' lets pandas auto-detect the delimiter.
    df = pd.read_csv(csv_path, sep=None, engine="python")
    print(f"[OK] Dataset loaded successfully from '{csv_path}'")
    return df


# ==============================================================================
# SECTION 5 - EXPLORATORY DATA INSPECTION
# ==============================================================================
def inspect_dataset(df, target_column):
    """Print the dataset overview required by the assignment brief."""
    print("\n" + "=" * 70)
    print("SECTION 5: EXPLORATORY DATA INSPECTION")
    print("=" * 70)

    print(f"\nDataset shape (rows, columns): {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    n_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {n_duplicates}")

    # Confirm the target column actually exists before going further.
    if target_column not in df.columns:
        print("=" * 70)
        print(f"ERROR: The configured target column '{target_column}' was")
        print("NOT found in this CSV.")
        print("Available columns are:")
        print(list(df.columns))
        print("Please open SECTION 2 of this script and set TARGET_COLUMN")
        print("to the correct column name for your CSV.")
        print("=" * 70)
        sys.exit(1)

    print(f"\nUnique values of target column '{target_column}':")
    print(df[target_column].unique())

    print(f"\nClass distribution of '{target_column}':")
    print(df[target_column].value_counts())

    return n_duplicates


# ==============================================================================
# SECTION 6 - DATA CLEANING
# ==============================================================================
def clean_dataset(df):
    """
    Handle missing values and duplicate rows.

    - Numerical columns : missing values filled with the column MEDIAN
      (median is robust to outliers, which matter in columns like 'balance').
    - Categorical columns: missing values filled with the MOST FREQUENT value.
    - Duplicate rows are removed only if they are exact, full-row duplicates,
      since a fully repeated row does not add new information and is very
      unlikely to represent two genuinely different customers.

    Unusual-looking but legitimate values (e.g. pdays = -1 meaning
    "never previously contacted") are deliberately NOT altered, since they
    carry real business meaning rather than being data errors.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Missing value handling ---
    total_missing_before = df.isnull().sum().sum()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  - Filled missing values in numerical column "
                  f"'{col}' with median ({median_val}).")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  - Filled missing values in categorical column "
                  f"'{col}' with most frequent value ('{mode_val}').")

    total_missing_after = df.isnull().sum().sum()
    if total_missing_before == 0:
        print("  - No missing values were found in the dataset.")
    else:
        print(f"  - Missing values before cleaning: {total_missing_before}")
        print(f"  - Missing values after cleaning : {total_missing_after}")

    # --- Duplicate row handling ---
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        print(f"  - Removed {n_duplicates} exact duplicate row(s).")
    else:
        print("  - No duplicate rows were found; nothing removed.")

    return df


# ==============================================================================
# SECTION 7 - TARGET ENCODING
# ==============================================================================
def encode_target_and_split_xy(df, target_column, positive_label):
    """
    Convert the target column to numeric:
        no  -> 0
        yes -> 1
    (1 = positive class = customer subscribed to the term deposit)

    Then separate predictors (X) from the target (y), and drop any
    identifier-like column that should not be used for prediction.
    """
    print("\n" + "=" * 70)
    print("SECTION 7: TARGET ENCODING")
    print("=" * 70)

    df = df.copy()
    df[target_column] = df[target_column].str.strip().str.lower()
    mapping = {"no": 0, "yes": 1}
    df[target_column] = df[target_column].map(mapping)

    print("Target mapping applied: {'no': 0, 'yes': 1}")
    print(f"Positive class (1) means: customer subscribed ('{positive_label}')")

    if df[target_column].isnull().any():
        print("WARNING: Some target values could not be mapped to 0/1.")
        print("Unmapped rows have been dropped to avoid corrupting the model.")
        df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(int)

    # Identify identifier-like columns (e.g. an ID column that is unique
    # per row and carries no predictive signal). None of the standard Bank
    # Marketing columns are identifiers, but this check is kept generic so
    # the script behaves safely on slightly different CSV exports.
    identifier_like_cols = [
        col for col in df.columns
        if col != target_column and df[col].nunique() == len(df)
    ]
    if identifier_like_cols:
        print(f"  - Dropping identifier-like column(s): {identifier_like_cols}")
        print("    (Every value is unique, so it cannot generalise to new "
              "customers and would only let the model memorise rows.)")
        df = df.drop(columns=identifier_like_cols)
    else:
        print("  - No identifier-like columns were found in this dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"\nFinal predictor matrix X shape: {X.shape}")
    print(f"Final target vector y shape   : {y.shape}")

    return X, y


# ==============================================================================
# SECTION 8 - CLASS DISTRIBUTION
# ==============================================================================
def analyse_and_plot_class_distribution(y, output_dir):
    """
    Analyse and visualise how balanced (or imbalanced) the target classes
    are. This matters because a very imbalanced target can make plain
    accuracy misleading (see explanation printed below and in the report).
    """
    print("\n" + "=" * 70)
    print("SECTION 8: CLASS DISTRIBUTION / IMBALANCE CHECK")
    print("=" * 70)

    counts = y.value_counts().sort_index()
    percentages = (counts / counts.sum() * 100).round(2)

    print("Class counts (0 = no, 1 = yes):")
    print(counts)
    print("\nClass percentages:")
    print(percentages)

    ratio = counts.max() / counts.min()
    print(f"\nImbalance ratio (majority/minority): {ratio:.2f} : 1")
    if ratio >= 1.5:
        print("This indicates a moderate class imbalance. Accuracy alone "
              "can look high just by favouring the majority class, so "
              "precision, recall and F1-score are reported for every model.")
    else:
        print("The classes are reasonably balanced, but precision, recall "
              "and F1-score are still reported for a complete evaluation.")

    plt.figure(figsize=(6, 5))
    labels = ["No (0)", "Yes (1)"]
    sns.barplot(x=labels, y=counts.values, hue=labels, palette="Blues_d", legend=False)
    plt.title("Class Distribution of Target Variable (Term Deposit Subscription)")
    plt.xlabel("Class")
    plt.ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()
    print(f"[OK] Saved chart: {output_dir}/class_distribution.png")

    return ratio


# ==============================================================================
# SECTION 9 - TRAIN / TEST SPLIT
# ==============================================================================
def split_train_test(X, y):
    """
    Split into training and test sets using stratified sampling.

    Stratification is important here because the target classes are not
    perfectly balanced. Without stratify=y, a random split could by chance
    place too many or too few 'yes' cases in the test set, which would make
    the evaluation metrics unreliable and not representative of the true
    class distribution.
    """
    print("\n" + "=" * 70)
    print("SECTION 9: TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Test set size    : {X_test.shape[0]} rows")
    print("\nTraining set class distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nTest set class distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 10 - IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==============================================================================
def identify_feature_types(X):
    """Automatically split predictor columns into numerical vs categorical."""
    print("\n" + "=" * 70)
    print("SECTION 10: FEATURE TYPE IDENTIFICATION")
    print("=" * 70)

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    return numerical_cols, categorical_cols


# ==============================================================================
# SECTION 11 - PREPROCESSING PIPELINE
# ==============================================================================
def build_preprocessing_pipeline(numerical_cols, categorical_cols, dense_output=False):
    """
    Build a ColumnTransformer that:
      - Scales numerical columns with StandardScaler
        (important for distance-based models like KNN, and harmless for
        Decision Tree / Naive Bayes).
      - One-hot encodes categorical columns with handle_unknown="ignore"
        (so the model does not crash if the test set contains a category
        that was not seen during training).

    This transformer is only ever *fit* on the training data (inside a
    Pipeline together with each classifier), and only *applied* (transform)
    to the test data. This prevents data leakage: no information from the
    test set is allowed to influence how the scaler or encoder are set up.

    dense_output=True forces the OneHotEncoder to output a dense array,
    which GaussianNB requires (it cannot work with a sparse matrix).
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor


def get_transformed_feature_names(preprocessor, numerical_cols, categorical_cols):
    """
    Retrieve human-readable feature names after the ColumnTransformer has
    been fitted (needed later for the Decision Tree feature importance
    chart, since one-hot encoding expands each categorical column into
    several new binary columns).
    """
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_ohe.get_feature_names_out(categorical_cols))
    return numerical_cols + cat_feature_names


# ==============================================================================
# SECTION 11B - OPTIONAL FEATURE SELECTION
# ==============================================================================
def select_features_if_appropriate(X_train_transformed, y_train, feature_names, k="all"):
    """
    Optionally reduce the transformed feature matrix using SelectKBest
    (ANOVA F-test), fitted ONLY on the training data to avoid leakage.

    For this dataset, one-hot encoding produces a manageable number of
    columns (well under 100), and every remaining feature is a plausible,
    business-meaningful predictor (age, job, balance, contact method,
    etc.). There is no strong reason to discard any of them, so full
    feature selection is not necessary for a BBA-level project - it would
    add complexity without a clear benefit. SelectKBest is still fitted
    here (with k='all') purely to report each feature's statistical
    relevance to the target, without actually removing any columns.
    """
    print("\n" + "=" * 70)
    print("SECTION 11B: FEATURE SELECTION ANALYSIS")
    print("=" * 70)

    n_features = X_train_transformed.shape[1]
    print(f"Total transformed features available: {n_features}")
    print("Decision: feature selection (removal) is NOT applied, because "
          "the transformed feature count is small and every feature is a "
          "meaningful business attribute. Instead, SelectKBest is used "
          "below only to RANK feature relevance for reporting purposes.")

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_train_transformed, y_train)

    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "F_Score": selector.scores_
    }).sort_values("F_Score", ascending=False)

    print("\nTop 10 most statistically relevant features (ANOVA F-test):")
    print(scores_df.head(10).to_string(index=False))

    return scores_df


# ==============================================================================
# SECTION 12-14 - CLASSIFIER TRAINING
# ==============================================================================
def train_decision_tree(preprocessor, X_train, y_train):
    """
    Decision Tree Classifier.

    Parameters chosen deliberately to keep the model simple and readable:
      - max_depth=6        : limits tree depth to reduce overfitting and
                              keep the tree interpretable for a viva.
      - min_samples_leaf=20: requires a reasonably sized group of customers
                              per leaf, avoiding tiny, noise-driven splits.
      - random_state=42    : reproducibility.
    """
    print("\n[Training] Decision Tree ...")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)
    print("[OK] Decision Tree trained.")
    return model


def train_naive_bayes(preprocessor_dense, X_train, y_train):
    """
    Gaussian Naive Bayes.

    GaussianNB requires a dense numeric matrix, so the preprocessing
    pipeline passed in here is configured to output a dense array
    (sparse_output=False in the OneHotEncoder) instead of the sparse
    matrix normally used to save memory.
    """
    print("\n[Training] Naive Bayes ...")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor_dense),
        ("classifier", GaussianNB())
    ])
    model.fit(X_train, y_train)
    print("[OK] Naive Bayes trained.")
    return model


def select_best_k_via_cv(preprocessor, X_train, y_train, candidate_ks):
    """
    Test a small range of K values using 5-fold cross-validation on the
    TRAINING data only (the test set is never touched here), and select
    the K with the highest mean F1-score.
    """
    print("\n[KNN] Selecting best K via cross-validation on training data...")
    best_k = None
    best_score = -1
    cv_results = {}

    for k in candidate_ks:
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=k))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        mean_score = scores.mean()
        cv_results[k] = mean_score
        print(f"  K={k}: mean CV F1-score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"[OK] Best K selected: {best_k} (CV F1-score = {best_score:.4f})")
    return best_k, cv_results


def train_knn(preprocessor, X_train, y_train, k):
    """
    K-Nearest Neighbors Classifier.

    KNN classifies a point based on the majority class among its nearest
    neighbours in feature space, using distance calculations. If features
    are left on very different scales (e.g. 'balance' in thousands vs
    'age' in tens), the larger-scale feature would dominate the distance
    calculation and distort the results. StandardScaler (inside the
    preprocessing pipeline) puts every numerical feature on a comparable
    scale, which is essential for KNN to work correctly.
    """
    print(f"\n[Training] KNN (K={k}) ...")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    print("[OK] KNN trained.")
    return model


# ==============================================================================
# SECTION 15 - EVALUATION METRICS
# ==============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculate Accuracy, Precision, Recall and F1-score for the positive
    class (1 = 'yes' = customer subscribed), and print the full
    classification report.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

    metrics = {
        "Classifier": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }
    return metrics, y_pred


# ==============================================================================
# SECTION 16 - CONFUSION MATRICES
# ==============================================================================
def plot_confusion_matrix(y_test, y_pred, model_name, filename, output_dir):
    """Plot and save a clearly labelled confusion matrix for one classifier."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"[OK] Saved chart: {output_dir}/{filename}")


# ==============================================================================
# SECTION 17 - MODEL COMPARISON VISUALISATION
# ==============================================================================
def plot_model_comparison(results_df, output_dir):
    """
    Create a grouped bar chart comparing Accuracy, Precision, Recall and
    F1-score across all three classifiers, generated entirely from the
    results DataFrame (no hard-coded numbers).
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score"]
    x = np.arange(len(results_df["Classifier"]))
    width = 0.2

    plt.figure(figsize=(9, 6))
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, results_df[metric], width, label=metric)

    plt.xticks(x + width * 1.5, results_df["Classifier"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Classifier Performance Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classifier_comparison.png"), dpi=150)
    plt.close()
    print(f"[OK] Saved chart: {output_dir}/classifier_comparison.png")


# ==============================================================================
# SECTION 18 - FEATURE IMPORTANCE (DECISION TREE)
# ==============================================================================
def plot_decision_tree_feature_importance(dt_model, feature_names, output_dir, top_n=15):
    """
    Plot the top N most important features according to the trained
    Decision Tree. Importance here reflects how much each feature reduced
    impurity across the tree's splits - it shows what the tree found
    useful for separating classes, but it is NOT a causal explanation of
    why customers subscribe.
    """
    importances = dt_model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title("Decision Tree - Top Feature Importances")
    plt.xlabel("Importance (reduction in impurity)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_tree_feature_importance.png"), dpi=150)
    plt.close()
    print(f"[OK] Saved chart: {output_dir}/decision_tree_feature_importance.png")

    return importance_df


# ==============================================================================
# SECTION 19 - AUTOMATICALLY GENERATED ANALYSIS
# ==============================================================================
def determine_best_classifier(results_df):
    """
    Automatically determine the best classifier.

    Primary criterion : highest F1-score (balances precision and recall,
    which matters because the target class is not perfectly balanced).
    Accuracy, precision and recall are also reported so the choice can be
    sanity-checked, not just based on one number.
    """
    best_row = results_df.sort_values("F1_Score", ascending=False).iloc[0]
    best_name = best_row["Classifier"]

    explanation = (
        f"'{best_name}' achieved the highest F1-score "
        f"({best_row['F1_Score']:.4f}) among the three classifiers, "
        f"indicating the strongest balance between precision "
        f"({best_row['Precision']:.4f}) and recall ({best_row['Recall']:.4f}) "
        f"for identifying customers who subscribed to the term deposit. "
        f"Its accuracy was {best_row['Accuracy']:.4f}."
    )
    return best_name, explanation


def build_model_analysis_text(
    df_shape, n_duplicates_removed, class_ratio, results_df,
    best_name, best_explanation, feature_importance_df
):
    """Assemble the full text-report content required for outputs/model_analysis.txt."""
    lines = []
    lines.append("BANK MARKETING - TERM DEPOSIT CLASSIFICATION - MODEL ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset size (after cleaning): {df_shape[0]} rows, {df_shape[1]} columns")
    lines.append(f"Duplicate rows removed during cleaning: {n_duplicates_removed}")
    lines.append("")
    lines.append("PREPROCESSING SUMMARY")
    lines.append("-" * 70)
    lines.append("- Missing values: numerical columns imputed with median, "
                  "categorical columns imputed with most frequent value.")
    lines.append("- Categorical variables encoded with OneHotEncoder "
                  "(handle_unknown='ignore').")
    lines.append("- Numerical variables standardised with StandardScaler.")
    lines.append("- Preprocessing was fitted only on the training set to "
                  "avoid data leakage into the test set.")
    lines.append("- Target encoded as: no -> 0, yes -> 1.")
    lines.append("")
    lines.append("CLASS DISTRIBUTION")
    lines.append("-" * 70)
    lines.append(f"Majority-to-minority class ratio: {class_ratio:.2f} : 1")
    lines.append("")
    lines.append("MODEL RESULTS")
    lines.append("-" * 70)
    lines.append(results_df.to_string(index=False))
    lines.append("")
    lines.append("BEST CLASSIFIER")
    lines.append("-" * 70)
    lines.append(f"Best Classifier: {best_name}")
    lines.append(best_explanation)
    lines.append("")
    lines.append("TOP FEATURES (Decision Tree importance)")
    lines.append("-" * 70)
    lines.append(feature_importance_df.head(10).to_string(index=False))
    lines.append("")
    lines.append("IMPORTANT OBSERVATIONS")
    lines.append("-" * 70)
    lines.append("- The 'duration' column (last call duration) is known only")
    lines.append("  AFTER a call takes place, so a model using it cannot be")
    lines.append("  used to decide who to call BEFORE the call happens. It is")
    lines.append("  kept in this academic exercise because the assignment asks")
    lines.append("  for a general classification comparison, but this caveat")
    lines.append("  should be mentioned when discussing real-world deployment.")
    lines.append("- Decision Tree feature importance reflects contribution to")
    lines.append("  the tree's splits, not proof of a causal relationship.")
    lines.append("- Accuracy alone can be misleading when classes are")
    lines.append("  imbalanced, which is why precision, recall and F1-score")
    lines.append("  were reported for every model.")

    return "\n".join(lines)


# ==============================================================================
# SECTION 20/21 - MAIN PROGRAM (OUTPUT SAVING + FINAL SUMMARY)
# ==============================================================================
def main():
    create_output_directory(OUTPUT_DIR)

    # ---- Load & inspect ----
    df_raw = load_dataset(CSV_FILE_PATH)
    inspect_dataset(df_raw, TARGET_COLUMN)

    # ---- Clean ----
    df_clean = clean_dataset(df_raw)
    n_duplicates_removed = len(df_raw) - len(df_clean)

    # Save the cleaned, human-readable dataset (target still text at this
    # point would be nice for readability, but per the assignment we save
    # AFTER target encoding so the encoding step is transparent too).

    # ---- Target encoding & X/y split ----
    X, y = encode_target_and_split_xy(df_clean, TARGET_COLUMN, POSITIVE_LABEL)

    # Save the cleaned + target-encoded dataset for submission
    # (this is the human-readable cleaned dataset, NOT the one-hot
    # encoded / scaled model matrix used internally for training).
    preprocessed_dataset = X.copy()
    preprocessed_dataset[TARGET_COLUMN] = y.values
    preprocessed_dataset.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_dataset.csv"), index=False)
    print(f"\n[OK] Saved cleaned dataset: {OUTPUT_DIR}/preprocessed_dataset.csv")
    print("     (Note: one-hot encoding and scaling happen inside the")
    print("     scikit-learn pipeline at training time, not in this file,")
    print("     so this CSV stays human-readable for submission.)")

    # ---- Class distribution ----
    class_ratio = analyse_and_plot_class_distribution(y, OUTPUT_DIR)

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # ---- Feature types ----
    numerical_cols, categorical_cols = identify_feature_types(X)

    # ---- Preprocessing pipelines ----
    # Sparse version (fine for Decision Tree and KNN)
    preprocessor_sparse = build_preprocessing_pipeline(numerical_cols, categorical_cols, dense_output=False)
    # Dense version (required for GaussianNB)
    preprocessor_dense = build_preprocessing_pipeline(numerical_cols, categorical_cols, dense_output=True)

    # Fit a copy of the dense preprocessor on the training data just to
    # extract transformed feature names and run the feature-selection
    # relevance report (this does not leak test data - it is fit on
    # X_train only).
    print("\n" + "=" * 70)
    print("SECTION 11: PREPROCESSING PIPELINE")
    print("=" * 70)
    print("Preprocessing = ColumnTransformer(StandardScaler on numerical "
          "columns + OneHotEncoder on categorical columns), fitted only "
          "on the training data inside each model's Pipeline.")

    fitted_preprocessor_for_names = build_preprocessing_pipeline(
        numerical_cols, categorical_cols, dense_output=True
    )
    X_train_transformed = fitted_preprocessor_for_names.fit_transform(X_train, y_train)
    transformed_feature_names = get_transformed_feature_names(
        fitted_preprocessor_for_names, numerical_cols, categorical_cols
    )

    # ---- Optional feature selection / relevance report ----
    feature_scores_df = select_features_if_appropriate(
        X_train_transformed, y_train, transformed_feature_names
    )

    # ---- Train classifiers ----
    print("\n" + "=" * 70)
    print("SECTIONS 12-14: TRAINING CLASSIFIERS")
    print("=" * 70)

    dt_model = train_decision_tree(preprocessor_sparse, X_train, y_train)
    nb_model = train_naive_bayes(preprocessor_dense, X_train, y_train)

    best_k, cv_results = select_best_k_via_cv(preprocessor_sparse, X_train, y_train, KNN_CANDIDATE_KS)
    knn_model = train_knn(preprocessor_sparse, X_train, y_train, best_k)

    # ---- Evaluate ----
    print("\n" + "=" * 70)
    print("SECTION 15: MODEL EVALUATION")
    print("=" * 70)

    dt_metrics, dt_pred = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    nb_metrics, nb_pred = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    knn_metrics, knn_pred = evaluate_model(knn_model, X_test, y_test, f"KNN (K={best_k})")

    results_df = pd.DataFrame([dt_metrics, nb_metrics, knn_metrics])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_results.csv"), index=False)
    print(f"\n[OK] Saved: {OUTPUT_DIR}/classifier_results.csv")
    print("\nResults summary:")
    print(results_df.to_string(index=False))

    # Save all predictions together for transparency / submission
    predictions_df = X_test.copy()
    predictions_df["Actual"] = y_test.values
    predictions_df["Predicted_DecisionTree"] = dt_pred
    predictions_df["Predicted_NaiveBayes"] = nb_pred
    predictions_df["Predicted_KNN"] = knn_pred
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_predictions.csv"), index=False)
    print(f"[OK] Saved: {OUTPUT_DIR}/classifier_predictions.csv")

    # ---- Confusion matrices ----
    print("\n" + "=" * 70)
    print("SECTION 16: CONFUSION MATRICES")
    print("=" * 70)
    plot_confusion_matrix(y_test, dt_pred, "Decision Tree", "decision_tree_confusion_matrix.png", OUTPUT_DIR)
    plot_confusion_matrix(y_test, nb_pred, "Naive Bayes", "naive_bayes_confusion_matrix.png", OUTPUT_DIR)
    plot_confusion_matrix(y_test, knn_pred, f"KNN (K={best_k})", "knn_confusion_matrix.png", OUTPUT_DIR)

    # ---- Comparison chart ----
    print("\n" + "=" * 70)
    print("SECTION 17: MODEL COMPARISON CHART")
    print("=" * 70)
    plot_model_comparison(results_df, OUTPUT_DIR)

    # ---- Feature importance ----
    print("\n" + "=" * 70)
    print("SECTION 18: DECISION TREE FEATURE IMPORTANCE")
    print("=" * 70)
    feature_importance_df = plot_decision_tree_feature_importance(
        dt_model, transformed_feature_names, OUTPUT_DIR
    )
    print("\nTop 10 features by Decision Tree importance:")
    print(feature_importance_df.head(10).to_string(index=False))
    print("\nNote: this shows what the tree found useful for splitting the")
    print("data - it does not prove that these features CAUSE subscription.")

    # ---- Best classifier ----
    print("\n" + "=" * 70)
    print("SECTION 19: AUTOMATICALLY GENERATED BEST-MODEL ANALYSIS")
    print("=" * 70)
    best_name, best_explanation = determine_best_classifier(results_df)
    print(f"\nBest Classifier: {best_name}")
    print(best_explanation)

    # ---- Save text analysis ----
    analysis_text = build_model_analysis_text(
        df_shape=X.shape,
        n_duplicates_removed=n_duplicates_removed,
        class_ratio=class_ratio,
        results_df=results_df,
        best_name=best_name,
        best_explanation=best_explanation,
        feature_importance_df=feature_importance_df
    )
    with open(os.path.join(OUTPUT_DIR, "model_analysis.txt"), "w") as f:
        f.write(analysis_text)
    print(f"\n[OK] Saved: {OUTPUT_DIR}/model_analysis.txt")

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("SECTION 21: FINAL SUMMARY")
    print("=" * 70)
    print(f"All outputs have been saved inside the '{OUTPUT_DIR}/' folder.")
    print(f"Best performing classifier: {best_name}")
    print("Run complete.")


if __name__ == "__main__":
    main()
