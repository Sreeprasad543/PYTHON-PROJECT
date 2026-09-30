# ============================================================
# NETWORK INTRUSION DETECTION USING LINEAR REGRESSION
# PCA + Linear Regression + Regression & Classification Metrics
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------

df1 = pd.read_csv(
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    low_memory=True
)

df2 = pd.read_csv(
    'Wednesday-workingHours.pcap_ISCX.csv',
    low_memory=True
)

df3 = pd.read_csv(
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    low_memory=True
)

# ------------------------------------------------------------
# 2. MERGE DATASETS
# ------------------------------------------------------------

dataset = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

# Remove spaces from column names
dataset.columns = dataset.columns.str.strip()

print("Dataset Shape:", dataset.shape)

# ------------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].copy()

# Convert features to numeric
X = X.apply(
    pd.to_numeric,
    errors='coerce'
)

# Replace infinity with NaN
X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ------------------------------------------------------------
# 4. ENCODE TARGET
# ------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nClasses:")
for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)

# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.6,       # CHANGE THIS VALUE
    random_state=0,
    stratify=y
)

# ------------------------------------------------------------
# 6. MISSING VALUE IMPUTATION
# ------------------------------------------------------------

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy='mean'
)

# Fit ONLY on training data
X_train = imputer.fit_transform(X_train)

# Apply training statistics to test data
X_test = imputer.transform(X_test)

# ------------------------------------------------------------
# 7. PCA
# ------------------------------------------------------------

from sklearn.decomposition import PCA

pca = PCA(
    n_components=5     # CHANGE THIS VALUE
)

# Fit PCA only on training data
X_train = pca.fit_transform(X_train)

# Transform test data
X_test = pca.transform(X_test)

print("\nPCA Components:", pca.n_components_)

# ------------------------------------------------------------
# 8. LINEAR REGRESSION
# ------------------------------------------------------------

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ------------------------------------------------------------
# 9. CONTINUOUS REGRESSION PREDICTION
# ------------------------------------------------------------

y_pred_regression = model.predict(X_test)

# ------------------------------------------------------------
# 10. REGRESSION PREDICTION → CLASS LABEL
# ------------------------------------------------------------

# Round continuous predictions to nearest class
y_pred_class = np.rint(
    y_pred_regression
).astype(int)

# Make sure predicted classes are within the valid range
y_pred_class = np.clip(
    y_pred_class,
    0,
    len(label_encoder.classes_) - 1
)

# ------------------------------------------------------------
# 11. REGRESSION METRICS
# ------------------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    y_test,
    y_pred_regression
)

mse = mean_squared_error(
    y_test,
    y_pred_regression
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred_regression
)

# ------------------------------------------------------------
# 12. CLASSIFICATION METRICS
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(
    y_test,
    y_pred_class
)

precision = precision_score(
    y_test,
    y_pred_class,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_class,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_class,
    average='weighted',
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred_class
)

# ------------------------------------------------------------
# 13. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print("\nREGRESSION METRICS")
print("-" * 30)

print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R2   :", round(r2, 4))

print("\nCLASSIFICATION METRICS")
print("-" * 30)

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

print("\nCONFUSION MATRIX")
print("-" * 30)
print(cm)

# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("-" * 30)

print(
    classification_report(
        y_test,
        y_pred_class,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 15. PCA EXPLAINED VARIANCE
# ------------------------------------------------------------

explained_variance = np.sum(
    pca.explained_variance_ratio_
)

print("\nPCA INFORMATION")
print("-" * 30)

print(
    "Number of Components:",
    pca.n_components_
)

print(
    "Explained Variance:",
    round(explained_variance, 4)
)

# ------------------------------------------------------------
# 16. SAMPLE ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

print("\nACTUAL VS PREDICTED (FIRST 20)")
print("-" * 30)

for i in range(min(20, len(y_test))):

    actual = label_encoder.inverse_transform(
        [y_test[i]]
    )[0]

    predicted = label_encoder.inverse_transform(
        [y_pred_class[i]]
    )[0]

    print(
        "Actual:",
        actual,
        "| Predicted:",
        predicted,
        "| Regression Output:",
        round(y_pred_regression[i], 4)
    )

# ------------------------------------------------------------
# END
# ------------------------------------------------------------