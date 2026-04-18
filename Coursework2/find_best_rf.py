import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('Coursework2/cs5228-Housing.csv')

# Preprocessing
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 1, 'semi-furnished': 2, 'furnished': 3})

X = df.drop(['price', 'class_label'], axis=1)
y = df['class_label']

# Split 90/10 with random_state=110
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=110, stratify=y)

# Drop hotwaterheating for tuning based on correlation analysis
X_train_reduced = X_train.drop(['hotwaterheating'], axis=1)
X_test_reduced = X_test.drop(['hotwaterheating'], axis=1)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 8, 10, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=110)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=110),
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train_reduced, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_reduced)

print(f"Best Params: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

# Compare with original baseline RF
rf_baseline = RandomForestClassifier(random_state=110, n_estimators=200, max_depth=10, class_weight='balanced')
rf_baseline.fit(X_train, y_train)
y_pred_base = rf_baseline.predict(X_test)
print(f"Baseline Test Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Baseline Test F1 (Macro): {f1_score(y_test, y_pred_base, average='macro'):.4f}")
