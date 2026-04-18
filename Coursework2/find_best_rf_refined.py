import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv('Coursework2/cs5228-Housing.csv')

# Preprocessing
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 1, 'semi-furnished': 2, 'furnished': 3})

X = df.drop(['price', 'class_label'], axis=1)
y = df['class_label']

# Exact split as original notebook
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=110)

def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    return (accuracy_score(y_te, y_pred), 
            precision_score(y_te, y_pred, average='macro', zero_division=0), 
            recall_score(y_te, y_pred, average='macro', zero_division=0))

# Baseline
rf_baseline = RandomForestClassifier(random_state=110, n_estimators=200, max_depth=10, class_weight='balanced')
rf_baseline.fit(X_train, y_train)
base_res = evaluate(rf_baseline, X_test, y_test)
print(f"Baseline (All Features): Accuracy: {base_res[0]:.4f}, Precision: {base_res[1]:.4f}, Recall: {base_res[2]:.4f}")

# Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9, None],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'bootstrap': [True, False]
}

print("\nStarting Grid Search with All Features...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=110),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_res = evaluate(best_rf, X_test, y_test)
print(f"\nBest RF (All Features): Accuracy: {best_res[0]:.4f}, Precision: {best_res[1]:.4f}, Recall: {best_res[2]:.4f}")
print(f"Best Params: {grid_search.best_params_}")

# Try without hotwaterheating
X_train_reduced = X_train.drop(['hotwaterheating'], axis=1)
X_test_reduced = X_test.drop(['hotwaterheating'], axis=1)

print("\nStarting Grid Search without 'hotwaterheating'...")
grid_search_reduced = GridSearchCV(
    RandomForestClassifier(random_state=110),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search_reduced.fit(X_train_reduced, y_train)

best_rf_reduced = grid_search_reduced.best_estimator_
best_res_reduced = evaluate(best_rf_reduced, X_test_reduced, y_test)
print(f"\nBest RF (Reduced): Accuracy: {best_res_reduced[0]:.4f}, Precision: {best_res_reduced[1]:.4f}, Recall: {best_res_reduced[2]:.4f}")
print(f"Best Params (Reduced): {grid_search_reduced.best_params_}")
