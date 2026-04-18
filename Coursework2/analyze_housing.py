import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('Coursework2/cs5228-Housing.csv')

# Preprocessing
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 2})
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 1, 'semi-furnished': 2, 'furnished': 3})

# Mapping class_label to numbers for correlation analysis
class_map = {'cheap': 1, 'medium': 2, 'expensive': 3, 'very expensive': 4}
df['class_label_num'] = df['class_label'].map(class_map)

# 1. Correlation Analysis
plt.figure(figsize=(12, 10))
features_for_corr = df.drop(['price', 'class_label'], axis=1)
corr_matrix = features_for_corr.corr()
print("Correlation with class_label_num:")
print(corr_matrix['class_label_num'].sort_values(ascending=False))

# 2. Hyperparameter Optimization
X = df.drop(['price', 'class_label', 'class_label_num'], axis=1)
y = df['class_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=110)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf = RandomForestClassifier(random_state=110)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate best model on test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("\nBest Model Results (Test):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

labels = ['cheap', 'medium', 'expensive', 'very expensive']
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion Matrix:")
print(cm)
