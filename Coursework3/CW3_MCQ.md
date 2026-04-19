# Coursework 3 MCQ Answers and Explanations

## CW3-Q2
**Answer:** **a**
**Explanation:** Standard community detection algorithms (like Louvain or Girvan-Newman) typically produce disjoint partitions where each node belongs to exactly one community. If a node represents a user belonging to multiple circles (e.g., family, work, friends), it should ideally be in multiple communities. To handle this, overlapping community detection methods like **Clique Percolation** should be used.

## CW3-Q3
**Answer:** **a**
**Explanation:** In user-based collaborative filtering, similarity (e.g., Cosine or Pearson) is calculated based on items co-rated by two users. If they have no items in common, the similarity score is zero. Consequently, they will not be considered neighbors, and their preferences will not influence each other's recommendations.

## CW3-Q4
**Answer:** **d**
**Explanation:** Hoeffding Trees are incremental and designed for data streams. However, when concept drift occurs (the underlying distribution changes), the model trained on old data becomes inaccurate. While sliding windows help adapt, severe or sudden drift might exceed the window's adaptive capacity. Augmenting the model with explicit drift detection mechanisms like **DDM (Drift Detection Method)** or **ADWIN (ADaptive WINdowing)** is a standard mitigation to trigger retraining or model updates.

## CW3-Q5
**Answer:** **d**
**Explanation:** 
Given: $X = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$, $y = \begin{bmatrix} 0.9 \\ 0.1 \\ 0.1 \end{bmatrix}$, $t^0 = [-2, 2.1, 1.3, 1.5]$, $\eta = 0.1$

1. **Epoch 1:**
   - $\hat{y}^0 = X t^0 = [2.9, 1.5, 4.9]^T$
   - $e^0 = \hat{y}^0 - y = [2.0, 1.4, 4.8]^T$
   - $L^1$ (Loss before update 2, calculated using $e^0$ and reported as MSE without 1/2 factor in options): $L^1 = \frac{1}{3} (2^2 + 1.4^2 + 4.8^2) = \frac{29}{3} \approx 9.666$
   - $\nabla L = \frac{1}{N} X^T e^0 = \frac{1}{3} [2.0, 6.8, 6.8, 8.2]^T = [0.666, 2.266, 2.266, 2.733]^T$
   - $t^1 = t^0 - 0.1 \nabla L = [-2.066, 1.873, 1.073, 1.226]^T$

2. **Epoch 2:**
   - $\hat{y}^1 = X t^1 = [2.106, 1.226, 4.172]^T$
   - $e^1 = \hat{y}^1 - y = [1.206, 1.126, 4.072]^T$
   - $\nabla L(t^1) = \frac{1}{N} X^T e^1 = [0.402, 1.759, 1.759, 2.134]^T$
   - $t^2 = t^1 - 0.1 \nabla L(t^1) = [-2.108, 1.697, 0.897, 1.013]^T$

Rounding to 2 decimal places: $L^1 = 9.66$, $t^2 = [-2.11, 1.7, 0.9, 1.01]$. This matches option **d**.
