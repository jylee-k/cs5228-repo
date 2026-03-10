# CS5228 Coursework 2 - Part 2: MCQ Explanations

## CW2-3: Ada-Boost Algorithm
**Answer: (b)**
*   **Initial Weights:** For 10 samples, the initial normalized weight $w_i = 1/10 = 0.1$.
*   **Classifier 1 Performance:**
    *   Threshold $< 172$ assigns Female ($1$), $\ge 172$ assigns Male ($2$).
    *   Errors: Sophia Loren (Height 172.7, True Gender 1, Pred 2), Ingrid Bergman (Height 175.3, True Gender 1, Pred 2).
    *   Total error $\epsilon_1 = 0.1 + 0.1 = 0.2$.
    *   Amount of Say $\alpha_1 = \frac{1}{2}\ln(\frac{1-\epsilon_1}{\epsilon_1}) = \frac{1}{2}\ln(\frac{0.8}{0.2}) = \frac{1}{2}\ln(4) \approx 0.693$.
*   **Classifier 2 Performance:**
    *   Threshold $< 60$ assigns Female ($1$), $\ge 60$ assigns Male ($2$).
    *   Errors: Kathy Parry (Weight 61, True 1, Pred 2), Sophia Loren (Weight 63.5, True 1, Pred 2), Ingrid Bergman (Weight 61.2, True 1, Pred 2).
    *   Total error $\epsilon_2 = 0.1 + 0.1 + 0.1 = 0.3$.
    *   Amount of Say $\alpha_2 = \frac{1}{2}\ln(\frac{1-\epsilon_2}{\epsilon_2}) = \frac{1}{2}\ln(\frac{0.7}{0.3}) \approx 0.423$.
*   **Selection:** Classifier 1 is selected as it has a higher "Amount of Say" (lower error).

*   **Weight Update (using Classifier 1, $\alpha \approx 0.693$):**
    *   **Unnormalized New Weights:**
        *   If Correct: $w_{new} = w_i \times e^{-\alpha} = 0.1 \times e^{-0.693} \approx 0.05$
        *   If Incorrect: $w_{new} = w_i \times e^{\alpha} = 0.1 \times e^{0.693} \approx 0.2$
    *   **Normalization:**
        *   Sum of new weights: $(8 \times 0.05) + (2 \times 0.2) = 0.4 + 0.4 = 0.8$
        *   Normalized Correct: $0.05 / 0.8 \approx \mathbf{0.063}$
        *   Normalized Incorrect: $0.2 / 0.8 = \mathbf{0.25}$
    *   **Updated Weight Array:** `[0.063, 0.063, 0.063, 0.063, 0.063, 0.063, 0.063, 0.25, 0.25, 0.063]` (Sophia and Ingrid are at indices 8 and 9).

## CW2-4: Macro and Micro Precision
**Answer: (a)**
*   **Confusion Matrix Totals:**
    *   Class 1: TP=12, FP=(1+0+0)=1
    *   Class 2: TP=11, FP=(1+2+0)=3
    *   Class 3: TP=11, FP=(0+2+2)=4
    *   Class 4: TP=7,  FP=(0+0+1)=1
*   **Precision per Class ($P = \frac{TP}{TP+FP}$):**
    *   $P_1 = 12/13 \approx 0.923$
    *   $P_2 = 11/14 \approx 0.786$
    *   $P_3 = 11/15 \approx 0.733$
    *   $P_4 = 7/8 = 0.875$
*   **Macro Precision (MAPr):** Average of individual precisions.
    *   $MAPr = (0.923 + 0.786 + 0.733 + 0.875) / 4 = 3.317 / 4 \approx 0.829$ (Closest to 0.82).
*   **Micro Precision (MIPr):** $\frac{\sum TP}{\sum TP + \sum FP}$.
    *   $\sum TP = 12+11+11+7 = 41$.
    *   $\sum FP = 1+3+4+1 = 9$.
    *   $MIPr = 41 / (41 + 9) = 41/50 = 0.82$.

## CW2-5: Apriori Association Rules
**Answer: (c)**
Based on the `cars3.csv` dataset (40 customers):
*   **Rule: Hyundai -> Kia**
    *   Support(Hyundai): 4 transactions (IDs 4, 13, 24, 35).
    *   Support(Hyundai, Kia): 4 transactions (all 4 bought both).
    *   **Confidence = 4/4 = 1.0 (100%)**.
*   **Other Rules:**
    *   Toyota -> Nissan: Confidence $\approx 0.38$.
    *   Ford -> Chevrolet: Confidence = 0.80.
    *   BMW -> Audi: Confidence = 0.75.
*   Hyundai -> Kia has the highest confidence.

## CW2-6: Gradient Boosting
**Answer: (a)**
*   **6-1:** In Gradient Boosting for regression with squared error loss $L = (y - F(x))^2$, the pseudo-residual is the negative gradient of the loss function with respect to the prediction: $-\frac{\partial L}{\partial F} = 2(y - F(x))$. Dropping the constant, the residual is $r = y - F_{m-1}(x)$.
*   **6-2:** Boosting reduces **bias** because each subsequent model specifically targets the remaining errors (residuals) of the ensemble, allowing it to capture increasingly complex patterns. However, as the number of iterations increases, the model can become overly sensitive to noise/fluctuations in the training data, leading to higher **variance** (overfitting).
