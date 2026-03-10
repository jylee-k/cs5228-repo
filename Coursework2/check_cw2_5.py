import pandas as pd

# Load dataset
df = pd.read_csv('Coursework2/cars3.csv')

# Preprocess: each row is a list of cars
transactions = []
for index, row in df.iterrows():
    # The car names start from the second column
    cars = [str(val).strip() for val in row.values[1:] if pd.notna(val) and str(val).strip() != '']
    transactions.append(cars)

def get_support_and_confidence(a, b):
    support_a = 0
    support_ab = 0
    for t in transactions:
        if a in t:
            support_a += 1
            if b in t:
                support_ab += 1
    
    confidence = support_ab / support_a if support_a > 0 else 0
    return support_a, support_ab, confidence

rules = [
    ('Toyota', 'Nissan'),
    ('Ford', 'Chevrolet'),
    ('Hyundai', 'Kia'),
    ('BMW', 'Audi')
]

for a, b in rules:
    supp_a, supp_ab, conf = get_support_and_confidence(a, b)
    print(f"Rule: {a} -> {b}")
    print(f"  Support({a}): {supp_a}")
    print(f"  Support({a}, {b}): {supp_ab}")
    print(f"  Confidence: {conf:.4f}")
