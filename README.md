# Tumor Detection Model: Simulation, Evaluation, and SHAP Interpretability

This documentation provides a detailed walkthrough of simulating medical image features, building a Random Forest classifier for tumor detection, evaluating the model, understanding feature importance, introducing label noise (misdiagnosis simulation), and interpreting model predictions using SHAP.

---

## 1. Simulating Medical Image Data

We simulate features that mimic medical images for tumor detection. Each image has six features:
- **Intensity**
- **Edge Sharpness**
- **Contrast**
- **Asymmetry**
- **Size**
- **Shape Complexity**

We simulate 300 samples: 100 healthy and 100 with tumors.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set a seed for reproducibility
np.random.seed(42)

num_samples = 300
X = np.zeros((num_samples, 6))

# Simulate healthy samples (class 0)
for i in range(150):
    X[i] = [
        np.random.normal(0.3, 0.05),  # intensity
        np.random.normal(0.2, 0.05),  # edge_sharpness
        np.random.normal(0.3, 0.05),  # contrast
        np.random.normal(0.2, 0.05),  # asymmetry
        np.random.normal(0.1, 0.02),  # size
        np.random.normal(0.2, 0.05),  # shape_complexity
    ]

# Simulate tumor samples (class 1)
for i in range(150, 300):
    X[i] = [
        np.random.normal(0.7, 0.1),
        np.random.normal(0.6, 0.1),
        np.random.normal(0.7, 0.1),
        np.random.normal(0.8, 0.1),
        np.random.normal(0.6, 0.1),
        np.random.normal(0.7, 0.1),
    ]

# Labels: 0 = No tumor, 1 = Tumor
y = np.array([0]*150 + [1]*150)

# Train Random Forest on entire dataset
model = RandomForestClassifier()
model.fit(X, y)

# Predict a new sample with tumor-like features
new_image = np.array([
    np.random.normal(0.65, 0.1),
    np.random.normal(0.6, 0.1),
    np.random.normal(0.75, 0.1),
    np.random.normal(0.85, 0.1),
    np.random.normal(0.7, 0.1),
    np.random.normal(0.75, 0.1),
]).reshape(1, -1)

prediction = model.predict(new_image)
print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")
```

### Explanation:
- Simulates plausible feature values based on domain knowledge.
- Trains a Random Forest classifier to distinguish between healthy and tumor samples.
- Predicts on a new simulated image.

---

## 2. Model Evaluation (Train-Test Split)

We evaluate the model using a test set and generate performance metrics.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation Report:\n")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Tumor"]))
```

### Explanation:
- Splits data (80% train, 20% test).
- Trains on training data and evaluates using `classification_report`, showing:
  - **Precision**: Correct positive predictions.
  - **Recall**: True positive rate.
  - **F1-score**: Balance between precision and recall.

---

## 3. Feature Importance (Model Insights)

Visualizing which features are most important for model decisions.

```python
import matplotlib.pyplot as plt

feature_names = ['Intensity', 'Edge Sharpness', 'Contrast', 'Asymmetry', 'Size', 'Shape Complexity']
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(range(len(importances)), importances[indices], color='skyblue')
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.show()
```

### Explanation:
- Identifies key features influencing tumor detection.
- Helps clinicians focus on critical aspects (e.g., asymmetry, size).

---

## 4. Misdiagnosis Simulation (Label Noise)

Simulate real-world label noise (e.g., misdiagnosis) and test model robustness.

```python
import random
from sklearn.metrics import classification_report

# Introduce noise: flip 10% of labels
y_noisy = y.copy()
num_noisy = int(0.1 * len(y_noisy))
indices_to_flip = random.sample(range(len(y_noisy)), num_noisy)
for idx in indices_to_flip:
    y_noisy[idx] = 1 - y_noisy[idx]

# Train-test split on noisy data
X_train_noise, X_test_noise, y_train_noise, y_test_noise = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Train and evaluate on noisy labels
model_noise = RandomForestClassifier()
model_noise.fit(X_train_noise, y_train_noise)

y_pred_noise = model_noise.predict(X_test_noise)

print("\n\U0001F4CA Model Evaluation with Label Noise (10% Misdiagnosis):\n")
print(classification_report(y_test_noise, y_pred_noise, target_names=["Healthy", "Tumor"]))
```

### Explanation:
- Simulates 10% label flipping to mimic diagnostic errors.
- Measures model resilience in noisy real-world data.

---

## 5. Confusion Matrix and ROC Curve

Detailed evaluation using a confusion matrix and ROC curve.

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Tumor"], yticklabels=["Healthy", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Tumor)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

### Explanation:
- **Confusion Matrix**: Shows true/false positives/negatives.
- **ROC Curve**: Visualizes trade-off between sensitivity and specificity.
- **AUC**: Summary metric; higher AUC = better classifier.

---

## 6. SHAP (Model Interpretability)

Interpret model predictions at an individual level using SHAP (SHapley Additive exPlanations).

```python
import shap

# 1. Initialize SHAP TreeExplainer
explainer = shap.Explainer(model, X_train)

# 2. Choose sample from test set
sample_image = X_test[0].reshape(1, -1)

# 3. Get SHAP values for this sample
shap_values = explainer(sample_image)

# 4. Inspect the SHAP output
print("Sample image shape:", sample_image.shape) 
print("SHAP values shape:", shap_values.values[0].shape) 

# 5. Feature names
feature_names = ['Intensity', 'Edge Sharpness', 'Contrast', 'Asymmetry', 'Size', 'Shape Complexity']

# 6. Visualize using force plot 
shap.initjs()
shap.force_plot(
    explainer.expected_value[0],      # Base value for first class
    shap_values.values[0, :, 0],      # SHAP values for first class (6 features)
    sample_image[0],                  # Feature values (6 elements)
    feature_names=feature_names,
    matplotlib=False
)
```

### Explanation:
- **SHAP** explains each prediction by attributing feature contributions.
- Helps understand **why** a tumor was detected (e.g., high asymmetry).
- Builds trust and transparency in AI-driven diagnosis.

---

## Summary
| Step                        | Purpose                                                   |
|----------------------------|-----------------------------------------------------------|
| Data Simulation            | Create realistic synthetic features for tumor detection  |
| Model Training & Evaluation| Train Random Forest, evaluate with precision, recall, etc|
| Feature Importance         | Visualize global feature relevance                       |
| Label Noise Simulation     | Test model robustness to diagnostic errors               |
| Confusion Matrix & ROC     | Detailed performance analysis                            |
| SHAP Interpretability      | Understand individual predictions                        |

---




[Google doc](https://docs.google.com/document/d/1aVXzKuAkLkYSsTUMH0BNK_1aypp_fyTE7zKsxO20W0o/edit?usp=sharing)