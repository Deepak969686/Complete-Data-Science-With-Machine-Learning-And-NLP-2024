# 🤖 Machine Learning — Complete Notes
### By Deepak | SKIT | README-Style Reference

---

> **Three-Section Structure:**
> - 📘 **Section 1** — ML Complete (Foundations to Advanced Pipeline)
> - 🎯 **Section 2** — Interview & Study Advanced
> - 🧪 **Section 3** — Mix: Math + Intuition + Code

---

# 📘 SECTION 1 — MACHINE LEARNING COMPLETE

---

## 📌 Table of Contents — Section 1

| # | Topic |
|---|-------|
| 1.1 | What is Machine Learning? |
| 1.2 | Types of ML |
| 1.3 | ML Workflow / Pipeline |
| 1.4 | Mathematics Foundations |
| 1.5 | Supervised Learning |
| 1.6 | Unsupervised Learning |
| 1.7 | Model Evaluation |
| 1.8 | Feature Engineering |
| 1.9 | Regularization |
| 1.10 | Ensemble Methods |

---

## 1.1 🧠 What is Machine Learning?

> **Definition:** Machine Learning is a subset of AI where systems learn patterns from data to make decisions without being explicitly programmed.

```
Traditional Programming:
  Data + Rules ──► Output

Machine Learning:
  Data + Output ──► Rules (Model)
```

### Arthur Samuel (1959):
> *"Field of study that gives computers the ability to learn without being explicitly programmed."*

### Tom Mitchell (1997) — Formal Definition:
> A computer program is said to learn from **Experience E**, with respect to **Task T** and **Performance measure P**, if its performance on T improves with E.

| Symbol | Meaning | Example (Spam Filter) |
|--------|---------|----------------------|
| T | Task | Classify email as spam/not |
| E | Experience | Labeled emails |
| P | Performance | Accuracy / F1 Score |

---

## 1.2 🗂️ Types of Machine Learning

```
Machine Learning
├── Supervised Learning       ← labeled data
│   ├── Classification        (discrete output)
│   └── Regression            (continuous output)
│
├── Unsupervised Learning     ← no labels
│   ├── Clustering
│   ├── Dimensionality Reduction
│   └── Association Rules
│
├── Semi-Supervised           ← few labels + lots of unlabeled
│
└── Reinforcement Learning    ← reward/punishment signal
    ├── Model-Based
    └── Model-Free
```

### 📊 Comparison Table

| Type | Data | Goal | Example |
|------|------|------|---------|
| Supervised | Labeled | Predict output | Spam detection |
| Unsupervised | Unlabeled | Find structure | Customer segmentation |
| Semi-Supervised | Partially labeled | Leverage both | Image tagging |
| Reinforcement | Reward signals | Maximize reward | Game playing (AlphaGo) |

---

## 1.3 🔄 ML Workflow / Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Problem    │───►│  Data        │───►│  Exploratory     │
│  Definition │    │  Collection  │    │  Data Analysis   │
└─────────────┘    └──────────────┘    └──────────────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Model      │◄───│  Feature     │◄───│  Data            │
│  Training   │    │  Engineering │    │  Preprocessing   │
└─────────────┘    └──────────────┘    └──────────────────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Model      │───►│  Model       │───►│  Deployment      │
│  Evaluation │    │  Tuning      │    │  & Monitoring    │
└─────────────┘    └──────────────┘    └──────────────────┘
```

### Each Step Explained:

| Step | What Happens | Key Tools |
|------|-------------|-----------|
| Problem Definition | Business → ML problem framing | - |
| Data Collection | Gather raw data | SQL, APIs, Web Scraping |
| EDA | Understand distributions, outliers | Pandas, Seaborn |
| Preprocessing | Clean, encode, scale | Sklearn, Pandas |
| Feature Engineering | Create/select features | Domain knowledge |
| Model Training | Fit model to training data | Sklearn, XGBoost |
| Evaluation | Measure performance | Metrics, Cross-Val |
| Tuning | Optimize hyperparameters | Optuna, GridSearch |
| Deployment | Serve predictions | Flask, FastAPI, Streamlit |

---

## 1.4 📐 Mathematics Foundations

### 1.4.1 Linear Algebra

#### Vectors
A vector is an ordered list of numbers:

```
x = [x₁, x₂, ..., xₙ]ᵀ  ← column vector
```

**Dot Product:**
```
a · b = Σᵢ aᵢbᵢ = |a||b|cos(θ)
```

**Why it matters:** Measures similarity between feature vectors.

#### Matrices
A matrix is a 2D array of numbers:

```
A = [a₁₁  a₁₂]   Shape: (m × n)
    [a₂₁  a₂₂]   m rows, n columns
```

**Matrix Multiplication:**
```
C = A × B    where Cᵢⱼ = Σₖ AᵢₖBₖⱼ
Shapes: (m×k) × (k×n) = (m×n)
```

**Transpose:**
```
Aᵀ: flip rows and columns
(AB)ᵀ = BᵀAᵀ
```

**Inverse:**
```
A⁻¹A = AA⁻¹ = I  (Identity Matrix)
Only exists for square, non-singular matrices
det(A) ≠ 0
```

#### Eigenvalues & Eigenvectors
```
Av = λv

where:
  A = square matrix
  v = eigenvector (direction unchanged by A)
  λ = eigenvalue (scaling factor)
```

> 💡 **Used in:** PCA — eigenvectors = principal components, eigenvalues = variance explained

---

### 1.4.2 Calculus for ML

#### Derivatives — Gradient Descent Foundation

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Chain Rule** (key for backpropagation):
```
d/dx[f(g(x))] = f'(g(x)) · g'(x)
```

**Partial Derivatives:**
```
∂L/∂w₁ = rate of change of Loss w.r.t. weight w₁
         (all other weights held constant)
```

#### Gradient
The gradient is a vector of all partial derivatives:
```
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]ᵀ
```

> 🎯 **Gradient points in the direction of steepest ascent.**
> We move **opposite** to gradient to minimize loss.

#### Gradient Descent Update Rule:
```
w := w - α · ∇L(w)

where:
  w   = weights (parameters)
  α   = learning rate (step size)
  ∇L  = gradient of loss w.r.t. weights
```

---

### 1.4.3 Probability & Statistics

#### Probability Basics
```
P(A) ∈ [0, 1]
P(Ω) = 1  (sample space)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

#### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)
```

#### Bayes' Theorem ⭐
```
P(A|B) = P(B|A) · P(A) / P(B)

         Likelihood × Prior
       = ─────────────────────
              Evidence
```

| Term | Meaning |
|------|---------|
| P(A\|B) | Posterior — updated belief |
| P(B\|A) | Likelihood — how probable evidence given hypothesis |
| P(A) | Prior — initial belief |
| P(B) | Marginal — normalizer |

#### Distributions

| Distribution | Formula | Use in ML |
|-------------|---------|-----------|
| Gaussian (Normal) | (1/σ√2π) exp(-(x-μ)²/2σ²) | Assumptions in LR, noise |
| Bernoulli | p^x (1-p)^(1-x) | Binary outcomes |
| Binomial | C(n,k) pᵏ(1-p)^(n-k) | Multiple trials |
| Multinomial | Extension of Binomial | Naive Bayes multi-class |

#### Key Statistics
```
Mean:     μ = (1/n) Σ xᵢ
Variance: σ² = (1/n) Σ (xᵢ - μ)²
Std Dev:  σ = √σ²
Covariance: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
Correlation: ρ = Cov(X,Y) / (σₓ σᵧ)  ∈ [-1, 1]
```

---

## 1.5 📊 Supervised Learning

### 1.5.1 Linear Regression

#### Problem Setup
```
Given: X (features), y (continuous target)
Find:  ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
     = Xw  (matrix form)
```

#### Cost Function — Mean Squared Error (MSE)
```
L(w) = (1/2m) Σᵢ (ŷᵢ - yᵢ)²
     = (1/2m) ||Xw - y||²
```

#### Closed-Form Solution (Normal Equation)
```
w* = (XᵀX)⁻¹ Xᵀy

Complexity: O(n³) — slow for large n
```

#### Gradient Descent Solution
```
∂L/∂w = (1/m) Xᵀ(Xw - y)

Update: w := w - α · (1/m) Xᵀ(Xw - y)
```

#### Assumptions of Linear Regression
1. **Linearity** — relationship between X and y is linear
2. **Independence** — observations are independent
3. **Homoscedasticity** — constant variance of residuals
4. **Normality** — residuals are normally distributed
5. **No multicollinearity** — features not highly correlated

```python
# Linear Regression — Code
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept:    {model.intercept_:.4f}")
print(f"MSE:          {mse:.4f}")
print(f"R² Score:     {r2:.4f}")
```

---

### 1.5.2 Logistic Regression

#### Motivation
Linear regression predicts continuous values. For **binary classification**, we need output ∈ [0, 1].

#### Sigmoid Function
```
σ(z) = 1 / (1 + e⁻ᶻ)

where z = w₀ + w₁x₁ + ... + wₙxₙ = Xw
```

```
σ(z) properties:
  σ(0)    = 0.5
  σ(+∞)   = 1
  σ(-∞)   = 0
  σ'(z)   = σ(z)(1 - σ(z))
```

#### Prediction
```
P(y=1|X) = σ(Xw)
ŷ = 1 if P(y=1|X) ≥ 0.5, else 0
```

#### Loss Function — Binary Cross-Entropy
```
L(w) = -(1/m) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

> ⚠️ Why not MSE for classification?
> MSE creates **non-convex** loss surface → multiple local minima → gradient descent fails to converge reliably.

#### Gradient
```
∂L/∂w = (1/m) Xᵀ(ŷ - y)  ← same form as linear regression!
```

```python
# Logistic Regression — Code
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
```

---

### 1.5.3 Decision Trees

#### How It Works
```
                [Age < 30?]
               /            \
           YES               NO
          /                    \
  [Income > 50K?]         [Credit > 700?]
   /         \              /           \
 YES          NO          YES            NO
 LOAN        DENY        LOAN           DENY
```

#### Splitting Criteria

**Gini Impurity (used in CART):**
```
Gini(S) = 1 - Σⱼ pⱼ²

where pⱼ = fraction of class j in set S
Range: [0, 0.5] — 0 = pure, 0.5 = maximum impurity
```

**Information Gain (used in ID3):**
```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) · H(Sᵥ)

H(S) = -Σⱼ pⱼ log₂(pⱼ)  ← Shannon Entropy
```

**Example:**
```
S = [+, +, +, -, -, -]  → 50/50 split
H(S) = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1.0 bit  (maximum entropy)

S = [+, +, +, +, +, +]  → pure class
H(S) = -1 log₂(1) = 0 bits  (zero entropy)
```

#### Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits tree depth → controls overfitting |
| `min_samples_split` | Min samples to split a node |
| `min_samples_leaf` | Min samples in a leaf |
| `max_features` | Features to consider per split |

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)

# Visualize
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=['0','1'],
          filled=True, ax=ax)
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
```

---

### 1.5.4 Support Vector Machines (SVM)

#### Core Idea
Find the **hyperplane** that maximizes the **margin** between classes.

```
Decision boundary: w·x + b = 0
Support vectors:   w·x + b = +1  (positive class boundary)
                   w·x + b = -1  (negative class boundary)

Margin = 2 / ||w||
```

#### Optimization Problem
```
Minimize:   (1/2)||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i
```

#### Soft Margin (C-SVM)
Allows some misclassifications:
```
Minimize:   (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

C = regularization parameter
  Large C → small margin, low bias, high variance (overfitting)
  Small C → large margin, high bias, low variance (underfitting)
```

#### Kernel Trick
Map data to higher dimensions where it becomes linearly separable:

| Kernel | Formula | Use Case |
|--------|---------|---------|
| Linear | K(x,z) = xᵀz | Linearly separable |
| RBF/Gaussian | exp(-γ\|\|x-z\|\|²) | General purpose |
| Polynomial | (xᵀz + c)ᵈ | Image classification |
| Sigmoid | tanh(αxᵀz + c) | Neural network proxy |

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requires feature scaling!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_sc, y_train)
```

---

### 1.5.5 k-Nearest Neighbors (kNN)

#### Algorithm
```
For a new point xₙₑw:
1. Compute distance to ALL training points
2. Find k nearest neighbors
3. Classification: majority vote
   Regression:     mean of neighbors
```

#### Distance Metrics

| Metric | Formula |
|--------|---------|
| Euclidean | √(Σ (xᵢ-yᵢ)²) |
| Manhattan | Σ \|xᵢ-yᵢ\| |
| Minkowski | (Σ \|xᵢ-yᵢ\|ᵖ)^(1/p) |
| Cosine | 1 - (x·y)/(‖x‖‖y‖) |

#### Choosing k
```
k too small → overfitting (noisy)
k too large → underfitting (too smooth)
k = √n is a common heuristic
Use cross-validation to pick optimal k
```

> ⚠️ **Curse of Dimensionality:** In high dimensions, all points become equidistant → kNN performance degrades. Always scale features!

---

### 1.5.6 Naive Bayes

#### Bayes' Theorem Applied
```
P(y|X) = P(X|y) · P(y) / P(X)

Naive assumption: features are conditionally independent given y
P(X|y) = Π P(xᵢ|y)

Prediction: ŷ = argmax_y P(y) Π P(xᵢ|y)
```

| Variant | Likelihood P(xᵢ\|y) | Use Case |
|---------|---------------------|---------|
| GaussianNB | Gaussian distribution | Continuous features |
| MultinomialNB | Multinomial | Text (word counts) |
| BernoulliNB | Bernoulli | Binary features |

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# For text (TF-IDF or count vectors)
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_counts, y_train)
```

---

## 1.6 🔍 Unsupervised Learning

### 1.6.1 K-Means Clustering

#### Algorithm
```
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroid = mean of assigned points
4. Repeat 2-3 until convergence (no change)
```

#### Objective Function (Inertia)
```
J = Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

where μₖ = centroid of cluster k
Minimize J over assignments and centroids
```

#### Choosing k — Elbow Method
```
Run K-Means for k = 1, 2, ..., 10
Plot Inertia vs k
"Elbow" = optimal k (diminishing returns after)
```

#### Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = avg distance to points in SAME cluster (cohesion)
b(i) = avg distance to points in NEAREST cluster (separation)
Range: [-1, 1] → 1 = perfect clustering
```

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find optimal k
inertias, silhouettes = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Fit final model
km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = km.fit_predict(X)
```

---

### 1.6.2 DBSCAN

> Density-Based Spatial Clustering of Applications with Noise

#### Key Concepts
```
Core Point:    has ≥ minPts neighbors within radius ε
Border Point:  within ε of a core point but < minPts neighbors
Noise Point:   not within ε of any core point → labeled -1
```

**Advantages over K-Means:**
- Finds arbitrary shaped clusters
- Detects outliers automatically
- Doesn't need k to be specified

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
```

---

### 1.6.3 Principal Component Analysis (PCA)

#### Intuition
PCA finds directions of **maximum variance** in data and projects it onto fewer dimensions.

#### Mathematical Steps
```
1. Standardize X → zero mean, unit variance
2. Compute covariance matrix: Σ = (1/n) XᵀX
3. Compute eigenvectors (V) and eigenvalues (λ) of Σ
4. Sort eigenvectors by eigenvalues (descending)
5. Project: Z = XV  (Z = principal components)
```

#### Explained Variance Ratio
```
EVR_k = λₖ / Σⱼ λⱼ

Cumulative EVR used to choose n_components
Rule of thumb: keep components explaining ≥ 95% variance
```

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Plot explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumvar)+1), cumvar, 'bo-')
plt.axhline(0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.title('PCA — Scree Plot')

# Apply with chosen components
pca95 = PCA(n_components=0.95)  # auto-select for 95% variance
X_reduced = pca95.fit_transform(X_scaled)
```

---

## 1.7 📏 Model Evaluation

### 1.7.1 Train/Validation/Test Split

```
Full Dataset
    │
    ├── Training Set   (60-70%) ← model learns from this
    ├── Validation Set (15-20%) ← hyperparameter tuning
    └── Test Set       (15-20%) ← final unbiased evaluation
```

> ⚠️ **Golden Rule:** Never touch the test set until final evaluation.

### 1.7.2 Cross-Validation

#### k-Fold CV
```
Dataset split into k folds:

Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]

Final score = mean of all fold scores
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 1.7.3 Classification Metrics

#### Confusion Matrix
```
                  Predicted
                  POS    NEG
Actual  POS  [  TP  |  FN  ]
        NEG  [  FP  |  TN  ]

TP = True Positive   (correctly predicted positive)
TN = True Negative   (correctly predicted negative)
FP = False Positive  (predicted positive, actually negative) ← Type I Error
FN = False Negative  (predicted negative, actually positive) ← Type II Error
```

#### Derived Metrics
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)          ← "Of all predicted positive, how many correct?"
Recall    = TP / (TP + FN)          ← "Of all actual positives, how many found?"
F1 Score  = 2 · (P · R) / (P + R)  ← harmonic mean of P and R
```

#### When to use which metric?

| Metric | Use When |
|--------|---------|
| Accuracy | Balanced classes |
| Precision | FP is costly (spam filter — don't mark legit as spam) |
| Recall | FN is costly (cancer detection — don't miss cases) |
| F1 | Imbalanced classes, balance between P and R |
| AUC-ROC | Ranking quality, threshold-independent |
| PR-AUC | Highly imbalanced datasets (better than AUC-ROC) |

#### ROC Curve
```
Plot: TPR (Recall) vs FPR at various thresholds

TPR = TP / (TP + FN)   ← y-axis
FPR = FP / (FP + TN)   ← x-axis

AUC = Area Under Curve ∈ [0, 1]
  AUC = 0.5 → random classifier
  AUC = 1.0 → perfect classifier
```

```python
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)

y_prob = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
print(classification_report(y_test, model.predict(X_test)))
```

### 1.7.4 Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | (1/n) Σ\|yᵢ - ŷᵢ\| | Robust to outliers |
| MSE | (1/n) Σ(yᵢ - ŷᵢ)² | Penalizes large errors |
| RMSE | √MSE | Same units as y |
| R² | 1 - SS_res/SS_tot | % variance explained |
| MAPE | (1/n) Σ\|(yᵢ-ŷᵢ)/yᵢ\| × 100 | % error |

---

## 1.8 🔧 Feature Engineering

### 1.8.1 Handling Missing Values

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

df.isnull().sum()  # Check missing values

# Strategy options:
# 1. Drop (if < 5% missing)
df.dropna(subset=['critical_column'])

# 2. Impute — numerical
imp_mean = SimpleImputer(strategy='mean')      # sensitive to outliers
imp_med  = SimpleImputer(strategy='median')    # robust
imp_knn  = KNNImputer(n_neighbors=5)           # best for structured data

# 3. Impute — categorical
imp_cat  = SimpleImputer(strategy='most_frequent')
```

### 1.8.2 Encoding Categorical Variables

| Method | When | Code |
|--------|------|------|
| Label Encoding | Ordinal categories | `LabelEncoder()` |
| One-Hot Encoding | Nominal, low cardinality | `pd.get_dummies()` |
| Target Encoding | High cardinality | `category_encoders.TargetEncoder()` |
| Ordinal Encoding | Ordered categories | `OrdinalEncoder()` |

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-Hot (for nominal)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_enc = ohe.fit_transform(X[['color', 'city']])

# Ordinal (for ordered)
oe = OrdinalEncoder(categories=[['Low','Medium','High']])
X_enc = oe.fit_transform(X[['education']])
```

### 1.8.3 Feature Scaling

#### Why Scale?
- Algorithms using distances (kNN, SVM, PCA) are sensitive to scale
- Gradient descent converges faster with scaled features

| Scaler | Formula | Range | Use When |
|--------|---------|-------|---------|
| StandardScaler | (x - μ) / σ | ~(-3, 3) | Normal distribution |
| MinMaxScaler | (x - min) / (max - min) | [0, 1] | Bounded, no outliers |
| RobustScaler | (x - median) / IQR | varies | Outliers present |
| Normalizer | x / ‖x‖ | unit norm | Text, sparse data |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ALWAYS fit on train, transform both train and test
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)  # ← NO fit here!
```

---

## 1.9 🛡️ Regularization

### Problem: Overfitting vs Underfitting

```
     Error
       │
       │   Training Error
  High │ ─────────────────────────────── 
       │                                  ╲  Test Error
       │                                   ╲──────────────
       │                      Test Error ╲
       │                                  ╲
  Low  │                                   ╲ 
       └─────────────────────────────────────────────────
         Low complexity         High complexity → Overfit
```

### 1.9.1 Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

Bias:     error from wrong assumptions (underfitting)
Variance: error from sensitivity to training data (overfitting)

High Bias   → underfitting → increase model complexity
High Variance → overfitting → regularize, more data, simpler model
```

### 1.9.2 L1 Regularization (Lasso)

```
L(w) = MSE + λ Σ|wⱼ|

Effect:
  - Drives some weights to EXACTLY zero → feature selection
  - Sparse model
  - Robust to irrelevant features
```

### 1.9.3 L2 Regularization (Ridge)

```
L(w) = MSE + λ Σwⱼ²

Effect:
  - Shrinks weights toward zero but NEVER exactly zero
  - All features retained
  - Better when all features are relevant
```

### 1.9.4 Elastic Net

```
L(w) = MSE + λ₁ Σ|wⱼ| + λ₂ Σwⱼ²
     = combination of L1 and L2

Best of both: feature selection + weight shrinkage
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)        # L2
lasso = Lasso(alpha=0.01)       # L1
enet  = ElasticNet(alpha=0.01, l1_ratio=0.5)  # L1+L2

# Use cross-validation to find best alpha
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

---

## 1.10 🌲 Ensemble Methods

### 1.10.1 Bagging (Bootstrap Aggregating)

```
                     Original Dataset
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Bootstrap 1        Bootstrap 2        Bootstrap 3
  (sample w/ rep)    (sample w/ rep)    (sample w/ rep)
        │                  │                  │
    Model 1             Model 2            Model 3
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    Aggregate (vote/mean)
                           │
                      Final Prediction
```

#### Random Forest = Bagging + Feature Randomness

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # grow full trees
    max_features='sqrt',    # features per split = √(total features)
    min_samples_leaf=5,
    oob_score=True,         # out-of-bag evaluation
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values(ascending=False).head(15).plot(kind='bar')
```

### 1.10.2 Boosting

```
Boosting: sequential ensemble — each model corrects previous errors

Model 1 → makes predictions → identify errors
                                    │
Model 2 → focuses on errors of Model 1 → new predictions
                                    │
Model 3 → focuses on errors of Model 2 → ...
                                    │
                              Final weighted sum
```

#### AdaBoost
```
For each round t:
1. Train weak learner hₜ on weighted data
2. Compute error: εₜ = Σᵢ wᵢ · I(hₜ(xᵢ) ≠ yᵢ)
3. Compute weight: αₜ = 0.5 · log((1-εₜ)/εₜ)
4. Update weights: wᵢ ← wᵢ · exp(-αₜ yᵢ hₜ(xᵢ))
5. Normalize weights

Final: H(x) = sign(Σₜ αₜ hₜ(x))
```

#### Gradient Boosting (GBM)
```
Fit new tree to residuals of previous ensemble:

F₀(x) = initial prediction (mean of y)
For m = 1 to M:
  rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)    ← residuals (pseudo-residuals)
  hₘ  = tree fitted to rᵢₘ
  Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)   ← η = learning rate
```

#### XGBoost (eXtreme Gradient Boosting)

Key improvements over vanilla GBM:
- **Regularization:** L1 + L2 on leaf weights
- **Second-order optimization:** uses Newton's method (Hessians)
- **Approximate split finding:** quantile-based
- **Parallel tree construction**
- **Handling missing values** automatically

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,         # L1
    reg_lambda=1.0,         # L2
    scale_pos_weight=ratio, # for imbalanced data
    eval_metric='aucpr',
    early_stopping_rounds=50,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
```

### 1.10.3 Stacking

```
Level 0 (Base Models):  RF | XGBoost | LightGBM | LogReg
                         │       │        │          │
                         └───────┴────────┴──────────┘
                                         │
                              Out-of-fold predictions
                                         │
Level 1 (Meta Model):           LogisticRegression
                                         │
                                  Final Prediction
```

```python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', XGBClassifier(n_estimators=100)),
    ('lgb', LGBMClassifier(n_estimators=100)),
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False
)
stack.fit(X_train, y_train)
```

---

# 🎯 SECTION 2 — INTERVIEW & STUDY ADVANCED

---

## 📌 Table of Contents — Section 2

| # | Topic |
|---|-------|
| 2.1 | Top ML Interview Questions |
| 2.2 | Advanced Concepts Deep Dive |
| 2.3 | Neural Networks & Deep Learning |
| 2.4 | Optimization Algorithms |
| 2.5 | Hyperparameter Tuning |
| 2.6 | Handling Imbalanced Data |
| 2.7 | Model Interpretability (SHAP) |
| 2.8 | Data Leakage |
| 2.9 | Production & MLOps Concepts |

---

## 2.1 🎤 Top ML Interview Questions

### Q1: What is the difference between Bias and Variance?

```
Bias:     How much the model's prediction differs from the truth
          on average across different datasets.
          HIGH BIAS = underfitting = too simple model

Variance: How much the model's prediction changes when trained
          on different datasets.
          HIGH VARIANCE = overfitting = too complex model

Total Error = Bias² + Variance + Irreducible Noise
```

**Analogy:**
```
Archer hitting a target:
  High Bias, Low Variance:  shots clustered AWAY from center
  Low Bias, High Variance:  shots scattered AROUND center
  High Bias, High Variance: shots scattered AND away from center
  Low Bias, Low Variance:   shots clustered AT center ← ideal!
```

---

### Q2: Why is logistic regression better than linear regression for classification?

| Reason | Explanation |
|--------|------------|
| Output range | LR output ∈ (-∞,+∞); LogReg ∈ [0,1] via sigmoid |
| Loss function | MSE → non-convex for classification; BCE → convex |
| Probabilistic | LogReg gives probabilities |
| Decision boundary | Naturally handles nonlinear boundaries with kernels |

---

### Q3: Explain the Curse of Dimensionality

```
As dimensions increase:
1. All points become equidistant (distance metric loses meaning)
2. Data becomes increasingly sparse
3. Exponentially more data needed to cover the space
4. Computational cost explodes
5. Overfitting risk increases

Volume of unit hypersphere:
  d=2:  π ≈ 3.14
  d=10: π⁵/120 ≈ 2.55 → ball occupies less of the cube!
```

**Solutions:** PCA, Feature Selection, t-SNE, UMAP

---

### Q4: What is the difference between L1 and L2 regularization?

| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| Penalty | λΣ\|w\| | λΣw² |
| Gradient | λ · sign(w) | 2λw |
| Effect on weights | Exact zeros → feature selection | Small but non-zero |
| Solution | Non-differentiable at 0 | Closed form exists |
| When | Sparse features, feature selection needed | All features informative |
| Geometric | Diamond constraint | Circular constraint |

---

### Q5: How does gradient descent work? What are its variants?

```
Vanilla GD:   Use ALL training data to compute gradient
              w := w - α · (1/m) ∇L(w)
              Slow for large datasets

Stochastic GD: Use ONE sample per update
              w := w - α · ∇L(w; xᵢ, yᵢ)
              Noisy, fast, can escape local minima

Mini-batch GD: Use batch of B samples (B = 32, 64, 128...)
              w := w - α · (1/B) ∇L(w; Xbatch)
              Best of both worlds ← used in practice
```

---

### Q6: Explain ROC-AUC vs PR-AUC

```
ROC Curve: TPR vs FPR at all thresholds
  AUC = 0.5 → random (diagonal)
  AUC = 1.0 → perfect

PR Curve: Precision vs Recall at all thresholds
  AUC = fraction of positives → baseline for random

When to prefer PR-AUC:
  ✅ Highly imbalanced classes (fraud, rare disease)
  ✅ When negatives >> positives
  ✅ When you care about the positive class

When to prefer ROC-AUC:
  ✅ Balanced classes
  ✅ When both classes matter equally
```

---

### Q7: What is data leakage? How do you prevent it?

```
Data Leakage: Information from OUTSIDE training data leaks into model,
              causing artificially inflated evaluation metrics.

Types:
1. Target Leakage:   feature created AFTER target is known
   Example: "next_day_price" used to predict today's price

2. Train-Test Contamination: preprocessing fitted on ALL data
   Example: fit StandardScaler on full dataset, then split

Prevention:
  ✅ Always fit preprocessing on TRAINING set only
  ✅ Use pipelines to prevent leakage
  ✅ Careful feature creation with respect to time
  ✅ Time-series: always split chronologically
```

```python
# ❌ WRONG - data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses test data info
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT - no leakage
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   # only transform!

# ✅ BEST - use Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)  # scaler fits only on X_train internally
```

---

### Q8: Explain Random Forest vs Gradient Boosting

| | Random Forest | Gradient Boosting |
|--|--------------|------------------|
| Type | Bagging (parallel) | Boosting (sequential) |
| Trees | Independent | Each corrects previous |
| Speed | Faster (parallelizable) | Slower |
| Overfitting | Hard to overfit | Prone to overfit |
| Key params | n_estimators, max_features | learning_rate, n_estimators |
| Noise robustness | High | Lower |
| Performance | Good baseline | Usually better accuracy |

---

### Q9: What happens if you have imbalanced classes?

```
Imbalanced Dataset: Class distribution is skewed
Example: Fraud detection — 99.9% non-fraud, 0.1% fraud

Problems:
  - Model learns to predict majority class always
  - Accuracy misleadingly high (99.9% accuracy by predicting all non-fraud!)
  - Minority class completely ignored

Solutions (in order of preference for fraud):
1. Use correct metric: F1, PR-AUC, Recall
2. Threshold tuning (lower threshold for minority class)
3. Class weights: class_weight='balanced'
4. SMOTE: oversample minority (ONLY on training data!)
5. Undersampling: RandomUnderSampler
6. Ensemble methods: BalancedRandomForest, EasyEnsemble
```

---

## 2.2 🧬 Advanced Concepts Deep Dive

### 2.2.1 Bayesian Optimization for Hyperparameter Tuning

```
Traditional Grid Search: exhaustive, O(nᵏ) combinations
Random Search: better than grid for high dims
Bayesian Optimization: builds probabilistic model of objective function

Steps:
1. Define surrogate model (Gaussian Process)
2. Maximize acquisition function to pick next point
3. Evaluate objective at that point
4. Update surrogate model
5. Repeat until budget exhausted

Acquisition Functions:
  EI (Expected Improvement): E[max(0, f(x) - f(x+))]
  UCB (Upper Confidence Bound): μ(x) + κσ(x)
```

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3,
                            scoring='average_precision').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
```

---

### 2.2.2 SMOTE — Synthetic Minority Oversampling

```
SMOTE Algorithm:
1. For each minority sample xᵢ:
2. Find k nearest minority neighbors
3. Pick random neighbor xⱼ
4. Create synthetic point: x_new = xᵢ + λ(xⱼ - xᵢ)  where λ ~ U[0,1]

⚠️ CRITICAL: Apply SMOTE ONLY to training data, NEVER to test/val!
```

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Apply only to training data
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"Before: {Counter(y_train)}")
print(f"After:  {Counter(y_train_sm)}")

# Or use imbalanced-learn pipeline
pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier())
])
```

---

### 2.2.3 Feature Importance Methods

| Method | Type | How it Works |
|--------|------|-------------|
| Impurity-based | Model-specific (RF) | Mean decrease in Gini/entropy |
| Permutation | Model-agnostic | Drop in score when feature shuffled |
| SHAP | Model-agnostic | Shapley values from game theory |
| Correlation | Statistical | Linear relationship with target |
| RFE | Wrapper | Recursively remove least important |

```python
# Permutation Importance (more reliable than impurity-based)
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_val, y_val,
                                n_repeats=10, random_state=42,
                                scoring='roc_auc')
imp = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)
```

---

## 2.3 🧠 Neural Networks & Deep Learning

### 2.3.1 The Perceptron

```
Inputs:  x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Wx + b
a = f(z)  ← activation function

f = step function → Perceptron
f = sigmoid       → Logistic Regression
f = ReLU          → Deep Learning unit
```

### 2.3.2 Feedforward Neural Network

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output
   x₁ ────────── h₁₁ ─────────── h₂₁ ─────────── ŷ
   x₂ ────────── h₁₂ ─────────── h₂₂
   x₃ ────────── h₁₃ ─────────── h₂₃
   x₄ ────────── h₁₄

Each connection has a weight.
Each neuron: a = f(Wx + b)
```

### 2.3.3 Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|---------|
| Sigmoid | 1/(1+e⁻ˣ) | (0,1) | Output (binary) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1,1) | Hidden layers (old) |
| ReLU | max(0,x) | [0,∞) | Hidden layers (default) |
| Leaky ReLU | max(0.01x, x) | (-∞,∞) | Avoids dying ReLU |
| ELU | x if x>0; α(eˣ-1) if x≤0 | (-α,∞) | Better than ReLU |
| Softmax | eˣⁱ / Σeˣʲ | (0,1) | Output (multi-class) |
| GELU | x·Φ(x) | ≈ReLU | Transformers |

**Dying ReLU Problem:**
```
ReLU(x) = 0 for x < 0
If neuron's pre-activation is always negative:
  → gradient = 0 → weight never updates → "dead neuron"
Solution: Leaky ReLU or ELU
```

### 2.3.4 Backpropagation

**Forward Pass:**
```
z¹ = W¹X + b¹
a¹ = f(z¹)
z² = W²a¹ + b²
a² = f(z²) = ŷ
L = loss(ŷ, y)
```

**Backward Pass (Chain Rule):**
```
∂L/∂W² = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂W²
∂L/∂W¹ = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹

Update:
W := W - α · ∂L/∂W
b := b - α · ∂L/∂b
```

### 2.3.5 Advanced Optimizers

| Optimizer | Key Idea | Formula |
|-----------|---------|---------|
| SGD | Basic gradient descent | w -= α·g |
| Momentum | Add velocity to updates | v = βv + g; w -= αv |
| RMSprop | Adaptive LR per param | w -= α·g/√(E[g²]+ε) |
| Adam | Momentum + RMSprop | Most used optimizer |

**Adam Update:**
```
m = β₁m + (1-β₁)g        ← first moment (momentum)
v = β₂v + (1-β₂)g²       ← second moment (RMSprop)
m̂ = m/(1-β₁ᵗ)            ← bias correction
v̂ = v/(1-β₂ᵗ)
w := w - α · m̂/(√v̂ + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001
```

### 2.3.6 Regularization in Neural Networks

```python
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),    # Batch Normalization
            nn.ReLU(),
            nn.Dropout(dropout),           # Dropout regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

**Dropout:**
```
During training: randomly zero out p fraction of neurons
During inference: scale activations by (1-p)

Effect: acts as ensemble of 2ⁿ subnetworks
Prevents co-adaptation of neurons
```

**Batch Normalization:**
```
For mini-batch B = {x₁,...,xₘ}:
μ_B = (1/m) Σ xᵢ
σ²_B = (1/m) Σ (xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
yᵢ = γx̂ᵢ + β   ← learned scale and shift

Benefits:
  ✅ Faster training
  ✅ Higher learning rates
  ✅ Less sensitive to initialization
  ✅ Acts as regularizer
```

---

## 2.4 ⚡ Optimization Algorithms

### 2.4.1 Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                       ReduceLROnPlateau,
                                       OneCycleLR)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Reduce LR when val loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                               factor=0.5, patience=5)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# One Cycle (fast.ai) — superconvergence
scheduler = OneCycleLR(optimizer, max_lr=0.01,
                        steps_per_epoch=len(train_loader),
                        epochs=num_epochs)
```

---

## 2.5 🎛️ Hyperparameter Tuning

### 2.5.1 Strategies Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|---------|
| Grid Search | Exhaustive | Exponential scaling | < 4 params |
| Random Search | Faster | May miss optima | 5-10 params |
| Bayesian (Optuna) | Smart sampling | More complex | > 5 params |
| Halving | Fast screening | Needs sufficient data | Many configs |

### 2.5.2 Key Hyperparameters by Algorithm

**XGBoost Critical Params:**
```
learning_rate:    0.01-0.3    (smaller = more trees needed)
n_estimators:     100-2000    (use early stopping)
max_depth:        3-10        (3-6 usually best)
min_child_weight: 1-10        (regularization)
subsample:        0.5-1.0     (row sampling)
colsample_bytree: 0.5-1.0     (column sampling)
reg_alpha:        0-1.0       (L1)
reg_lambda:       0-2.0       (L2)
```

**Neural Network Critical Params:**
```
learning_rate:    1e-4 to 1e-2  (most important!)
batch_size:       32, 64, 128, 256
hidden_layers:    1-5
hidden_units:     64-1024 (powers of 2)
dropout:          0.1-0.5
weight_decay:     1e-5 to 1e-3
```

---

## 2.6 ⚖️ Handling Imbalanced Data (Deep Dive)

### Complete Strategy for Fraud Detection

```python
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

# Step 1: Check imbalance
print(Counter(y_train))  # e.g., {0: 9900, 1: 100}
ratio = Counter(y_train)[0] / Counter(y_train)[1]  # 99

# Step 2: Compute class weights
weights = compute_class_weight('balanced',
                               classes=np.unique(y_train),
                               y=y_train)
class_weight_dict = dict(enumerate(weights))

# Step 3: Choose resampling strategy
# Mild imbalance (< 10:1): class weights may be enough
# Severe imbalance (> 10:1): SMOTE + class weights

# SMOTE variants
smote = SMOTE(k_neighbors=5, random_state=42)             # standard
bsmote = BorderlineSMOTE(k_neighbors=5, random_state=42)  # borderline only
adasyn = ADASYN(n_neighbors=5, random_state=42)           # adaptive density

# Hybrid
smt = SMOTETomek(random_state=42)  # oversample + clean

X_res, y_res = smote.fit_resample(X_train, y_train)

# Step 4: Threshold tuning (crucial!)
from sklearn.metrics import precision_recall_curve, fbeta_score

y_prob = model.predict_proba(X_val)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_val, y_prob)

# Find threshold maximizing F-beta (beta > 1 → recall preferred)
f_beta = [(1 + 1.2**2) * p * r / (1.2**2 * p + r + 1e-9)
          for p, r in zip(prec[:-1], rec[:-1])]
best_thresh = thresholds[np.argmax(f_beta)]

y_pred_tuned = (y_prob >= best_thresh).astype(int)
```

---

## 2.7 🔎 Model Interpretability — SHAP

### SHAP (SHapley Additive exPlanations)

#### Theory — Shapley Values from Game Theory
```
φᵢ(f) = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] · [f(S∪{i}) - f(S)]

Interpretation:
  φᵢ = average marginal contribution of feature i
       across all possible feature orderings

Properties (desirable):
  ✅ Efficiency:    Σφᵢ = f(x) - E[f(x)]
  ✅ Symmetry:      equal features get equal SHAP values
  ✅ Dummy:         unused features get φᵢ = 0
  ✅ Additivity:    ensemble SHAP = sum of individual SHAPs
```

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val)

# Global importance — summary plot
shap.summary_plot(shap_values, X_val, feature_names=feature_names)

# Local explanation — single prediction
shap.waterfall_plot(explainer(X_val)[0])

# Feature interaction
shap.dependence_plot('V14', shap_values, X_val,
                     interaction_index='V4')

# Force plot
shap.force_plot(explainer.expected_value,
                shap_values[0], X_val.iloc[0])
```

---

## 2.8 🚰 Data Leakage (Advanced)

### Types of Leakage

```
1. Feature Leakage:
   Feature contains info about target that wouldn't be
   available at prediction time.
   
   Example: "was_approved" predicting loan default
            (you know approval = likely no default)

2. Temporal Leakage:
   Using future data to predict past.
   
   Example: Using next month's sales to predict this month's fraud

3. Preprocessing Leakage:
   Fitting transformers on full dataset before split.
   
   Example: StandardScaler fit on all data → test mean leaks into train

4. Group Leakage:
   Same entity appears in both train and test.
   
   Example: Same customer in train and test for fraud detection
```

---

## 2.9 🏭 Production & MLOps Concepts

### Model Monitoring

```
Monitoring Categories:
  1. Data Drift:    Input distribution shifts over time
  2. Concept Drift: Relationship between X and y changes
  3. Performance:   Model accuracy degrades
  4. Infrastructure: Latency, throughput, errors

Detection Methods:
  Population Stability Index (PSI):
    PSI = Σ (Actual% - Expected%) × ln(Actual%/Expected%)
    PSI < 0.1  → no significant change
    PSI < 0.25 → moderate change
    PSI > 0.25 → significant drift → retrain!

  Kolmogorov-Smirnov Test:
    Compares distribution of train vs production features
```

### ML Pipeline with Sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define feature groups
num_features = ['age', 'income', 'amount']
cat_features = ['occupation', 'city']

# Preprocessing pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

# Full pipeline
full_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier())
])

full_pipe.fit(X_train, y_train)

# Save
import joblib
joblib.dump(full_pipe, 'model_pipeline.pkl')

# Load & predict
pipe = joblib.load('model_pipeline.pkl')
predictions = pipe.predict(new_data)
```

---

# 🧪 SECTION 3 — MIX: MATH + INTUITION + CODE NOTEBOOK

---

## 📌 Table of Contents — Section 3

| # | Topic |
|---|-------|
| 3.1 | Complete Math Derivations |
| 3.2 | Visual Intuitions |
| 3.3 | Complete Code Notebook |
| 3.4 | Real Project: Fraud Detection Pipeline |
| 3.5 | Common Pitfalls & Debugging |
| 3.6 | Quick Reference Cheat Sheet |

---

## 3.1 📐 Complete Math Derivations

### 3.1.1 Deriving Linear Regression Normal Equation

```
Problem: Minimize L(w) = ||Xw - y||²

Expand:
L(w) = (Xw - y)ᵀ(Xw - y)
     = wᵀXᵀXw - wᵀXᵀy - yᵀXw + yᵀy
     = wᵀXᵀXw - 2yᵀXw + yᵀy  [since wᵀXᵀy is scalar = yᵀXw]

Take gradient w.r.t. w:
∂L/∂w = 2XᵀXw - 2Xᵀy = 0

Solve:
XᵀXw = Xᵀy
w* = (XᵀX)⁻¹ Xᵀy  ✓
```

### 3.1.2 Deriving Logistic Regression Gradient

```
Log-likelihood:
ℓ(w) = Σᵢ [yᵢ log(σᵢ) + (1-yᵢ) log(1-σᵢ)]

where σᵢ = σ(wᵀxᵢ)

Gradient (one sample):
∂ℓ/∂w = [y/σ · ∂σ/∂w] + [(1-y)/(1-σ) · (-∂σ/∂w)]

Note: ∂σ/∂wᵀx = σ(1-σ)  [sigmoid derivative]
     ∂wᵀx/∂w = x

∂ℓ/∂w = [y/σ · σ(1-σ) - (1-y)/(1-σ) · σ(1-σ)] · x
       = [y(1-σ) - (1-y)σ] · x
       = [y - σ] · x
       = [y - ŷ] · x

Matrix form:
∂L/∂w = -(1/m) Xᵀ(y - ŷ) = (1/m) Xᵀ(ŷ - y)  ✓
```

### 3.1.3 Information Gain Derivation

```
We want to find split that maximizes information gain.

Entropy of set S:
H(S) = -Σ_{c∈C} p(c|S) log₂ p(c|S)

For binary classification:
H(S) = -p log₂ p - (1-p) log₂ (1-p)

Maximum at p=0.5: H = -0.5·(-1) - 0.5·(-1) = 1 bit

After split on attribute A (with values v):
H(S|A) = Σᵥ (|Sᵥ|/|S|) · H(Sᵥ)

Information Gain:
IG(S,A) = H(S) - H(S|A)

Choose A that maximizes IG(S,A)
```

### 3.1.4 Gradient Boosting Math

```
Goal: Find F*(x) = argmin_F E[L(y, F(x))]

Functional gradient descent:
F₀(x) = argmin_c Σ L(yᵢ, c) = mean(y)  [for MSE]

For m = 1, ..., M:
  rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}  ← negative gradient

  For MSE: L = (y-F)²/2
  rᵢₘ = -(yᵢ - Fₘ₋₁(xᵢ)) · (-1) = yᵢ - Fₘ₋₁(xᵢ)  ← residuals!

  Fit tree hₘ to pseudo-residuals {(xᵢ, rᵢₘ)}

  Find step size γₘ:
  γₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γhₘ(xᵢ))

  Update: Fₘ(x) = Fₘ₋₁(x) + η · γₘ · hₘ(x)

Intuition: We're doing gradient descent in FUNCTION space,
           not parameter space!
```

---

## 3.2 🎨 Visual Intuitions

### 3.2.1 Bias-Variance Visualization (Text)

```
Model Complexity →

Underfitting           Optimal              Overfitting
     │                    │                     │
─────┼────────────────────┼─────────────────────┼────
     │                    │                     │
  y  │    Training:   ────╪───────╮             │
     │                    │       ╰─────────────┤
     │    Test:      ─────╪─────────────╮       │
     │                    │             ╰───────┤
─────┼────────────────────┼─────────────────────┼────
  Low k-neighbors       Medium                High k
  High depth           depth               Low depth
```

### 3.2.2 Decision Boundary Complexity

```
Linear (LR)          RBF Kernel SVM         Decision Tree
  ──────────────      ┌─────────────┐          │   │   │
  Class A  │          │  Class A    │     ─────┤   │   ├─────
  ─────────┤──────    │      ╭──╮   │     Class│   │   │Class
  Class B  │          │    ╭─╯  ╰─╮ │      A   │   │    B
           │          │   ╰────────╯│          │   │
                      │    Class B  │
                      └─────────────┘
```

### 3.2.3 Gradient Descent Landscape

```
Loss
 │
 │  ╮                                   ← starting point
 │   ╲      local minimum
 │    ╰──╮
 │       ╰────────────╮
 │                    ╰───────╮        ← global minimum
 │                            ╰─────
 └──────────────────────────────────
                              Weights

SGD: noisy path, can escape local minima
Adam: smooth path, fast convergence
```

---

## 3.3 💻 Complete Code Notebook

### Setup & Imports

```python
# ============================================================
# ML Complete Notebook — SKIT | Deepak
# ============================================================

# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              mean_squared_error, r2_score)

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Advanced
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("✅ All imports successful")
```

### EDA Template

```python
def full_eda(df, target_col):
    """Complete EDA function"""
    print(f"{'='*60}")
    print(f"EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}\n")

    # Shape
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    # Target distribution
    print(f"\n{'─'*40}")
    print(f"TARGET DISTRIBUTION")
    vc = df[target_col].value_counts()
    print(vc)
    print(f"Imbalance Ratio: {vc.max()/vc.min():.1f}:1")

    # Data types
    print(f"\n{'─'*40}")
    print(f"DATA TYPES")
    print(df.dtypes.value_counts())

    # Missing values
    print(f"\n{'─'*40}")
    print(f"MISSING VALUES")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) > 0:
        miss_pct = (miss / len(df) * 100).round(2)
        print(pd.DataFrame({'Count': miss, 'Pct': miss_pct}))
    else:
        print("No missing values ✅")

    # Numerical stats
    print(f"\n{'─'*40}")
    print("NUMERICAL STATISTICS")
    print(df.describe().T.to_string())

    # Correlation with target
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    corr = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    print(f"\nTop Correlations with {target_col}:")
    print(corr.head(10))
```

### Model Training Template

```python
def train_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test,
                          feature_names, threshold=0.5):
    """
    Train multiple models and compare performance.
    Returns: results DataFrame, best model
    """

    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1),
            eval_metric='aucpr', random_state=42),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights='Balanced', verbose=False,
            random_seed=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Fit
        if name == 'XGBoost':
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        results[name] = {
            'ROC-AUC':  roc_auc_score(y_val, y_prob),
            'PR-AUC':   average_precision_score(y_val, y_prob),
            'F1':       f1_score(y_val, y_pred),
            'Precision': (y_pred[y_val==1] == 1).mean() if y_pred.sum() > 0 else 0,
            'Recall':   (y_pred[y_val==1] == 1).mean(),
        }
        trained[name] = model
        print(f"  ROC-AUC: {results[name]['ROC-AUC']:.4f}  "
              f"PR-AUC: {results[name]['PR-AUC']:.4f}  "
              f"F1: {results[name]['F1']:.4f}")

    results_df = pd.DataFrame(results).T.sort_values('PR-AUC', ascending=False)
    best_name  = results_df.index[0]
    best_model = trained[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(results_df.to_string())

    return results_df, best_model, trained
```

### Threshold Optimization

```python
def optimize_threshold(model, X_val, y_val, beta=1.2):
    """
    Find optimal threshold using F-beta score.
    beta > 1 → recall preferred (fraud, medical)
    beta < 1 → precision preferred
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)

    f_betas = []
    for p, r in zip(prec[:-1], rec[:-1]):
        denom = (beta**2 * p + r)
        fb = (1 + beta**2) * p * r / denom if denom > 0 else 0
        f_betas.append(fb)

    best_idx   = np.argmax(f_betas)
    best_thresh = thresholds[best_idx]
    best_fb     = f_betas[best_idx]

    print(f"Optimal Threshold: {best_thresh:.4f}")
    print(f"F-{beta} Score:    {best_fb:.4f}")
    print(f"Precision:         {prec[best_idx]:.4f}")
    print(f"Recall:            {rec[best_idx]:.4f}")

    return best_thresh

# Plot Precision-Recall curve
def plot_pr_curve(models_dict, X_val, y_val):
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        auc_pr = average_precision_score(y_val, y_prob)
        ax.plot(rec, prec, lw=2, label=f'{name} (AUC={auc_pr:.3f})')
    ax.axhline(y_val.mean(), color='navy', linestyle='--',
               label=f'Baseline ({y_val.mean():.3f})')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curves', fontsize=15)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
```

---

## 3.4 🏦 Real Project: Fraud Detection Pipeline

### Complete End-to-End Pipeline

```python
# ============================================================
# FRAUD DETECTION — COMPLETE PIPELINE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import shap
import optuna
import joblib

# ─── 1. Load Data ───────────────────────────────────────────
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Dataset shape: {X.shape}")
print(f"Fraud rate: {y.mean()*100:.4f}%")

# ─── 2. Three-way Split ─────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
# Result: 60% train, 20% val, 20% test

print(f"Train: {X_train.shape[0]:,}  |  "
      f"Val: {X_val.shape[0]:,}  |  "
      f"Test: {X_test.shape[0]:,}")

# ─── 3. Feature Engineering ─────────────────────────────────
def engineer_features(X):
    X = X.copy()
    # Interaction terms (top SHAP features)
    X['V4_V14'] = X['V4'] * X['V14']
    X['V4_sq']  = X['V4'] ** 2
    X['V14_sq'] = X['V14'] ** 2
    X['V10_V12'] = X['V10'] * X['V12']
    X['Amount_log'] = np.log1p(X['Amount'])
    return X

X_train = engineer_features(X_train)
X_val   = engineer_features(X_val)
X_test  = engineer_features(X_test)

# ─── 4. Scaling ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─── 5. SMOTE (only on train!) ──────────────────────────────
sm = SMOTE(k_neighbors=5, random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)
print(f"After SMOTE — Fraud: {y_train_sm.sum():,} | "
      f"Non-fraud: {(y_train_sm==0).sum():,}")

# ─── 6. Optuna Tuning (XGBoost) ─────────────────────────────
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cbt', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('lambda', 0.0, 2.0),
        'min_child_weight': trial.suggest_int('mcw', 1, 10),
    }
    model = xgb.XGBClassifier(
        **params, eval_metric='aucpr',
        early_stopping_rounds=30, random_state=42)
    model.fit(X_train_sm, y_train_sm,
              eval_set=[(X_val_sc, y_val)], verbose=False)
    return average_precision_score(y_val,
           model.predict_proba(X_val_sc)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60,
               show_progress_bar=True)
best_xgb_params = study.best_params

# ─── 7. Train Final Models ──────────────────────────────────
xgb_model = xgb.XGBClassifier(**best_xgb_params,
                               eval_metric='aucpr', random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                class_weight='balanced', random_state=42)
cat_model = CatBoostClassifier(iterations=500, learning_rate=0.05,
                               auto_class_weights='Balanced',
                               verbose=False, random_seed=42)

for m in [xgb_model, lgb_model, cat_model]:
    m.fit(X_train_sm, y_train_sm)

# ─── 8. Weighted Voting Ensemble ────────────────────────────
# Get val PR-AUC as weights
val_scores = [
    average_precision_score(y_val, m.predict_proba(X_val_sc)[:,1])
    for m in [xgb_model, lgb_model, cat_model]
]
weights = np.array(val_scores) / sum(val_scores)
print(f"Ensemble weights — XGB:{weights[0]:.3f} | "
      f"LGB:{weights[1]:.3f} | CAT:{weights[2]:.3f}")

# Soft voting
probs_val = np.average([
    m.predict_proba(X_val_sc)[:,1]
    for m in [xgb_model, lgb_model, cat_model]
], weights=weights, axis=0)

# ─── 9. Threshold Optimization ──────────────────────────────
prec, rec, thresholds = precision_recall_curve(y_val, probs_val)
f_betas = [(1+1.44)*p*r/((1.44*p+r)+1e-9) for p,r in zip(prec[:-1],rec[:-1])]
best_thresh = thresholds[np.argmax(f_betas)]
print(f"Optimal threshold: {best_thresh:.4f}")

# ─── 10. Final Test Evaluation ──────────────────────────────
probs_test = np.average([
    m.predict_proba(X_test_sc)[:,1]
    for m in [xgb_model, lgb_model, cat_model]
], weights=weights, axis=0)

y_pred_test = (probs_test >= best_thresh).astype(int)

print("\n" + "="*50)
print("FINAL TEST RESULTS")
print("="*50)
print(classification_report(y_test, y_pred_test,
      target_names=['Non-Fraud', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, probs_test):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, probs_test):.4f}")

# ─── 11. SHAP Explainability ────────────────────────────────
explainer = shap.TreeExplainer(xgb_model)
shap_vals  = explainer.shap_values(X_val_sc[:500])

shap.summary_plot(shap_vals, X_val.iloc[:500],
                  feature_names=X_train.columns.tolist(),
                  show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)

# ─── 12. Save ───────────────────────────────────────────────
joblib.dump({'scaler': scaler,
             'xgb': xgb_model, 'lgb': lgb_model, 'cat': cat_model,
             'weights': weights, 'threshold': best_thresh},
            'fraud_model_bundle.pkl')
print("✅ Model saved.")
```

---

## 3.5 ⚠️ Common Pitfalls & Debugging

### Pitfall 1: Fitting Scaler on Full Data (Data Leakage)
```python
# ❌ Wrong
scaler.fit(X)
X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

# ✅ Correct
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

### Pitfall 2: Applying SMOTE to Validation/Test Data
```python
# ❌ Wrong
X_res, y_res = SMOTE().fit_resample(X, y)  # includes val/test!

# ✅ Correct
X_train_sm, y_train_sm = SMOTE().fit_resample(X_train, y_train)
# X_val, X_test → untouched
```

### Pitfall 3: Using Accuracy for Imbalanced Data
```python
# ❌ Wrong metric
accuracy = model.score(X_test, y_test)  # 99.9% but detects no fraud!

# ✅ Right metrics for fraud
pr_auc = average_precision_score(y_test, y_prob)
f1     = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
```

### Pitfall 4: Tuning Threshold on Test Data
```python
# ❌ Wrong
threshold = find_best_threshold(model, X_test, y_test)  # leaks test info!

# ✅ Correct
threshold = find_best_threshold(model, X_val, y_val)  # use val set only
y_pred_test = (model.predict_proba(X_test)[:,1] >= threshold).astype(int)
```

### Pitfall 5: Forgetting Early Stopping
```python
# ❌ Wrong — may overfit
xgb.fit(X_train, y_train)

# ✅ Correct — stops when val metric stops improving
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100)
```

---

## 3.6 📋 Quick Reference Cheat Sheet

### Algorithm Selection Guide

```
Tabular Data
├── Small dataset (< 1K):       Logistic Regression, SVM
├── Medium dataset (1K-100K):   Random Forest, XGBoost
└── Large dataset (> 100K):     LightGBM, Neural Networks

Classification
├── Binary:    Logistic Regression → RF → XGBoost
├── Multi-class: Softmax LR → RF → XGBoost (one-vs-rest)
└── Imbalanced: XGBoost + SMOTE + threshold tuning

Regression
├── Linear relationship:  Linear Regression / Ridge
├── Non-linear:           RF Regressor / GBM
└── With outliers:        Huber Regression / LAD

Text Data
├── Simple:   Naive Bayes (TF-IDF)
├── Complex:  LSTM / Transformers (BERT)

Image Data:            CNNs
Time Series:           LSTM / GRU / Transformer
Anomaly Detection:     Isolation Forest / DBSCAN / Autoencoders
```

### Metric Selection Guide

```
┌─────────────────────────────────────────────────────┐
│                    METRIC SELECTOR                    │
├────────────────────┬────────────────────────────────┤
│ Situation          │ Metric                         │
├────────────────────┼────────────────────────────────┤
│ Balanced classes   │ Accuracy, F1                   │
│ Imbalanced classes │ F1, PR-AUC, Recall             │
│ FP costly          │ Precision (spam filter)        │
│ FN costly          │ Recall (cancer detection)      │
│ Ranking/threshold  │ ROC-AUC                        │
│ Severe imbalance   │ PR-AUC (better than ROC-AUC)   │
│ Regression         │ RMSE, MAE, R²                  │
│ Regression+outliers│ MAE, Huber loss                │
└────────────────────┴────────────────────────────────┘
```

### Hyperparameter Cheat Sheet

```
Random Forest:
  n_estimators: 100-1000 (more = better, diminishing returns)
  max_depth: None or 5-20
  max_features: 'sqrt' (classification) / 'log2' / 0.5
  min_samples_leaf: 1-20

XGBoost:
  n_estimators: 100-3000 (with early stopping)
  max_depth: 3-8 (default 6)
  learning_rate: 0.01-0.3 (lower + more estimators = better)
  subsample: 0.5-1.0
  colsample_bytree: 0.5-1.0

LightGBM:
  n_estimators: 100-3000
  num_leaves: 20-300 (vs max_depth in XGB)
  learning_rate: 0.01-0.3
  feature_fraction: 0.5-1.0

SVM:
  C: 0.01-1000 (log scale)
  gamma: 'scale' or 0.001-10 (log scale)
  kernel: 'rbf' (default), 'linear', 'poly'

Neural Network:
  lr: 1e-4 to 1e-2 (most important!)
  batch_size: 32-512
  dropout: 0.1-0.5
  weight_decay: 1e-5 to 1e-3
```

### Formula Reference Card

```
┌───────────────────────────────────────────────────────────┐
│                    KEY FORMULAS                            │
├─────────────────────┬─────────────────────────────────────┤
│ Sigmoid             │ σ(z) = 1/(1+e⁻ᶻ)                  │
│ Softmax             │ eˣⁱ / Σeˣʲ                         │
│ ReLU                │ max(0, x)                           │
│ Cross-Entropy       │ -Σ yᵢ log(ŷᵢ)                     │
│ MSE                 │ (1/n) Σ(y-ŷ)²                      │
│ L1 Reg              │ λΣ|w|                               │
│ L2 Reg              │ λΣw²                                │
│ Gradient Descent    │ w -= α∇L                            │
│ Bayes' Theorem      │ P(A|B) = P(B|A)P(A)/P(B)           │
│ Gini Impurity       │ 1 - Σpⱼ²                           │
│ Entropy             │ -Σpⱼ log₂(pⱼ)                     │
│ Normal Equation     │ w = (XᵀX)⁻¹Xᵀy                    │
│ F1 Score            │ 2PR/(P+R)                           │
│ F-beta              │ (1+β²)PR/(β²P+R)                   │
│ Silhouette          │ (b-a)/max(a,b)                      │
│ Information Gain    │ H(S) - Σ|Sv|/|S| H(Sv)             │
└─────────────────────┴─────────────────────────────────────┘
```

---

## 📚 Resources & Further Reading

| Topic | Resource |
|-------|---------|
| ML Theory | Pattern Recognition & ML — Bishop |
| Deep Learning | Deep Learning — Goodfellow, Bengio, Courville |
| Practical ML | Hands-on ML — Aurélien Géron |
| Statistics | Statistical Learning — James, Witten, Hastie |
| XGBoost | Original paper — Chen & Guestrin (2016) |
| SHAP | Lundberg & Lee (2017) — NeurIPS paper |
| Bayesian Opt | Brochu et al. (2010) Tutorial |
| Sklearn Docs | scikit-learn.org |
| Kaggle | kaggle.com/learn |

---

> 📌 **Notes by Deepak | SKIT | b230665@skit.ac.in**
> 
> _Last updated: 2025 | Version 3.0 | Fraud Detection Project_

---
*End of ML Complete Notes — 3 Sections*
