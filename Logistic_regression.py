import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD THE PROCESSED DATASETS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv").to_numpy(dtype=float)
X_test = pd.read_csv(DATA_DIR / "X_test.csv").to_numpy(dtype=float)

y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy(dtype=float).reshape(-1, 1)
y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy(dtype=float).reshape(-1, 1)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ==========================================
# 2. DEFINE THE LOGISTIC REGRESSION MODEL
# ==========================================

class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.1, epochs=5000,
                 class_weight="balanced", l2=0.01, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight   # FIX 1: handle class imbalance
        self.l2 = l2                       # FIX 2: L2 regularisation
        self.sample_weights = None
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    # Convert scores into probabilities between 0 and 1
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # Calculate binary cross-entropy loss
    # FIX 1: weighted binary cross-entropy so the minority class counts
    def compute_loss(self, y_true, y_prob, sample_weights=None):
        epsilon = 1e-15
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)

        if sample_weights is None:
            sample_weights = np.ones_like(y_true)

        loss = -np.mean(
            sample_weights * (
                y_true * np.log(y_prob)
                + (1 - y_true) * np.log(1 - y_prob)
            )
        )

        # FIX 2: L2 penalty (bias is not penalised)
        loss += (self.l2 / 2) * np.sum(self.weights ** 2)
        return loss

    # Train the model using gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        # FIX 1: balanced class weights  w_c = n_samples / (2 * n_c)
        if self.class_weight == "balanced":
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w_pos = n_samples / (2 * n_pos)
            w_neg = n_samples / (2 * n_neg)
            self.sample_weights = np.where(y == 1, w_pos, w_neg)
            if self.verbose:
                print(f"Class weights | non-bestseller: {w_neg:.3f} | bestseller: {w_pos:.3f}")
        else:
            self.sample_weights = np.ones_like(y)

        for epoch in range(self.epochs):

            # Forward propagation
            linear_output = np.dot(X, self.weights) + self.bias
            probabilities = self.sigmoid(linear_output)

            # Calculate loss
            loss = self.compute_loss(y, probabilities, self.sample_weights)
            self.loss_history.append(loss)

            # Calculate gradients
            # FIX 1: weighted error, FIX 2: L2 gradient term
            error = self.sample_weights * (probabilities - y)
            dw = np.dot(X.T, error) / n_samples + self.l2 * self.weights
            db = np.mean(error)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Display progress
            if self.verbose and (epoch + 1) % 500 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"| Loss: {loss:.4f}"
                )

        return self

    # Return predicted probabilities
    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    # Convert probabilities into class predictions
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# ==========================================
# 3. TRAIN THE MODEL
# ==========================================

# FIX 3: choose the decision threshold with 5 fold cross validation
# on the TRAINING data only. The test set is never used for tuning.
def f1_at(probs, labels, t):
    pred = (probs >= t).astype(int)
    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

rng = np.random.default_rng(42)
labels_flat = y_train.ravel()
pos_idx = rng.permutation(np.where(labels_flat == 1)[0])
neg_idx = rng.permutation(np.where(labels_flat == 0)[0])
n_folds = 5
pos_folds = np.array_split(pos_idx, n_folds)
neg_folds = np.array_split(neg_idx, n_folds)

oof_prob = np.zeros(len(labels_flat))   # out of fold probabilities
for k in range(n_folds):
    val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
    fit_idx = np.setdiff1d(np.arange(len(labels_flat)), val_idx)
    fold_model = LogisticRegressionScratch(learning_rate=0.1, epochs=5000, verbose=False)
    fold_model.fit(X_train[fit_idx], y_train[fit_idx])
    oof_prob[val_idx] = fold_model.predict_proba(X_train[val_idx]).ravel()

candidate_thresholds = np.round(np.arange(0.30, 0.71, 0.01), 2)
cv_f1s = [f1_at(oof_prob, labels_flat, t) for t in candidate_thresholds]
best_threshold = float(candidate_thresholds[int(np.argmax(cv_f1s))])
print(f"\nBest threshold from 5 fold CV: {best_threshold:.2f} "
      f"(CV F1 = {max(cv_f1s):.4f})")

# Retrain on the full training set with the same settings
model = LogisticRegressionScratch(learning_rate=0.1, epochs=5000)
model.fit(X_train, y_train)

# ==========================================
# INVESTIGATE WHAT THE MODEL HAS LEARNED
# ==========================================

# Calculate probabilities for the training data
train_prob = model.predict_proba(X_train)
train_pred = model.predict(X_train, threshold=best_threshold)

# Read the feature names from the original CSV
feature_names = pd.read_csv(
    DATA_DIR / "X_train.csv"
).columns

print("\n========== TRAINING DIAGNOSTICS ==========")

print("Learned bias:", round(float(model.bias), 4))

print("Training minimum probability:",
      round(float(train_prob.min()), 4))

print("Training maximum probability:",
      round(float(train_prob.max()), 4))

print("Training mean probability:",
      round(float(train_prob.mean()), 4))

print("Training books predicted as bestsellers:",
      int(np.sum(train_pred == 1)))

print("Actual training bestsellers:",
      int(np.sum(y_train == 1)))

# Display the learned weights
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Weight": model.weights.ravel()
})

coefficients["Absolute_Weight"] = (
    coefficients["Weight"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Weight",
    ascending=False
)

print("\nLargest learned weights:")
print(
    coefficients[
        ["Feature", "Weight"]
    ].head(10).to_string(index=False)
)


# ==========================================
# COMPARE PROBABILITIES BY ACTUAL CLASS
# ==========================================

train_labels = y_train.ravel()
train_prob_flat = train_prob.ravel()

actual_bestseller_probs = train_prob_flat[
    train_labels == 1
]

actual_non_bestseller_probs = train_prob_flat[
    train_labels == 0
]

print("\n========== CLASS SEPARATION ==========")

print(
    "Mean probability for actual bestsellers:",
    round(float(actual_bestseller_probs.mean()), 4)
)

print(
    "Mean probability for actual non-bestsellers:",
    round(float(actual_non_bestseller_probs.mean()), 4)
)

print(
    "Median probability for actual bestsellers:",
    round(float(np.median(actual_bestseller_probs)), 4)
)

print(
    "Median probability for actual non-bestsellers:",
    round(float(np.median(actual_non_bestseller_probs)), 4)
)

# Investigate alternative thresholds on training data
print("\n========== TRAINING THRESHOLD CHECK ==========")

for threshold in [0.40, 0.45, 0.50, best_threshold, 0.55, 0.60]:
    predictions = (
        train_prob_flat >= threshold
    ).astype(int)

    positive_count = int(np.sum(predictions == 1))

    true_positives = int(
        np.sum((predictions == 1) & (train_labels == 1))
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Positive predictions: {positive_count} | "
        f"Actual bestsellers identified: {true_positives}/{int(train_labels.sum())}"
    )



# ==========================================
# 4. GENERATE TEST PREDICTIONS
# ==========================================

y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=best_threshold)  # FIX 3

# ==========================================
# 4.1. INVESTIGATE THE PREDICTED PROBABILITIES
# ==========================================

print("\n========== DIAGNOSTICS ==========")

print("Training bestseller proportion:",
      round(float(y_train.mean()), 4))

print("Actual test bestseller proportion:",
      round(float(y_test.mean()), 4))

print("Minimum predicted probability:",
      round(float(y_prob.min()), 4))

print("Maximum predicted probability:",
      round(float(y_prob.max()), 4))

print("Mean predicted probability:",
      round(float(y_prob.mean()), 4))

print("Predictions classified as bestsellers:",
      int(np.sum(y_pred == 1)))

print("Predictions classified as non-bestsellers:",
      int(np.sum(y_pred == 0)))

print("\nPredicted probability percentiles:")
print(np.percentile(y_prob, [0, 25, 50, 75, 90, 95, 100]))



# ==========================================
# 5. EVALUATE THE MODEL
# ==========================================

y_true = y_test.astype(int).ravel()
y_pred = y_pred.ravel()
y_prob = y_prob.ravel()

true_positive = np.sum((y_true == 1) & (y_pred == 1))
true_negative = np.sum((y_true == 0) & (y_pred == 0))
false_positive = np.sum((y_true == 0) & (y_pred == 1))
false_negative = np.sum((y_true == 1) & (y_pred == 0))

accuracy = (true_positive + true_negative) / len(y_true)

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0 else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0 else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}")

print("\nConfusion Matrix:")
print(
    np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])
)

# ROC AUC (threshold free view of how well the model ranks books)
order = np.argsort(y_prob)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(y_prob) + 1)
n_pos_t = y_true.sum()
n_neg_t = len(y_true) - n_pos_t
auc = (ranks[y_true == 1].sum() - n_pos_t * (n_pos_t + 1) / 2) / (n_pos_t * n_neg_t)
print(f"\nROC AUC:   {auc:.4f}")
print(f"Threshold used: {best_threshold:.2f}")

# Baseline: always predict "not a bestseller"
print(f"Baseline accuracy (always 0): {np.mean(y_true == 0):.4f}")

print("\nFinal training loss:", round(model.loss_history[-1], 4))