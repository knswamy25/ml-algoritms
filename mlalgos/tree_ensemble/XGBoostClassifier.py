from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load data set
cancer_dataset = load_breast_cancer()
#  cancer_dataset.DESCR - Provides dataset description and
#  cancer_dataset.feature_names - Provides feature names present in the dataset
X = cancer_dataset.data
y = cancer_dataset.target
# Check the shape
print(X.shape)

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create eval set
# One interesting thing about the XGBoost is that during fitting,
# it can take in an evaluation dataset of the form (X_val,y_val).
# On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
# Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop.
# More iterations lead to more estimators, and more estimators can result in overfitting.
# By stopping once the validation metric no longer improves, we can limit the number of estimators created, and reduce overfitting.
n = int(len(X_train)*0.8) # use 80% to train and 20% to eval
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

# Create model
xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, verbosity=1, random_state=42, early_stopping_rounds=10)
xgb_classifier.fit(X_train_fit, y_train_fit, eval_set = [(X_train_eval, y_train_eval)])

# Round of training that had best performance
print(xgb_classifier.best_iteration)

# Calculate accuracy score
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_classifier.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_classifier.predict(X_test),y_test):.4f}")







