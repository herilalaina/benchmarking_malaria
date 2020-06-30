"""This file is for patches classification (step 1).

Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
"""
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class Model(BaseEstimator):
    """Main class for Classification problem."""

    def __init__(self):
        """Init method.

        We define here a simple (shallow) CNN.
        """
        self.is_trained = False

        self.model = LogisticRegression(random_state=42)

    def fit(self, X, y):
        """Fit method.

        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape 7500 features
               (it can be reshape to (50, 50, 3)).
            y: Training label matrix of dim num_train_samples.
        Both inputs are numpy arrays.
        """
        self.model.fit(X / 255, y)
        self.is_trained = True

    def predict(self, X):
        """Predict method.

        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape 7500 features
               (it can be reshape to (50, 50, 3)).
        This function should provide *probabilities* of labels on (test) data.
        """
        return self.model.predict_proba(X / 255)[:, 0]
