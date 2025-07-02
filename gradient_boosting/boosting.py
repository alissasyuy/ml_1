from __future__ import annotations
from collections import defaultdict

import numpy as np
from typing import Optional, Union
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def score(clf, X, y):
    return roc_auc_score(y == 1, clf.predict_proba(X)[:, 1])


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None, 
        epsilon: float = 1e-4,
        subsample: Union[float, int] = 1.0,
        bagging_temperature: Union[float, int] = 1.0,
        bootstrap_type: Optional[str] = 'bernoulli',  # 'bernoulli', 'bayesian' или None
    ):
        
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate

        self.early_stopping_rounds = early_stopping_rounds
        self.epsilon = epsilon
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type.lower() if bootstrap_type is not None else None

        self.models: list = []
        self.gammas: list = []

        self.history = defaultdict(list)  # например, {"train_loss": [...], "val_loss": [...]}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def _get_indices(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.bootstrap_type == 'bernoulli':
            if self.subsample <= 1.0: p = self.subsample / self.bagging_temperature
            elif isinstance(self.subsample, int): p = self.subsample / n / self.bagging_temperature
            elif isinstance(self.subsample, float): p = 1.0 / self.bagging_temperature
            else: p = self.subsample / n / self.bagging_temperature
            p = np.clip(p, 0, 1)
            m = np.random.binomial(1, p, n).astype(bool)
            i = np.where(m)[0]
            if len(i) == 0: i = np.random.choice(n, 1, replace=False)
            return i
        elif self.bootstrap_type == 'bayesian':
            x = np.random.uniform(0, 1, n)
            w = (-np.log(x)) ** self.bagging_temperature
            w /= w.sum()
            return np.random.choice(n, n, replace=True, p=w)
        else:
            if isinstance(self.subsample, float) and self.subsample <= 1.0:
                s = int(n * self.subsample)
            elif isinstance(self.subsample, int):
                s = min(self.subsample, n)
            else:
                s = self.subsample
            s = min(s, n)
            return np.random.choice(n, s, replace=False)


    def find_optimal_gamma(self, y, old_pred, new_pred) -> float:
        gammas = np.linspace(0, 1, 100)
        losses = [self.loss_fn(y, old_pred + gamma * new_pred) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def partial_fit(self, X, y, old_predictions):
        indices = self._get_indices(X)

        bx = X[indices]
        by = y[indices]
        b_pred = old_predictions[indices]

        anti_grad = -self.loss_derivative(by, b_pred)

        model = self.base_model_class(**self.base_model_params)
        model.fit(bx, anti_grad)

        new_predictions_bx = model.predict(bx)
        gamma = self.find_optimal_gamma(by, b_pred, new_predictions_bx)

        self.models.append(model)
        self.gammas.append(gamma)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        valid_predictions = None
        if X_val is not None and y_val is not None:
            valid_predictions = np.zeros_like(y_val, dtype=float)

        train_predictions = np.zeros_like(y_train, dtype=float)

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_train)
            train_loss = self.loss_fn(y_train, train_predictions)
            self.history['train_loss'].append(train_loss)

            if X_val is not None and y_val is not None:
                valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)
                val_loss = self.loss_fn(y_val, valid_predictions)
                self.history['val_loss'].append(val_loss)

                improvement = self.best_val_loss - val_loss
                if improvement > self.epsilon:
                    self.best_val_loss = val_loss
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

                if (
                    self.early_stopping_rounds is not None
                    and self.no_improvement_count >= self.early_stopping_rounds
                ):
                    print(f"Остановка на итерации {i+1} (Δ лосс={improvement:.6f} < eps={self.epsilon})")
                    break

            if (i + 1) % 10 == 0 or (i + 1) == self.n_estimators:
                msg = f"Эпоха {i+1}: Трейн лосс={train_loss:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f", Лосс на валидации={val_loss:.4f}, Счетчик Δ лосс={self.no_improvement_count}/{self.early_stopping_rounds}"
                print(msg)

        if plot:
            self.plot_history()

    def predict_proba(self, X):
        z = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            z += self.learning_rate * gamma * model.predict(X)
        proba_plus = self.sigmoid(z)
        proba_minus = 1 - proba_plus
        return np.column_stack((proba_minus, proba_plus))

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(0, 1, 100)
        losses = [self.loss_fn(y, old_predictions + g * new_predictions) for g in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

#         plt.figure(figsize=(10, 6))
        if 'train_loss' in self.history:
            plt.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')

        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(epochs, self.history['val_loss'], label='Validation Loss', marker='o')

        plt.title('История обучения Boosting')
        plt.xlabel('Эпоха')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
