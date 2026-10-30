"""EASE: замкнутое решение из семинара 05, по watch_ratio как в исходном ноутбуке."""
import numpy as np
from .common import Recommender


class EASE(Recommender):
    def __init__(self, reg=100.):
        """Задаёт положительный коэффициент L2-регуляризации."""
        if reg <= 0:
            raise ValueError('reg должен быть положительным')
        self.reg = reg

    def fit(self, df):
        """Находит матрицу связей видео с нулевой диагональю; возвращает self."""
        self._prepare(df)
        gram = (self.ratings_.T @ self.ratings_).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = np.linalg.solve(gram, np.eye(len(self.items_)))
        self.weights_ = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights_, 0)
        self.weight_norm_squared_ = float(np.sum(self.weights_**2))
        self.regularization_loss_ = self.reg * self.weight_norm_squared_
        # В оптимуме с diag(B)=0: J = tr(R.T R) - <R.T R, B>.
        # Диагональная добавка reg не влияет на скалярное произведение, так как diag(B)=0.
        trace_original = float(np.trace(gram) - self.reg * len(self.items_))
        self.loss_ = trace_original - float(np.sum(gram * self.weights_))
        self.reconstruction_loss_ = self.loss_ - self.regularization_loss_
        return self

    def score_user(self, u):
        """Умножает историю пользователя u на обученную матрицу связей видео."""
        return np.asarray(self.ratings_[u] @ self.weights_).ravel()
