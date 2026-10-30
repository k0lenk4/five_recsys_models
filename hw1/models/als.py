"""Самостоятельная iALS: p=M, c=1+alpha*M; ненаблюдаемые пары входят в loss."""
import numpy as np
from .common import Recommender


class ImplicitALS(Recommender):
    def __init__(self, n_factors=32, reg=0.1, n_iter=15, alpha=10., random_state=42):
        """Задаёт размер факторов, регуляризацию, число итераций, confidence и seed."""
        if n_factors < 1 or reg <= 0 or n_iter < 1 or alpha < 0:
            raise ValueError('Некорректные параметры ALS')
        self.n_factors, self.reg, self.n_iter = n_factors, reg, n_iter
        self.alpha, self.random_state = alpha, random_state

    def _targets(self):
        """Возвращает бинарный таргет: единица для любой наблюдавшейся пары."""
        return self.observed_.copy()

    def _step(self, factors, targets, observed):
        """Решает ALS-системы для одной стороны при фиксированных факторах другой."""
        gram = factors.T @ factors + self.reg * np.eye(self.n_factors)
        output = np.zeros((observed.shape[0], self.n_factors))
        for row in range(observed.shape[0]):
            indices = observed.indices[observed.indptr[row]:observed.indptr[row + 1]]
            vectors = factors[indices]
            a = gram + self.alpha * (vectors.T @ vectors)
            # Targets сохраняют explicit zeros; доступ через маску наблюдений.
            values = targets[row, indices].toarray().ravel()
            b = (1 + self.alpha) * (vectors.T @ values)
            output[row] = np.linalg.solve(a, b)
        return output

    def _loss_components(self, target):
        """Считает полный weighted SSE и L2 без плотной матрицы всех предсказаний."""
        x, y = self.user_factors_, self.item_factors_
        # Сумма квадратов предсказаний по всем парам, включая ненаблюдаемые.
        squared_error = float(np.sum((x.T @ x) * (y.T @ y)))
        observed = self.observed_.tocoo()
        for start in range(0, observed.nnz, 16384):
            rows = observed.row[start:start + 16384]
            cols = observed.col[start:start + 16384]
            predictions = np.sum(x[rows] * y[cols], axis=1)
            values = np.asarray(target[rows, cols]).ravel()
            # Для наблюдавшихся пар заменяем базовый квадрат на confidence-weighted SSE.
            squared_error += np.sum((1 + self.alpha) * (values - predictions)**2
                                    - predictions**2)
        penalty = self.reg * (np.sum(x**2) + np.sum(y**2))
        size = len(self.users_) * len(self.items_)
        return float(squared_error / size), float(penalty / size)

    def fit(self, df):
        """Поочерёдно обучает факторы пользователей и видео; возвращает self."""
        self._prepare(df)
        target = self._targets().tocsr()
        mask_t = self.observed_.T.tocsr()
        target_t = target.T.tocsr()
        rng = np.random.default_rng(self.random_state)
        self.item_factors_ = rng.normal(0, .01, (len(self.items_), self.n_factors))
        self.user_factors_ = np.zeros((len(self.users_), self.n_factors))
        self.loss_history_, self.data_loss_history_, self.regularization_history_ = [], [], []
        self.loss_description_ = 'Взвешенная квадратичная ошибка с L2-регуляризацией'
        for iteration in range(self.n_iter + 1):
            if iteration:
                self.user_factors_ = self._step(self.item_factors_, target, self.observed_)
                self.item_factors_ = self._step(self.user_factors_, target_t, mask_t)
            data_loss, penalty = self._loss_components(target)
            self.data_loss_history_.append(data_loss)
            self.regularization_history_.append(penalty)
            self.loss_history_.append(data_loss + penalty)
        self.history_steps_ = list(range(self.n_iter + 1))
        return self

    def score_user(self, u):
        """Возвращает скалярные произведения факторов пользователя u со всеми видео."""
        return self.item_factors_ @ self.user_factors_[u]
