"""FM второго порядка, MSE, аналитический градиент и Adam на NumPy/SciPy.
Признаки: one-hot user_id/video_id и multi-hot статические категории видео.
"""
import ast
import numpy as np
from scipy.sparse import csr_matrix
from .common import Recommender


def fm_predict(x, bias, linear, factors):
    """Вычисляет смещение, линейный вклад и попарные взаимодействия признаков."""
    summed = x @ factors
    interactions = .5 * (summed**2 - x.multiply(x) @ (factors**2)).sum(axis=1)
    return bias + np.asarray(x @ linear).ravel() + interactions


def fm_loss_grad(x, y, bias, linear, factors, reg):
    """Возвращает MSE с L2 и градиенты смещения, линейных весов и факторов."""
    error = fm_predict(x, bias, linear, factors) - y
    scaled = 2 * error / len(y)
    loss = np.mean(error**2) + reg * (np.sum(linear**2) + np.sum(factors**2))
    grad_linear = np.asarray(x.T @ scaled).ravel() + 2 * reg * linear
    grad_factors = x.T @ (scaled[:, None] * (x @ factors))
    grad_factors -= np.asarray(x.multiply(x).T @ scaled)[:, None] * factors
    grad_factors += 2 * reg * factors
    return loss, (scaled.sum(), grad_linear, grad_factors)


class FactorizationMachine(Recommender):
    def __init__(self, n_factors=16, reg=1e-4, learning_rate=.01, n_iter=10,
                 batch_size=2048, random_state=42):
        """Задаёт размер факторов, L2, скорость обучения, эпохи, размер пакета и seed."""
        if n_factors < 1 or reg < 0 or learning_rate <= 0 or n_iter < 1 or batch_size < 1:
            raise ValueError('Некорректные параметры FM')
        self.n_factors, self.reg, self.learning_rate = n_factors, reg, learning_rate
        self.n_iter, self.batch_size, self.random_state = n_iter, batch_size, random_state

    def _features(self, user_indices, item_indices):
        """Кодирует пары внутренних индексов: one-hot ID и multi-hot категории."""
        rows, cols = [], []
        for row, (u, i) in enumerate(zip(user_indices, item_indices)):
            active = [int(u), len(self.users_) + int(i)] + self.item_tags_[i]
            rows.extend([row] * len(active))
            cols.extend(active)
        return csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(user_indices), self.n_features_))

    def fit(self, df, item_categories=None):
        """Готовит категории и обучает FM пакетным Adam на NumPy; возвращает self."""
        self._prepare(df)
        if item_categories is None or not {'video_id', 'feat'} <= set(item_categories.columns):
            raise ValueError('FM требует item_categories с video_id и feat')
        metadata = item_categories.drop_duplicates('video_id').set_index('video_id').feat.to_dict()
        tags = []
        for item in self.items_:
            values = metadata.get(item, [])
            if isinstance(values, str):
                values = ast.literal_eval(values)
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = []
            tags.append(sorted(set(map(str, values))) or ['__missing__'])
        vocabulary = sorted({tag for values in tags for tag in values})
        if vocabulary == ['__missing__']:
            raise ValueError('В обучающем каталоге отсутствуют категории видео')
        offset = len(self.users_) + len(self.items_)
        encoding = {tag: offset + j for j, tag in enumerate(vocabulary)}
        self.item_tags_ = [[encoding[t] for t in values] for values in tags]
        self.n_features_ = offset + len(vocabulary)
        self.feature_names_ = ([f'user_id={u}' for u in self.users_]
                               + [f'video_id={i}' for i in self.items_]
                               + [f'category={tag}' for tag in vocabulary])
        u = np.searchsorted(self.users_, self.data_.user_id)
        i = np.searchsorted(self.items_, self.data_.video_id)
        x = self._features(u, i)
        y = self.data_.watch_ratio.to_numpy().clip(0, 1)
        rng = np.random.default_rng(self.random_state)
        parameters = [np.array(y.mean()), np.zeros(self.n_features_),
                      rng.normal(0, .01, (self.n_features_, self.n_factors))]
        first = [np.zeros_like(p) for p in parameters]
        second = [np.zeros_like(p) for p in parameters]
        step = 0
        # Отдельный RNG сохраняет прежнюю инициализацию и порядок обучающих пакетов.
        diagnostic_rng = np.random.default_rng(self.random_state + 1)
        self.diagnostic_indices_ = np.sort(diagnostic_rng.choice(
            len(y), size=min(16384, len(y)), replace=False))
        diagnostic_x, diagnostic_y = x[self.diagnostic_indices_], y[self.diagnostic_indices_]
        self.loss_history_, self.data_loss_history_, self.regularization_history_ = [], [], []
        self.batch_loss_history_ = []
        self.loss_description_ = (f'MSE + L2 на фиксированных {len(diagnostic_y)} train-парах')
        self._record_loss(diagnostic_x, diagnostic_y, parameters)
        for _ in range(self.n_iter):
            order = rng.permutation(len(y))
            epoch_loss = 0.
            for start in range(0, len(y), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss, gradients = fm_loss_grad(x[batch], y[batch], *parameters, self.reg)
                epoch_loss += loss * len(batch)
                step += 1
                for j, gradient in enumerate(gradients):
                    first[j] = .9 * first[j] + .1 * gradient
                    second[j] = .999 * second[j] + .001 * gradient**2
                    parameters[j] -= self.learning_rate * (first[j] / (1 - .9**step)) / (
                        np.sqrt(second[j] / (1 - .999**step)) + 1e-8)
            self.batch_loss_history_.append(epoch_loss / len(y))
            self._record_loss(diagnostic_x, diagnostic_y, parameters)
        self.history_steps_ = list(range(self.n_iter + 1))
        self.bias_, self.linear_, self.factors_ = parameters
        return self

    def _record_loss(self, x, y, parameters):
        """Записывает MSE, L2 и их сумму на неизменной train-подвыборке после эпохи."""
        bias, linear, factors = parameters
        mse = float(np.mean((fm_predict(x, bias, linear, factors) - y)**2))
        penalty = float(self.reg * (np.sum(linear**2) + np.sum(factors**2)))
        self.data_loss_history_.append(mse)
        self.regularization_history_.append(penalty)
        self.loss_history_.append(mse + penalty)

    def score_user(self, u):
        """Оценивает весь каталог для пользователя u пакетами пар пользователь–видео."""
        result = []
        for start in range(0, len(self.items_), self.batch_size):
            indices = np.arange(start, min(start + self.batch_size, len(self.items_)))
            x = self._features(np.full(len(indices), u), indices)
            result.append(fm_predict(x, self.bias_, self.linear_, self.factors_))
        return np.concatenate(result)
