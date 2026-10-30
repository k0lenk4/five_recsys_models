"""UserKNN: косинус либо Pearson по совместно просмотренным видео."""
import numpy as np
from .common import Recommender


class UserKNN(Recommender):
    def __init__(self, similarity='cosine', k=20):
        """Задаёт меру сходства и максимальное число положительных соседей."""
        if similarity not in ('cosine', 'pearson') or k < 1:
            raise ValueError('Ожидаются cosine/pearson и k >= 1')
        self.similarity, self.k = similarity, k

    def fit(self, df):
        """Вычисляет cosine или Pearson для каждой пары пользователей; возвращает self."""
        self._prepare(df)
        r, m = self.ratings_, self.observed_
        product = (r @ r.T).toarray()
        if self.similarity == 'cosine':
            norm = np.sqrt(np.asarray(r.multiply(r).sum(axis=1)).ravel())
            denominator = norm[:, None] * norm[None, :]
            numerator = product
        else:
            count = (m @ m.T).toarray()
            sums = (r @ m.T).toarray()
            squares = (r.multiply(r) @ m.T).toarray()
            safe_count = np.maximum(count, 1)
            numerator = product - sums * sums.T / safe_count
            var = np.maximum(squares - sums**2 / safe_count, 0)
            denominator = np.sqrt(var * var.T)
            denominator[count < 2] = 0
        self.similarity_ = np.divide(numerator, denominator,
            out=np.zeros_like(product), where=denominator > 1e-12)
        self.similarity_ = np.clip(self.similarity_, -1, 1)
        np.fill_diagonal(self.similarity_, 0)
        return self

    def score_user(self, u):
        """Оценивает весь каталог по просмотрам соседей для внутреннего индекса u."""
        sim = self.similarity_[u]
        # Положительные веса позволяют усреднять сигналы похожих пользователей.
        neighbors = np.flatnonzero(sim > 0)
        neighbors = neighbors[np.argsort(-sim[neighbors], kind='stable')[:self.k]]
        weights = sim[neighbors]
        numerator = np.asarray(self.ratings_[neighbors].T @ weights).ravel()
        denominator = np.asarray(self.observed_[neighbors].T @ weights).ravel()
        return np.divide(numerator, denominator, out=np.full(len(self.items_), -np.inf),
                         where=denominator > 0)
