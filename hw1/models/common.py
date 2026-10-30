"""Общий каталог train и маска наблюдений, включая watch_ratio == 0."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def prepare_interactions(df):
    data = df.dropna(subset=['user_id', 'video_id', 'watch_ratio']).copy()
    data = data.groupby(['user_id', 'video_id'], as_index=False, sort=True).watch_ratio.mean()
    users = np.sort(data.user_id.unique())
    items = np.sort(data.video_id.unique())
    rows = pd.Index(users).get_indexer(data.user_id)
    cols = pd.Index(items).get_indexer(data.video_id)
    shape = (len(users), len(items))
    ratings = csr_matrix((data.watch_ratio.to_numpy(), (rows, cols)), shape=shape)
    observed = csr_matrix((np.ones(len(data)), (rows, cols)), shape=shape)
    return data, users, items, ratings, observed


class Recommender:
    def _prepare(self, df):
        self.data_, self.users_, self.items_, self.ratings_, self.observed_ = prepare_interactions(df)
        self.user_index_ = {u: j for j, u in enumerate(self.users_)}
        return self

    def recommend(self, users=None, n=10, exclude=None):
        blocked = {} if exclude is None else exclude.groupby('user_id').video_id.apply(set).to_dict()
        results = []
        for user in self.users_ if users is None else users:
            if user not in self.user_index_:
                continue  # единый warm-start протокол
            u = self.user_index_[user]
            scores = np.asarray(self.score_user(u), dtype=float).copy()
            scores[self.observed_[u].indices] = -np.inf
            if user in blocked:
                scores[np.isin(self.items_, list(blocked[user]))] = -np.inf
            valid = np.flatnonzero(np.isfinite(scores))
            # Стабильный tie-break по отсортированному video_id.
            top = valid[np.argsort(-scores[valid], kind='stable')[:n]]
            results.extend((user, self.items_[i], scores[i]) for i in top)
        return pd.DataFrame(results, columns=['user_id', 'video_id', 'predicted_score']).astype(
            {'predicted_score': float})
