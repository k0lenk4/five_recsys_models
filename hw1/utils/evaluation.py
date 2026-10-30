"""Единый warm-start full-catalog протокол и подбор только на validation."""
from itertools import product
import time
import numpy as np
import pandas as pd


def eligible_truth(heldout, history, users, items, threshold=.8):
    """Оставляет релевантные отложенные пары из train-каталога, исключая историю."""
    data = heldout[heldout.user_id.isin(users) & heldout.video_id.isin(items)]
    if data.duplicated(['user_id', 'video_id']).any():
        raise ValueError('Повторные пары нужно агрегировать при подготовке данных')
    seen = pd.MultiIndex.from_frame(history[['user_id', 'video_id']])
    data = data[~pd.MultiIndex.from_frame(data[['user_id', 'video_id']]).isin(seen)]
    # Непрерывный watch_ratio превращаем в бинарную релевантность порогом.
    return data[data.watch_ratio >= threshold].reset_index(drop=True)


def ranking_metrics(recommendations, truth, k=10):
    """Усредняет Recall и NDCG по пользователям truth; отсутствие рекомендаций даёт 0."""
    if k < 1:
        raise ValueError('k должен быть положительным')
    # G_u — наблюдавшиеся отложенные видео, прошедшие порог и исключения истории.
    relevant = truth.groupby('user_id').video_id.apply(set)
    if relevant.empty:
        raise ValueError('Нет пользователей с доступными релевантными видео')
    predictions = {u: g.sort_values('predicted_score', ascending=False, kind='stable')
                   .video_id.drop_duplicates().iloc[:k].tolist()
                   for u, g in recommendations.groupby('user_id')}
    recalls, ndcgs = [], []
    for user, targets in relevant.items():
        gains = np.array([item in targets for item in predictions.get(user, [])], dtype=float)
        dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(targets))) + 2))
        recalls.append(gains.sum() / len(targets))
        ndcgs.append(dcg / ideal)
    return {f'Recall@{k}': float(np.mean(recalls)), f'NDCG@{k}': float(np.mean(ndcgs)),
            'n_users': len(relevant)}


def grid_search(model_class, grid, train, validation_truth, fit_kwargs=None):
    """Перебирает настройки по validation NDCG@10; возвращает модель, параметры и таблицу."""
    records, best_model, best_score, best_params = [], None, -np.inf, None
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        start = time.perf_counter()
        model = model_class(**params).fit(train, **(fit_kwargs or {}))
        recommendations = model.recommend(validation_truth.user_id.unique(), n=10)
        metrics = ranking_metrics(recommendations, validation_truth)
        diagnostics = {}
        if hasattr(model, 'loss_'):
            size = len(model.users_) * len(model.items_)
            diagnostics = {'train_objective': model.loss_ / size,
                           'train_reconstruction': model.reconstruction_loss_ / size,
                           'train_regularization': model.regularization_loss_ / size,
                           'weight_norm_squared': model.weight_norm_squared_}
        records.append({**params, **metrics, **diagnostics, 'seconds': time.perf_counter() - start})
        print(model_class.__name__, params, metrics, flush=True)
        if metrics['NDCG@10'] > best_score:
            best_score, best_model, best_params = metrics['NDCG@10'], model, params
    if best_model is None:
        raise ValueError('Пустая сетка гиперпараметров')
    return best_model, best_params, pd.DataFrame(records)
