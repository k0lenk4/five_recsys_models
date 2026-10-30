"""Графики обучения и validation-перебора для вывода в ноутбуке."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

METRICS = ('NDCG@10', 'Recall@10')


PARAMETER_NAMES = {'n_factors': 'числа факторов', 'reg': 'L2-регуляризации',
                   'learning_rate': 'скорости обучения', 'n_iter': 'числа итераций',
                   'k': 'числа соседей', 'power': 'степени преобразования таргета',
                   'similarity': 'меры сходства'}
HYPERPARAMETERS = ('similarity', 'k', 'n_factors', 'reg', 'n_iter', 'alpha',
                  'power', 'learning_rate', 'batch_size')


def _dependency_title(name, quantity, variable, params=None, exclude=(), width=75):
    """Описывает зависимость и добавляет только известные фиксированные параметры."""
    import textwrap
    title = f'{name}: зависимость {quantity} от {variable}' if name else f'Зависимость {quantity} от {variable}'
    fixed = []
    for key in HYPERPARAMETERS:
        if key in exclude or not params or key not in params or params[key] is None:
            continue
        value = params[key]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                continue
            value = f'{value:g}'
        fixed.append(f'{key}={value}')
    title = textwrap.fill(title, width=width)
    if fixed:
        title += '\n' + textwrap.fill('при ' + ', '.join(fixed), width=width)
    return title


def _fixed_parameters(table, exclude=()):
    """Извлекает гиперпараметры с одним значением в показанном наборе результатов."""
    return {key: table[key].iloc[0] for key in HYPERPARAMETERS
            if key not in exclude and key in table and table[key].nunique() == 1}


def _finish(fig):
    """Показывает график в ноутбуке и закрывает фигуру без записи на диск."""
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return None


def plot_loss_history(model, name):
    """Показывает цель, ошибку данных и L2 на всех шагах, включая шаг 0."""
    if not hasattr(model, 'loss_history_'):
        raise ValueError(f'{name}: нет итерационного обучения и истории loss')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = model.history_steps_
    ax.plot(steps, model.loss_history_, marker='o', ms=3, label='Лосс: ошибка + L2')
    ax.plot(steps, model.data_loss_history_, linestyle='--', label='Ошибка данных')
    ax.plot(steps, model.regularization_history_, linestyle=':', label='L2')
    ax.set_xlabel('Эпоха' if name == 'FM' else 'Итерация')
    ax.set_ylabel('Loss')
    ax.grid(alpha=.25)
    fixed = {key: getattr(model, key) for key in HYPERPARAMETERS if hasattr(model, key)}
    ax.set_title(_dependency_title(name, 'лосса',
        'числа эпох' if name == 'FM' else 'числа итераций', fixed,
        exclude=('n_iter',)), fontsize=11)
    ax.legend(fontsize=8)
    return _finish(fig)


def plot_validation_search(table, best_params, name):
    """Рисует срезы сетки: один параметр меняется, остальные фиксированы в best_params."""
    varying = [p for p in best_params if p in table and table[p].nunique() >= 3]
    if not varying:
        return None
    fig, axes = plt.subplots(2, len(varying), figsize=(4.2 * len(varying), 7), squeeze=False)
    for column, parameter in enumerate(varying):
        subset = table
        for other, value in best_params.items():
            if other != parameter and other in table:
                subset = subset[subset[other] == value]
        subset = subset.sort_values(parameter)
        if subset.empty:
            raise ValueError(f'Нет среза для {parameter}; best_params не соответствуют таблице')
        numeric = pd.api.types.is_numeric_dtype(subset[parameter])
        positions = subset[parameter].to_numpy() if numeric else np.arange(len(subset))
        for row, metric in enumerate(METRICS):
            ax = axes[row, column]
            ax.plot(positions, subset[metric], 'o-', linewidth=1.5)
            selected = subset[parameter].to_numpy() == best_params[parameter]
            ax.scatter(positions[selected], subset.loc[selected, metric], s=80,
                       marker='*', color='darkorange', zorder=3, label='Выбор по NDCG@10')
            if numeric and parameter in ('reg', 'learning_rate', 'k') and np.min(positions) > 0:
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xticks(positions, [f'{x:g}' if numeric else str(x) for x in subset[parameter]])
            ax.tick_params(axis='x', rotation=35)
            ax.set_xlabel(parameter)
            ax.set_ylabel(f'Validation {metric}')
            ax.set_title(_dependency_title(name, metric + ' на validation',
                PARAMETER_NAMES.get(parameter, parameter), best_params, exclude=(parameter,), width=48), fontsize=10)
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    return _finish(fig)


def plot_search_heatmap(table, best_params, name, row_parameter, column_parameter):
    """Рисует две validation-метрики для пары параметров при остальных лучших настройках."""
    subset = table
    for parameter, value in best_params.items():
        if parameter not in (row_parameter, column_parameter) and parameter in table:
            subset = subset[subset[parameter] == value]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, METRICS):
        matrix = subset.pivot(index=row_parameter, columns=column_parameter, values=metric).sort_index()
        values = matrix.to_numpy(dtype=float)
        artist = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(matrix.columns)), [str(v) for v in matrix.columns], rotation=35)
        ax.set_yticks(range(len(matrix.index)), [str(v) for v in matrix.index])
        ax.set_xlabel(column_parameter)
        ax.set_ylabel(row_parameter)
        ax.set_title(_dependency_title(name, metric + ' на validation',
            PARAMETER_NAMES.get(column_parameter, column_parameter) + ' и ' + PARAMETER_NAMES.get(row_parameter, row_parameter),
            best_params, exclude=(row_parameter, column_parameter), width=65), fontsize=10)
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                # Белая подложка сохраняет читаемость при любом фоне heatmap.
                ax.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
        fig.colorbar(artist, ax=ax, fraction=.046)
    return _finish(fig)


def plot_model_comparison(table, split):
    """Сравнивает Recall/NDCG моделей на одном и том же наборе оценки."""
    table = table.sort_values('NDCG@10', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, METRICS):
        bars = ax.bar(table.model, table[metric], color='#4487a8')
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
        ax.set_title(_dependency_title('', metric + ' на ' + split, 'метода'), fontsize=11)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=25)
        ax.set_ylim(0, max(float(table[metric].max()) * 1.25, 1e-5))
        ax.grid(axis='y', alpha=.2)
    return _finish(fig)


def plot_history_age(train, validation, test):
    """Показывает давность последнего train-события перед первым отложенным событием."""
    last_train = train.groupby('user_id').timestamp.max()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, frame) in zip(axes, [('validation', validation), ('test', test)]):
        first = frame.groupby('user_id').timestamp.min()
        days = ((first - last_train.reindex(first.index)) / 86400).dropna()
        ax.hist(days, bins=30, edgecolor='white', color='#4487a8')
        ax.axvline(days.median(), color='darkorange', linestyle='--',
                   label=f'Медиана: {days.median():.2f} дня')
        ax.set_xlabel('Дней от последнего train-просмотра до первого события периода')
        ax.set_ylabel('Пользователи')
        ax.set_title(_dependency_title('', 'числа пользователей', f'давности истории ({name})', width=60), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    return _finish(fig)


def _heatmap_on_axis(ax, matrix, metric):
    """Рисует одну тепловую карту с подписями значений."""
    values = matrix.to_numpy(dtype=float)
    artist = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(matrix.columns)), [str(v) for v in matrix.columns], rotation=30)
    ax.set_yticks(range(len(matrix.index)), [str(v) for v in matrix.index])
    ax.set_xlabel(matrix.columns.name)
    ax.set_ylabel(matrix.index.name)
    ax.set_title(_dependency_title('UserKNN', metric + ' на validation',
        PARAMETER_NAMES.get(matrix.columns.name, matrix.columns.name) + ' и ' + PARAMETER_NAMES.get(matrix.index.name, matrix.index.name),
        width=65), fontsize=10)
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
    ax.figure.colorbar(artist, ax=ax, fraction=.046)


def plot_knn_search(table):
    """KNN: две кривые метрик по k сверху и две карты similarity × k снизу."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, metric in enumerate(METRICS):
        ax = axes[0, column]
        for similarity, group in table.groupby('similarity', sort=False):
            group = group.sort_values('k')
            if group.k.nunique() >= 3:
                ax.plot(group.k, group[metric], 'o-', label=similarity)
        ticks = sorted(table.k.unique())
        ax.set_xscale('log')
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.set_xlabel('Число соседей k')
        ax.set_ylabel(f'Validation {metric}')
        ax.set_title(_dependency_title('UserKNN', metric + ' на validation', 'числа соседей', width=65), fontsize=11)
        ax.legend()
        ax.grid(alpha=.25)
        matrix = table.pivot(index='similarity', columns='k', values=metric)
        _heatmap_on_axis(axes[1, column], matrix, metric)
    return _finish(fig)


def plot_parameter_curve(table, best_params, name, parameter):
    """Рисует две метрики по одному параметру при фиксированных остальных; минимум 3 точки."""
    subset = table
    for other, value in best_params.items():
        if other != parameter and other in subset:
            subset = subset[subset[other] == value]
    subset = subset.sort_values(parameter)
    if subset[parameter].nunique() < 3:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, METRICS):
        ax.plot(subset[parameter], subset[metric], 'o-')
        if parameter == 'reg':
            ax.set_xscale('log')
        ticks = subset[parameter].to_numpy()
        ax.set_xticks(ticks, [f'{v:g}' for v in ticks], rotation=30)
        ax.set_xlabel(parameter)
        ax.set_ylabel(f'Validation {metric}')
        ax.set_title(_dependency_title(name, metric + ' на validation',
            PARAMETER_NAMES.get(parameter, parameter), best_params, exclude=(parameter,), width=65), fontsize=10)
        ax.grid(alpha=.25)
    return _finish(fig)


def plot_ease_objective(table):
    """Показывает минимум функционала EASE и его компоненты в зависимости от lambda."""
    needed = ['reg', 'train_objective', 'train_reconstruction', 'train_regularization']
    data = table.dropna(subset=needed).sort_values('reg')
    if len(data) < 3:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label, style in [('train_objective', 'SSE + λ‖B‖²', 'o-'),
                                  ('train_reconstruction', 'SSE восстановления', 's--'),
                                  ('train_regularization', 'Штраф λ‖B‖²', '^:')]:
        ax.plot(data.reg, data[column], style, label=label)
    ax.set_xscale('log')
    ax.set_xticks(data.reg, [f'{v:g}' for v in data.reg], rotation=30)
    ax.set_xlabel('L2-регуляризация λ')
    ax.set_ylabel('Значение лосса')
    ax.set_title(_dependency_title('EASE', 'обучающего лосса', 'L2-регуляризации'), fontsize=11)
    ax.grid(alpha=.25)
    ax.legend()
    return _finish(fig)


def plot_fm_search(table, best_params=None):
    """Показывает всю FM-сетку: строки — метрики, столбцы — learning_rate, клетки — reg × factors."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    keys = ['n_factors', 'reg', 'learning_rate']
    if table.duplicated(keys).any():
        raise ValueError('Для FM-карт остальные гиперпараметры должны быть фиксированы')
    rates = sorted(table.learning_rate.unique())
    factors = sorted(table.n_factors.unique())
    regularizations = sorted(table.reg.unique())
    if best_params is None:
        best_params = table.loc[table['NDCG@10'].idxmax(), keys].to_dict()
    fig = plt.figure(figsize=(4.2 * len(rates) + 1, 7.5))
    layout = fig.add_gridspec(2, len(rates) + 1, width_ratios=[1] * len(rates) + [.05])
    for row, metric in enumerate(METRICS):
        norm = Normalize(vmin=float(table[metric].min()), vmax=float(table[metric].max()))
        for column, rate in enumerate(rates):
            ax = fig.add_subplot(layout[row, column])
            matrix = (table[table.learning_rate == rate]
                      .pivot(index='reg', columns='n_factors', values=metric)
                      .reindex(index=regularizations, columns=factors))
            values = matrix.to_numpy(dtype=float)
            artist = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='viridis', norm=norm)
            ax.set_xticks(range(len(factors)), [str(int(f)) for f in factors])
            labels = []
            for reg in regularizations:
                exponent = int(round(np.log10(reg))) if reg > 0 else 0
                labels.append(f'$10^{{{exponent}}}$' if reg > 0 and np.isclose(reg, 10.**exponent,
                              rtol=1e-8, atol=0) else f'{reg:g}')
            ax.set_yticks(range(len(regularizations)), labels)
            ax.set_xlabel('Число факторов')
            ax.set_ylabel('L2-регуляризация' if column == 0 else '')
            fixed = _fixed_parameters(table, exclude=('reg', 'n_factors'))
            fixed['learning_rate'] = rate
            ax.set_title(_dependency_title('', metric + ' на validation',
                'числа факторов и L2-регуляризации', fixed, width=46), fontsize=10)
            for (i, j), value in np.ndenumerate(values):
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=10,
                            bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 2})
            if rate == best_params['learning_rate']:
                i = regularizations.index(best_params['reg'])
                j = factors.index(best_params['n_factors'])
                ax.add_patch(Rectangle((j - .48, i - .48), .96, .96,
                                       fill=False, edgecolor='darkorange', linewidth=3))
        color_axis = fig.add_subplot(layout[row, -1])
        fig.colorbar(artist, cax=color_axis, label=metric)
    fig.suptitle('FM: зависимость validation-метрик от гиперпараметров', fontsize=12)
    return _finish(fig)
