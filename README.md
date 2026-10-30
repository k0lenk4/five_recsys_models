# Рекомендательные системы — домашнее задание 1

**ФИО студента:** Акопова Елена Николаевна.

Пять самостоятельно реализованных моделей top-10 рекомендаций на **KuaiRec**:
UserKNN, iALS, EASE, FM с категориями видео и iALS с собственным целевым сигналом.
Работа находится в [hw1/](hw1/), ветка для сдачи — main.
[Условие задания](https://docs.google.com/document/d/14s3DslYRrrzBeAOyZA0Stbd-IqXih-8EdnF84P43dRc/mobilebasic).

## Содержание

| Путь | Назначение |
|---|---|
| [hw1/DATA.md](hw1/DATA.md) | ФИО, описание, источник и получение данных |
| [hw1/experiments.ipynb](hw1/experiments.ipynb) | Выполненный основной ноутбук |
| [hw1/models/](hw1/models/) | userknn.py, als.py, ease.py, fm.py, custom_als.py; общие функции в common.py |
| [hw1/utils/evaluation.py](hw1/utils/evaluation.py) | Метрики и validation-перебор |
| [hw1/utils/visualization.py](hw1/utils/visualization.py) | Графики обучения и качества |
| [hw1/requirements.txt](hw1/requirements.txt) | Версии зависимостей |

## Запуск

Проверено на Python 3.11.17. Из корня репозитория:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r hw1/requirements.txt
~~~

Скачайте шесть CSV из [официальной папки KuaiRec](https://drive.google.com/drive/folders/1bAm07YnKRKB6SVHB8Mqz6v6W8ppLaJwB)
по инструкции в [DATA.md](hw1/DATA.md) и сохраните в локальную папку data/ в корне.
Для другого расположения задайте абсолютный путь:

~~~bash
export KUAIREC_DATA_DIR=/path/to/KuaiRec/data
~~~

Для data/ в корне задавать переменную не нужно. Данные исключены из Git.

~~~bash
cd hw1
../.venv/bin/python -m notebook experiments.ipynb
~~~

Выберите ядро созданного окружения, перезапустите его и выполните ячейки сверху
вниз. Полный поиск содержит 193 конфигурации и один контрольный запуск CustomALS.
Сумма времени ранее выполненных конфигураций — около 74 минут без EDA и финальной
оценки; фактическое время зависит от оборудования и BLAS. CSV загружаются в память.
Таблицы, параметры и обученные модели хранятся в памяти ядра; графики и результаты
остаются в выводах ноутбука. Отдельные файлы результатов не читаются и не создаются.

Выполнение без интерфейса из hw1/:

~~~bash
../.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 experiments.ipynb
~~~

## Протокол и результаты

Зерно — 42. Из очищенной big_matrix выбираются 2200 пользователей и 4700 видео.
Временные границы — квантили 0,70 и 0,85. Train: 976 768 пар, 2199 пользователей,
3129 видео. Каталог фиксируется по train; новые ID исключаются. Повторные пары
усредняются внутри периодов. На validation исключаются train-просмотры, на test —
train и validation. Модели обучаются только на train, настройки выбираются по
**validation NDCG@10**. Общая релевантность — средний watch_ratio >= 0.8 у отложенной
пары. Основной test: 11 897 релевантных пар для 1803 пользователей.

| Модель | Test NDCG@10 | Test Recall@10 | Лучшие параметры validation |
|---|---:|---:|---|
| iALS | 0.017938 | 0.018994 | factors=8, reg=0.01, iter=10 |
| CustomALS | 0.010869 | 0.012757 | factors=32, reg=0.01, iter=10, power=0.5 |
| EASE | 0.007427 | 0.008638 | reg=3000 |
| UserKNN | 0.002842 | 0.002939 | cosine, k=5 |
| FM | 0.000258 | 0.000101 | factors=32, reg=0.001, lr=0.001, epochs=10 |

Для обеих ALS alpha=10; для FM batch_size=2048. В контроле при одинаковых
параметрах ALS непрерывный таргет даёт NDCG@10=0.010988 против 0.017938 у бинарного.
На validation лучшим был CustomALS, на test — iALS. Отдельная абляция категорий FM
не проводилась, поэтому их самостоятельный вклад не измерен.

Основной test взят из big_matrix. Протокол уточнялся после первичной оценки
small_matrix, что ограничивает независимость итоговой проверки. Дополнительный
small_matrix-test (296 пар, 189 пользователей) даёт нулевые метрики всех моделей.

## Источники

[KuaiRec](https://github.com/chongminggao/KuaiRec), [страница проекта](https://kuairec.com/),
[официальный EDA](https://kuairec.com/Statistics_KuaiRec.html).
Учебные материалы: [girafe-ai/recsys, ветка 25f_msu](https://github.com/girafe-ai/recsys/tree/25f_msu).
Лицензия материалов: [Apache-2.0](https://github.com/girafe-ai/recsys/blob/25f_msu/LICENSE).
Обучение, предсказания и рекомендации реализованы на NumPy/SciPy без готовых
рекомендательных моделей.
