# Пять рекомендательных моделей на KuaiRec

**Автор**: Акопова Елена Николаевна

## Задача
В данном репозитории представлена работа по построению пяти рекомендательных систем: UserKNN, iALS, EASE, FM, iALS и их апробация на датасете по взаимодействию людей с короткими видео KuaiRec.

Результаты на двух тестовых наборах для всех реализованных моделей с лучшими гиперпараметрами, подобраннными по валидационной выборке. Подробнее с ними можно ознакомиться в [ноутбуке с экпериментами](hw1/experiments.ipynb).
![Число попаданий в top-10 на big_matrix и small_matrix](https://github.com/user-attachments/assets/4fb4a585-f1dd-44c8-8acf-4e3b670ea9b3)

Код моделей — в [models/](hw1/models/), метрики и графики — в [utils/](hw1/utils/).
Источник данных указан в [DATA.md](hw1/DATA.md).


## Запуск

Нужен Python 3.11. Скачайте из [папки KuaiRec](https://drive.google.com/drive/folders/1bAm07YnKRKB6SVHB8Mqz6v6W8ppLaJwB)
файлы big_matrix.csv, small_matrix.csv, item_categories.csv, user_features.csv,
item_daily_features.csv и social_network.csv. Положите их в data/ в корне
репозитория.

Из корня репозитория:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r hw1/requirements.txt
cd hw1
../.venv/bin/python -m notebook experiments.ipynb
```

Выберите ядро созданного окружения и выполните ячейки по порядку.
Если CSV лежат в другом месте, перед запуском задайте путь:
`export KUAIREC_DATA_DIR=/path/to/data`.

