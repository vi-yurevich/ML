## Постановка задачи

Я выбрал набор данных [Diabetes classification for beginner](https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures) для выполнения лабораторной работы. Требуется определить, болен ли человк диабетом или нет

## Описание работы

В ЛР 0 проведена обработка и анализ данных:

* анализ на наличие нецельных данных
* проверка типа признаков (все они оказались числовыми)
* анализ попарных зависимостей
* анализ корреляционной матрицы
* анализ распределений данных
* удаление выбросов

В ЛР 1 реализованы четыре алгоритма классического машинного обучения:

* метод k-ближайших соседей
* логистическая регрессия
* метод опорных векторов
* наивный байесовский классификатор

Каждый из них сравнивается с готовым из библиотеки scikit-learn.

## Вывод

Полученая точность 80-90% объясняется хорошей разделимостью данных, но данных в датасете достаточно мало, из-за чего провести полноценный анализ было проблематично.
