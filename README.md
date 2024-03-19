# Описание

Проект представляет собой реализацию алгоритма сокрытия данных для электронной системы здравоохранения.

## Содержание проекта

• Папка CoverImages – содержит изображения-обложки.

• Папка StegoImages – содержит стего-изображения (или изображения после встраивания).

• Папка RecoveredImages – содержит восстановленные после встраивания изображения-обложки.

• Папка ExtractedWatermarks – содержит извлеченные водяные знаки.

• Папка ReversibilityAnalysis – содержит результат соответствующего анализа

• Watermark.png – водяной знак

• EmbeddingAndExtraction.py – файл с основными и вспомогательными функциями, необходимыми для встраивания и извлечения

• metrics.py – содержит функции, рассчитывающие метрики

• main.py – файл с исполняющим кодом, содержит код, который проходит по всем изображениям в CoverImages, проводит встраивание и извлечение, сохраняет картинки каждого этапа в соответствующие папки и проводит аналитику метриками.

## Требования

• opencv-python>=4.9.0.80

• scikit-image>=0.22.0

• numpy>=1.26.4

• scipy>=1.12.0


