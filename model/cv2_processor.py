import cv2
import numpy


def crop(path_to_image: str, yolo_predict) -> numpy.ndarray:
    """
    Получение исходного изображения и возврат обрезанного изображения (bbox из yolo)
    :параметр path_to_image: путь до файла (строка)
    :параметр yolo_predict: предсказание модели
    :return: массив для обрезанного изображения
    """
    image = cv2.imread(path_to_image)
    height, width, _ = image.shape
    try:
        x_min, y_min, x_max, y_max = yolo_predict[0].boxes.xyxyn[0]
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)
        w = x_max - x_min
        h = y_max - y_min
        crop_img = yolo_predict[0].orig_img[y_min:y_min + h, x_min:x_min + w]  # np.array
        return crop_img
    except:
        return None


def draw_boxes(path_to_image: str, yolo_predict) -> numpy.ndarray:
    """
    Получение исходного изображения и возврат изображения с визуализированными bbox (координатый из предсказания модели)
    :параметр path_to_image: путь до файла (строка)
    :параметр yolo_predict: предсказание модели
    :return: массив с визуализированными bbox-ами
    """
    image = cv2.imread(path_to_image)
    height, width, _ = image.shape
    x_min, y_min, x_max, y_max = yolo_predict[0].boxes.xyxyn[0]
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    # цвет и толщина ограничивающей рамки
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
    return image