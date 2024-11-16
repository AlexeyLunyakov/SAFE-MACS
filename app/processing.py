import cv2
import cv2_processor

from PIL import Image
from ultralytics import YOLO
from collections import defaultdict
import torch.nn.functional as F

import supervision as sv
import csv

model = YOLO('./yolo11m_20.pt')
# model = YOLO('./rtdetrl_10.pt')

# ----------------------------- image processing -----------------------------

def ppe_detection(source: str, result_name: str, conf_threshold: float=0.4) -> None:
    """
    Получение предсказания модели по входной фотографии
    :параметр source: путь до оригинального файла (строка)
    :параметр result_name: путь до нового файла (строка)
    :return: лист предсказанний модели в формате supervision
    """
    # # YOLO
    yolo_predict = model(source)[0]
    yolo_predict.boxes[yolo_predict.boxes.conf >= conf_threshold]

    detections = sv.Detections.from_ultralytics(yolo_predict)
    mask = detections.confidence >= conf_threshold
    detections = detections[mask]

    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    bounding_box_annotator = sv.BoxAnnotator()
    annotated_image = cv2.imread(source)
    annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imwrite(result_name, annotated_image)
   
    return detections

# ----------------------------- video processing -----------------------------

def save_interval(start: int, end: int, overall: int) -> None:
    """
    Получение интервалов детекции СИЗ по видео
    :параметр start: начало записи (целое число)
    :параметр result_name: путь до нового файла (строка)
    :параметр overall: общее время (целое число)
    """
    with open('detections.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([start, end, overall])


def process_video(source: str, destination: str = 'result.mp4') -> None:
    """
    Получение интервалов детекции СИЗ по видео
    :параметр source: путь до оригинального файла (строка)
    :параметр destination: путь до нового файла (строка)
    """
    video_info = sv.VideoInfo.from_video_path(video_path=source)
    frames_generator = sv.get_video_frames_generator(source_path=source)
    box_annotator = sv.BoundingBoxAnnotator()

    start_interval = None
    interval_ids = set()

    with sv.VideoSink(target_path=destination, video_info=video_info, codec='h264') as sink:
        for i, frame in enumerate(frames_generator):
            print(i)
            result = model.track(frame, verbose=False, persist=True, agnostic_nms=True)[0]
            if len(result.boxes) and result.boxes.id and start_interval is None:
                print(result.boxes)
                start_interval = int(i / video_info.fps)
                interval_ids.update(result.boxes.id.cpu().tolist())
            elif start_interval:
                print('yes')
                save_interval(start_interval, int(i / video_info.fps), len(interval_ids))
                start_interval = None
                interval_ids = set()
            if len(result.boxes) and result.boxes.id:
                interval_ids.update(result.boxes.id.cpu().tolist())
            detections = sv.Detections.from_ultralytics(result)
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            sink.write_frame(frame=annotated_frame)
        if len(interval_ids):
            save_interval(start_interval, int(video_info.total_frames / video_info.fps), len(interval_ids))