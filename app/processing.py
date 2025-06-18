import os
import cv2
import csv
import time
import numpy as np
import pandas as pd
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR


MODEL_CACHE = {}

# ----------------------------- image processing -----------------------------

def img_ppe_detection(
    source: str, 
    result_name: str, 
    conf_threshold: float=0.4, 
    model_type: str='YOLOv11-L'
) -> tuple:
    """
    Obtaining model prediction from input photo
    
    :parameter source: path to original file (string)
    :parameter result_name: path to new file (string)
    :parameter conf_threshold: limit of prediction value (float)
    :parameter model_type: model handler name (string)
    :return: list of model predictions in supervision format, processing time dataframe
    
    ------------------
    Получение предсказания модели по входной фотографии
    
    :parameter source: путь до оригинального файла (строка)
    :parameter result_name: путь до нового файла (строка)
    :parameter conf_threshold: ограничение значения предсказания (флоат)
    :parameter model_type: название модели-обработчика (строка)
    :return: лист предсказанний модели в формате supervision, датафрейм времени обработки
    """
    pil_image = Image.open(source)
    annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    resolution = pil_image.size
    
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, smart_position=True)
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    
    model_key = model_type.split('-')[0]
    if model_key not in MODEL_CACHE:
        if model_key == 'YOLOv11':
            MODEL_CACHE[model_key] = YOLO('./yolo11l_75.pt')
        elif model_key == 'RT':
            MODEL_CACHE[model_key] = RTDETR('./rtdetrl_50.pt')
        elif model_key == 'RF':
            MODEL_CACHE[model_key] = RFDETRBase(pretrain_weights='./rfdetr_base_20.pth')
    model = MODEL_CACHE[model_key]
    
    start_time = time.time()
    
    class_names = ['Coverall', 'Face_Shield', 'Gloves', 'Googles', 'Mask']
    
    if model_key == 'RF':
        detections = model.predict(pil_image, threshold=0)
        class_key = 1
    elif model_key == 'RT' or model_key == 'YOLOv11':
        results = model(pil_image)[0]
        detections = sv.Detections.from_ultralytics(results)
        class_key = 0

    mask = detections.confidence >= conf_threshold
    detections = detections[mask]
    
    labels = [
        f"{class_names[class_num - class_key]} {confidence:.2f}"
        for class_num, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=detections)
    
    cv2.imwrite(result_name, annotated_image)
    elapsed_time = time.time() - start_time
    
    time_df = pd.DataFrame({
        'filename': [os.path.basename(source)],
        'model_type': [model_type],
        'processing_time': [f'{elapsed_time:.3f} sec'],
    })
    
    return detections, time_df

# ----------------------------- video processing -----------------------------

def ms_to_time(ms):
    """
    milliseconds in the format "HH:MM:SS.sss"
    """
    ms = int(ms)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def video_ppe_detection(source: str, result_name: str, output_folder: str, conf_threshold: float=0.5, model_type: str='YOLOv11-L') -> None:
    """
    Obtaining model prediction from input video
    
    :parameter source: path to original file (string)
    :parameter result_name: path to new file (string)
    :parameter output_folder: path to file storage folder (string)
    :parameter conf_threshold: prediction value limit (float)
    :parameter model_type: model handler name (string)
    
    ------------------
    Получение предсказания модели по входному видео
    
    :parameter source: путь до оригинального файла (строка)
    :parameter result_name: путь до нового файла (строка)
    :parameter output_folder: путь до папки хранения файлов (строка)
    :parameter conf_threshold: ограничение значения предсказания (флоат)
    :parameter model_type: название модели-обработчика (строка)
    """
    if model_type.find('YOLO'):
        model = YOLO('./yolo11l_75.pt')
    elif model_type.find('RT'):
        model = RTDETR('./rtdetrl_50.pt')
    elif model_type.find('RF'):
        model = RFDETRBase(pretrain_weights='./rfdetr_base_20.pth')
    
    # Video setup
    video = cv2.VideoCapture(source)
    if not video.isOpened(): exit()
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_info = sv.VideoInfo.from_video_path(source)
    
    detection_data = []
    start_time = time.time()
    
    with sv.VideoSink(target_path = result_name, video_info = video_info) as sink:
        for frame_idx in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            detection_data.append({
                'frame': frame_idx,
                'timestamp': frame_idx/fps,
                'class_counts': pd.Series(detections.class_id).value_counts().to_dict(),
                'boxes': detections.xyxy,
                'confidence': detections.confidence
            })
            
            current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_str = ms_to_time(current_time_ms)
            
            annotated_frame = results.plot()
            cv2.putText(annotated_frame, timestamp_str, (10, height - 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            sink.write_frame(annotated_frame)
    
    video.release()
        
    elapsed_time = time.time() - start_time
        
    time_df = pd.DataFrame({
        'filename': [os.path.split(source)[1]],
        'model_type': [model_type],
        'processing_time': [f'{elapsed_time:.3f} sec'],
    })
    
    return detection_data, time_df 

# -------------------------------------------------------------------------------