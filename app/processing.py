import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import supervision as sv
from ultralytics import YOLO

if torch.cuda.is_available():
    from rfdetr import RFDETRBase
    from ultralytics import RTDETR

MODEL_CACHE = {}
CLASS_LIST = ['Coverall', 'Face_Shield', 'Gloves', 'Googles', 'Mask']

# ----------------------------- image processing -----------------------------

def img_ppe_detection(source: str, result_name: str, conf_threshold: float=0.4, model_type: str='YOLOv11-L', class_names = ['car', 'bus']) -> tuple:
    """
    Obtaining model prediction from input photo
    
    :parameter source: path to original file (string)
    :parameter result_name: path to new file (string)
    :parameter conf_threshold: limit of prediction value (float)
    :parameter model_type: model handler name (string)
    :parameter class_names: limit of prediction classes (list)
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
        elif model_key == 'RT' and torch.cuda.is_available():
            MODEL_CACHE[model_key] = RTDETR('./rtdetrl_50.pt')
        elif model_key == 'RF' and torch.cuda.is_available():
            MODEL_CACHE[model_key] = RFDETRBase(pretrain_weights='./rfdetr_base_20.pth')
    model = MODEL_CACHE[model_key]
    
    start_time = time.time()
    if model_key == 'RF':
        detections = model.predict(pil_image, threshold=0)
        detections.class_id = detections.class_id - 1
    elif model_key == 'RT' or model_key == 'YOLOv11':
        results = model(pil_image)[0]
        detections = sv.Detections.from_ultralytics(results)
    
    target_class_ids = [CLASS_LIST.index(name) for name in class_names if name in CLASS_LIST]

    mask = np.isin(detections.class_id, target_class_ids)
    detections = detections[mask]
    
    mask = detections.confidence >= conf_threshold
    detections = detections[mask]
        
    labels = [
        f"{CLASS_LIST[class_num]} {confidence:.2f}"
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


def video_ppe_detection(source: str, result_name: str, output_folder: str, conf_threshold: float=0.5, model_type: str='YOLOv11-L', class_names = ['car', 'bus']) -> None:
    """
    Obtaining model prediction from input video
    
    :parameter source: path to original file (string)
    :parameter result_name: path to new file (string)
    :parameter output_folder: path to file storage folder (string)
    :parameter conf_threshold: prediction value limit (float)
    :parameter model_type: model handler name (string)
    :parameter class_names: limit of prediction classes (list)
    :return: list of model predictions in supervision format, processing time dataframe
    
    ------------------
    Получение предсказания модели по входному видео
    
    :parameter source: путь до оригинального файла (строка)
    :parameter result_name: путь до нового файла (строка)
    :parameter output_folder: путь до папки хранения файлов (строка)
    :parameter conf_threshold: ограничение значения предсказания (флоат)
    :parameter model_type: название модели-обработчика (строка)
    """
    model_key = model_type.split('-')[0]
    if model_key not in MODEL_CACHE:
        if model_key == 'YOLOv11':
            MODEL_CACHE[model_key] = YOLO('./yolo11l_75.pt')
        elif model_key == 'RT' and torch.cuda.is_available():
            MODEL_CACHE[model_key] = RTDETR('./rtdetrl_50.pt')
        elif model_key == 'RF' and torch.cuda.is_available():
            MODEL_CACHE[model_key] = RFDETRBase(pretrain_weights='./rfdetr_base_20.pth')
    model = MODEL_CACHE[model_key]
    
    video = cv2.VideoCapture(source)
    if not video.isOpened(): exit()
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_info = sv.VideoInfo.from_video_path(source)
    
    detection_data = []
    start_time = time.time()
    
    with sv.VideoSink(target_path = result_name, video_info = video_info, codec="mp4v") as sink:
        for frame_idx in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break
            
            if model_key == 'RF':
                detections = model.predict(frame, threshold=conf_threshold)
                detections.class_id = detections.class_id - 1
            else:
                results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
            
            target_class_ids = [CLASS_LIST.index(name) for name in class_names if name in CLASS_LIST]
            mask = np.isin(detections.class_id, target_class_ids)
            detections = detections[mask]
            
            labels = [
                    f"{CLASS_LIST[class_num]} {confidence:.2f}"
                    for class_num, confidence
                    in zip(detections.class_id, detections.confidence)
                ]
            
            annotated_frame = frame.copy()
            annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
            annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
            
            detection_data.append({
                'frame': frame_idx,
                'timestamp': frame_idx/fps,
                'class_counts': pd.Series(detections.class_id).value_counts().to_dict(),
                'boxes': detections.xyxy,
                'confidence': detections.confidence
            })
            
            current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_str = ms_to_time(current_time_ms)
            
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