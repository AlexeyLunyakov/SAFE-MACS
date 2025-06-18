import os
import cv2
import uuid
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO, RTDETR
from roboflow_pipes import dataset_prepare
from supervision.metrics import F1Score


HOME = os.getcwd()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    
    
def train_yolo(api_key, workspace, project, ver, model_type, size, epochs, batch_size, progress_callback=None):
    """
    YOLO model training
    
    :parameter api_key: Roboflow API key (string)
    :parameter workspace_name: workspace name (string)
    :parameter project_name: project name (string)
    :parameter ver: project version (string)
    :parameter model_type: model name (string)
    :parameter size: model size (string)
    :parameter epochs: number of training epochs (string)
    :parameter batch_size: batch size (string)
    
    ------------------
    Обучение модели YOLO
    
    :parameter api_key: API-ключ Roboflow (строка)
    :parameter workspace_name: название рабочей группы (строка)
    :parameter project_name: название проекта (строка)
    :parameter ver: версия проекта (строка)
    :parameter model_type: название модели (строка)
    :parameter size: размер модели (строка)
    :parameter epochs: количество эпох обучения (строка)
    :parameter batch_size: размер батча (строка)
    """
    dataset = []

    curr_model = f"{model_type.replace('v', '')}{size[0]}"
    try:
        model = YOLO(f"{curr_model}.pt")
        
        if progress_callback:
                progress_callback("Downloading dataset...")
        _, dataset = dataset_prepare(api_key, workspace, project, ver, f"{model_type}")
        
        if progress_callback:
                progress_callback(f"Starting training: {model_type}-{size}...")
        _ = model.train(
                data=f'{dataset.location}/data.yaml',
                epochs=int(epochs),
                batch=int(batch_size),
                imgsz=640,
                device=0,
                verbose=True,
                save_dir='./models/yolo/'
            )
        if progress_callback:
                progress_callback("Training completed successfully!")
        
        valid_res = ultralytics_validation(model, dataset)
        return f"Model {model_type}-{size} train compited!", valid_res
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"Model {model_type}-{size} train failed!", ''


def train_rtdetr(api_key, workspace, project, ver, model_type, size, epochs, batch_size, progress_callback=None):
    """
    RTDETR (v1) model training
    
    :parameter api_key: Roboflow API key (string)
    :parameter workspace_name: workspace name (string)
    :parameter project_name: project name (string)
    :parameter ver: project version (string)
    :parameter size: model size (string)
    :parameter epochs: number of training epochs (string)
    :parameter batch_size: batch size (string)
    
    ------------------
    Обучение модели RTDETR (v1)
    
    :parameter api_key: API-ключ Roboflow (строка)
    :parameter workspace_name: название рабочей группы (строка)
    :parameter project_name: название проекта (строка)
    :parameter ver: версия проекта (строка)
    :parameter size: размер модели (строка)
    :parameter epochs: количество эпох обучения (строка)
    :parameter batch_size: размер батча (строка)
    """
    dataset = []
    curr_model = f"rtdetr-{size[0]}"
    try:
        model = RTDETR(f"{curr_model}.pt")
        
        if progress_callback:
                    progress_callback("Downloading dataset...")
        _, dataset = dataset_prepare(api_key, workspace, project, ver, f"{model_type}")
        
        if progress_callback:
                    progress_callback(f"Starting training: {model_type}-{size}...")
        _ = model.train(
                data=f'{dataset.location}/data.yaml',
                epochs=int(epochs),
                batch=int(batch_size),
                imgsz=640,
                device=0,
                verbose=True,
                save_dir='./models/rt_detr_v1/'
            )
        if progress_callback:
                    progress_callback("Training completed successfully!")
        valid_res = ultralytics_validation(model, dataset)
        return f"Model {curr_model} train compited!", valid_res
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"Model {model_type}-{size} train failed!", ''

def ultralytics_validation(model, dataset):
    """
    Validating Ultralytics Models

    :parameter model: model to validate metrics
    :parameter dataset: dataset
    
    ------------------
    Валидация моделей Ultralytics
    
    :parameter model: модель для валидации метрик
    :parameter dataset: набор данных
    
    """
    print('Metrics Validation Flag')
    ds_test = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/test/images",
        annotations_directory_path=f"{dataset.location}/test/labels/",
        data_yaml_path=f'{dataset.location}/data.yaml',
    )
    targets = []
    predictions = []

    for i in range(len(ds_test)):
        path, source_image, annotations = ds_test[i]

        image = Image.open(path)
        result = model.predict(image, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        targets.append(annotations)
        predictions.append(detections)
    
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )
    f1_metric = F1Score().update(predictions, targets).compute()
    
    metrics_data = f"map50-95: {mean_average_precision.map50_95:.3f},\nmap50: {mean_average_precision.map50_95:.3f}\nF1 Score: F1 Score: {f1_metric.f1_50:.3f}"
    return metrics_data


def ultralytics_inference(img_source, check_path, conf_threshold, model_type):
    """
    Ultralytics Models Inference
    
    :parameter img_source: path to original file (string)
    :parameter check_path: path to model checkpoint (string)
    :parameter conf_threshold: prediction value limit (float)
    :parameter model_type: name of processing model (string)
    
    ------------------
    Инференс моделей Ultralytics
    
    :parameter img_source: путь до оригинального файла (строка)
    :parameter check_path: путь до чекпоинта модели (строка)
    :parameter conf_threshold: ограничение значения предсказания (флоат)
    :parameter model_type: название модели-обработчика (строка)
    """
    image = Image.open(img_source)
    result_name = f'./files/{str(uuid.uuid4())}/d_{img_source}'
    
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, smart_position=True)
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    
    if model_type.find('yolo'):
        model = YOLO(check_path)
    elif model_type.find('rt'):
        model = RTDETR(check_path)
        
    result = model.predict(image, conf=conf_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imwrite(result_name, annotated_image)
    return annotated_image