import os
import cv2
import uuid
from PIL import Image
import supervision as sv
from roboflow import Roboflow
from supervision.metrics import F1Score
from rfdetr import RFDETRBase, RFDETRLarge

HOME = os.getcwd()


def dataset_prepare(api_key, workspace_name, project_name, ver, export_type):
    """
    Preparing a dataset for model training
    :api_key parameter: Roboflow API key (string)
    :workspace_name parameter: workspace name (string)
    :project_name parameter: project name (string)
    :ver parameter: project version (string)
    :export_type parameter: export format (string)
    ------------------
    Подготовка датасета под обучение модели
    :параметр api_key: API-ключ Roboflow (строка)
    :параметр workspace_name: название рабочей группы (строка)
    :параметр project_name: название проекта (строка)
    :параметр ver: версия проекта (строка)
    :параметр export_type: формат экспорта (строка)
    """
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(ver)
    dataset = version.download(export_type)
    status = "Import completed"
        
    return status, dataset


def train_rfdetr(api_key, workspace, project, ver, model_type, size, epochs, batch_size, progress_callback=None):
    """
    Roboflow DETR model training
    
    :api_key parameter: Roboflow API key (string)
    :workspace_name parameter: workspace name (string)
    :project_name parameter: project name (string)
    :ver parameter: project version (string)
    :size parameter: model size (string)
    :epochs parameter: number of training epochs (string)
    :batch_size parameter: batch size (string)
    
    ------------------
    Обучение модели Roboflow DETR
    
    :параметр api_key: API-ключ Roboflow (строка)
    :параметр workspace_name: название рабочей группы (строка)
    :параметр project_name: название проекта (строка)
    :параметр ver: версия проекта (строка)
    :параметр size: размер модели (строка)
    :параметр epochs: количество эпох обучения (строка)
    :параметр batch_size: размер батча (строка)
    """
    dataset = []
    curr_model = f"rfdetr-{size}"
    try:
        if size == 'base':
            model = RFDETRBase()
        elif size == 'large':
            model = RFDETRLarge()
        
        if progress_callback:
                    progress_callback("Downloading dataset...")
        _, dataset = dataset_prepare(api_key, workspace, project, ver, curr_model)
        
        if progress_callback:
                    progress_callback(f"Starting training: {model_type}-{size}...")
        model.train(
            dataset_dir=dataset.location, 
            epochs=int(epochs), 
            batch_size=int(batch_size), 
            grad_accum_steps=2, 
            lr=1e-4, 
            early_stopping=True,
            save_dir='./models/rf_detr/'
        )
        if progress_callback:
                    progress_callback("Training completed successfully!")
        valid_res = roboflow_validation(model, dataset)
        return f"Model {curr_model} train compited!", valid_res
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"Model {model_type}-{size} train failed!", ''


def roboflow_validation(model, dataset):
    """
    Validating Roboflow Models

    :parameter model: model to validate metrics
    :parameter dataset: dataset
    
    ------------------
    Валидация моделей Roboflow
    
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
        detections = model.predict(image, conf=0.3)

        targets.append(annotations)
        predictions.append(detections)
    
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )
    f1_metric = F1Score().update(predictions, targets).compute()
    
    metrics_data = f"map50-95: {mean_average_precision.map50_95:.3f},\nmap50: {mean_average_precision.map50_95:.3f}\nF1 Score: F1 Score: {f1_metric.f1_50:.3f}"

    return metrics_data

def roboflow_inference(img_source, check_path, conf_threshold, model_type):
    """
    Roboflow Models Inference
    
    :parameter img_source: path to original file (string)
    :parameter check_path: path to model checkpoint (string)
    :parameter conf_threshold: prediction value limit (float)
    :parameter model_type: name of processing model (string)
    
    ------------------
    Инференс моделей Roboflow
    
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
    
    if model_type.find('base'):
        model = RFDETRBase(pretrain_weights=check_path)
    elif model_type.find('large'):
        model = RFDETRLarge(pretrain_weights=check_path)
        
    detections = model.predict(image, threshold=conf_threshold)
    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)
    cv2.imwrite(result_name, annotated_image)
    
    return annotated_image