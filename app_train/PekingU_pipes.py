import cv2
import uuid
import torch
import numpy as np
import supervision as sv

from PIL import Image
from roboflow import Roboflow
from dataclasses import dataclass
import albumentations as A

from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from supervision.metrics import F1Score, Precision, Recall
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from roboflow_pipes import dataset_prepare

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    
id2label = {}
label2id = {}

# ----------- dataset preprocess and mAP train compute ------------------------------------------------------------------------

class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]

        image = image[:, :, ::-1]
        boxes = annotations.xyxy
        categories = annotations.class_id

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]
        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt")
        result = {k: v[0] for k, v in result.items()}
        return result

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, (height, width) in zip(target_batch, image_size_batch):
                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def additional_dataset_preprocess(model, processor, dataset, CHECKPOINT):
    
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/train",
        annotations_path=f"{dataset.location}/train/_annotations.coco.json",
    )
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/valid",
        annotations_path=f"{dataset.location}/valid/_annotations.coco.json",
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/test",
        annotations_path=f"{dataset.location}/test/_annotations.coco.json",
    )
    IMAGE_SIZE = 640
    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        size={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
    )
    pytorch_dataset_train = PyTorchDetectionDataset(ds_train, processor)
    pytorch_dataset_valid = PyTorchDetectionDataset(ds_valid, processor)
    pytorch_dataset_test = PyTorchDetectionDataset(ds_test, processor)
    
    global id2label, label2id
    
    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}
    
    return id2label, label2id, pytorch_dataset_train, pytorch_dataset_valid, pytorch_dataset_test

# ---------------------------------------------------------------------------------------------------------------------------------

def train_rtdetrv2(api_key, workspace, project, ver, model_type, size, epochs, batch_size, progress_callback=None):
    """
    RT-DETR v2 model training
    
    :parameter api_key: Roboflow API key (string)
    :parameter workspace_name: workspace name (string)
    :parameter project_name: project name (string)
    :parameter ver: project version (string)
    :parameter model_type: model name (string)
    :parameter size: model size (string)
    :parameter epochs: number of training epochs (string)
    :parameter batch_size: batch size (string)
    
    ------------------
    Обучение модели RT-DETR v2
    
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

    CHECKPOINT = f"PekingU/rtdetr_{size}"
    
    try:
        model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
        processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

        if progress_callback:
                progress_callback("Downloading dataset...")
        _, dataset = dataset_prepare(api_key, workspace, project, ver, "coco")
        
        id2label, label2id, pytorch_dataset_train, pytorch_dataset_valid, pytorch_dataset_test = additional_dataset_preprocess(model, processor, dataset, CHECKPOINT)
        
        eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label)
        
        model = AutoModelForObjectDetection.from_pretrained(
            CHECKPOINT,
            id2label=id2label,
            label2id=label2id,
            anchor_image_size=None,
            ignore_mismatched_sizes=True,
        )
        
        if progress_callback:
                progress_callback(f"Starting training: {model_type}-{size}...")
        training_args = TrainingArguments(
            output_dir=f"./rtdetr-v2-{size}-{epochs}",
            num_train_epochs=epochs,
            max_grad_norm=0.1,
            learning_rate=5e-5,
            warmup_steps=300,
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=2,
            metric_for_best_model="eval_map",
            greater_is_better=True,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            report_to="tensorboard",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pytorch_dataset_train,
            eval_dataset=pytorch_dataset_valid,
            processing_class=processor,
            data_collator=collate_fn,
            compute_metrics=eval_compute_metrics_fn,
        )
        trainer.train()
        if progress_callback:
                progress_callback("Training completed successfully!")
        
        valid_res = pekingu_validation(model, dataset)
        return f"Model {model_type}-{size} train compited!", _
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"Model {model_type}-{size} train failed!", ''


def pekingu_validation(model, dataset):
    """
    Validating RT-DETR v2 Model

    :parameter model: model to validate metrics
    :parameter dataset: dataset
    
    ------------------
    Валидация модели RT-DETR v2
    
    :parameter model: модель для валидации метрик
    :parameter dataset: набор данных
    
    """
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


def pekingu_inference(img_source, check_path, conf_threshold, model_type):
    """
    RT-DETR v2 Model Inference
    
    :parameter img_source: path to original file (string)
    :parameter check_path: path to model checkpoint (string)
    :parameter conf_threshold: prediction value limit (float)
    :parameter model_type: name of processing model (string)
    
    ------------------
    Инференс модели RT-DETR v2
    
    :parameter img_source: путь до оригинального файла (строка)
    :parameter check_path: путь до чекпоинта модели (строка)
    :parameter conf_threshold: ограничение значения предсказания (флоат)
    :parameter model_type: название модели-обработчика (строка)
    """
    # standart preprocess
    image = Image.open(img_source)
    result_name = f'./files/{str(uuid.uuid4())}/d_{img_source}'
    
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, smart_position=True)
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    
    # model load
    model = AutoModelForObjectDetection.from_pretrained(check_path).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(check_path)
    
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        
    w, h = image.size
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3)

    detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)

    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)
    cv2.imwrite(result_name, annotated_image)
    
    return annotated_image