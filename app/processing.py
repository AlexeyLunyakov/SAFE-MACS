import os
import cv2
import csv
from ultralytics import YOLO, RTDETR
import supervision as sv

model_1 = YOLO('./yolo11x_100.pt')
# model_1 = YOLO('./yolo11m_20.pt')
model_0 = RTDETR('./rtdetrl_10.pt')

# ----------------------------- image processing -----------------------------

def img_ppe_detection(source: str, result_name: str, conf_threshold: float=0.4, model_type: int=1) -> None:
    """
    Получение предсказания модели по входной фотографии
    :параметр source: путь до оригинального файла (строка)
    :параметр result_name: путь до нового файла (строка)
    :return: лист предсказанний модели в формате supervision
    """
    
    label_annotator = sv.LabelAnnotator()
    bbox_annotator = sv.BoxAnnotator()
    annotated_image = cv2.imread(source)
    
    if model_type:
        # YOLO
        predict = model_1(source)[0]
        predict.boxes[predict.boxes.conf >= conf_threshold]
        detections = sv.Detections.from_ultralytics(predict)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
    else:
        # RT-DETR
        results = model_0(source)[0]
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(), 
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int))
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections,
            labels=[model_0.model.names[class_id] for class_id in detections.class_id]
        )
        
    mask = detections.confidence >= conf_threshold
    detections = detections[mask]
    
    annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=detections)
    
    cv2.imwrite(result_name, annotated_image)
    # print(detections)
    return detections

# ----------------------------- video processing -----------------------------

def ms_to_time(ms):
    """миллисекунды в формате «ЧЧ:ММ:СС.ссс»"""
    ms = int(ms)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def video_ppe_detection(source: str, result_name: str, output_folder: str, conf_threshold: float=0.5, model_type: int=1) -> None:
    """
    Получение предсказания модели по входному видео
    :параметр source: путь до оригинального файла (строка)
    :параметр result_name: путь до нового файла (строка)
    """
    video = cv2.VideoCapture(source)
    if not video.isOpened(): print("Error opening video file"); exit()
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(result_name, fourcc, fps, (frame_width, frame_height))
    
    csv_filename = os.path.join(output_folder, 'detections.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['time', 'found_classes'])
        
        last_class_count = None
        label_annotator = sv.LabelAnnotator()
        bbox_annotator = sv.BoxAnnotator()
        
        while video.isOpened():
            success, frame = video.read()
            if not success: break
            
            if model_type:
                predict = model_1(frame)[0]
                predict.boxes[predict.boxes.conf >= conf_threshold]
                detections = sv.Detections.from_ultralytics(predict)
                mask = detections.confidence >= conf_threshold
                detections = detections[mask]
                frame = label_annotator.annotate(scene=frame, detections=detections)
            else:
                predict = model_0(frame)[0]
                detections = sv.Detections(
                    xyxy=predict.boxes.xyxy.cpu().numpy(), 
                    confidence=predict.boxes.conf.cpu().numpy(),
                    class_id=predict.boxes.cls.cpu().numpy().astype(int))
                mask = detections.confidence >= conf_threshold
                detections = detections[mask]
                frame = label_annotator.annotate(
                    scene=frame, 
                    detections=detections,
                    labels=[model_0.model.names[class_id] for class_id in detections.class_id]
                )
            
            current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_str = ms_to_time(current_time_ms)
            
            frame = bbox_annotator.annotate(scene=frame, detections=detections)
            
            cv2.putText(frame, timestamp_str, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            result.write(frame)
            
            if len(predict.boxes) > 0:
                class_ids = predict.boxes.cls.cpu().numpy().astype(int)
                unique_classes = set(class_ids)
                current_count = len(unique_classes)
                
                if current_count != last_class_count:
                    class_names = [model_1.names[c_id] for c_id in unique_classes]
                    csv_writer.writerow([timestamp_str, ', '.join(class_names)])
                    last_class_count = current_count

    video.release()
    result.release()

# -------------------------------------------------------------------------------