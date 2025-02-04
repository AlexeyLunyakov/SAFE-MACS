import os
import cv2
import csv
from ultralytics import YOLO
import supervision as sv

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
    # YOLO
    predict = model(source)[0]
    predict.boxes[predict.boxes.conf >= conf_threshold]

    detections = sv.Detections.from_ultralytics(predict)
    mask = detections.confidence >= conf_threshold
    detections = detections[mask]

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()
    
    annotated_image = cv2.imread(source)
    annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imwrite(result_name, annotated_image)
   
    return detections

# ----------------------------- video processing -----------------------------

def ms_to_time(ms):
    """миллисекунды в формате «ЧЧ:ММ:СС.ссс»"""
    ms = int(ms)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def video_ppe_detection(source: str, result_name: str, output_folder: str, conf_threshold: float=0.4) -> None:
    """
    Получение интервалов детекции СИЗ по видео
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
        
        while video.isOpened():
            success, frame = video.read()
            if not success: break
            
            predict = model(frame)[0]
            predict.boxes[predict.boxes.conf >= conf_threshold]
            
            current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_str = ms_to_time(current_time_ms)
            
            detections = sv.Detections.from_ultralytics(predict)
            mask = detections.confidence >= conf_threshold
            detections = detections[mask]
            
            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()
            
            frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)
            cv2.putText(frame, timestamp_str, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            result.write(frame)
            
            if len(predict.boxes) > 0:
                class_ids = predict.boxes.cls.cpu().numpy().astype(int)
                unique_classes = set(class_ids)
                current_count = len(unique_classes)
                print(current_count)
                
                if current_count != last_class_count:
                    class_names = [model.names[c_id] for c_id in unique_classes]
                    csv_writer.writerow([timestamp_str, ', '.join(class_names)])
                    last_class_count = current_count

    video.release()
    result.release()

# -------------------------------------------------------------------------------