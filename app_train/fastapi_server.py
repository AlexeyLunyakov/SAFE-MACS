import os
import uuid
import logging
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form

from PekingU_pipes import train_rtdetrv2, pekingu_inference
from ultralytics_pipes import *
from roboflow_pipes import *


app = FastAPI()
tasks = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

TRAINING_FUNCTIONS = {
    "yolov10": train_yolo,
    "yolov11": train_yolo,
    "yolov12": train_yolo,
    "rt-detr v1": train_rtdetr,
    "rt-detr v2": train_rtdetrv2,
    "rf-detr": train_rfdetr,
}

INFERENCE_FUNCTIONS = {
    "yolov10": ultralytics_inference,
    "yolov11": ultralytics_inference,
    "yolov12": ultralytics_inference,
    "rt-detr v1": train_rtdetr,
    "rt-detr v2": pekingu_inference,
    "rf-detr": roboflow_inference,
}

class TrainRequest(BaseModel):
    model_name: str = "yolov11", 
    model_size: str = "nano", 
    api_key: str = "0000", 
    workspace: str = "safemacsws", 
    project: str = "mppe-custom-set", 
    version: str = "4", 
    epochs: str = "5", 
    batch_size: str = "8",

@app.post("/start_training")
async def start_training(
    background_tasks: BackgroundTasks,
    model_name: str = Form("yolov11"), 
    model_size: str = Form("nano"), 
    api_key: str = Form("0000"), 
    workspace: str = Form("safemacsws"), 
    project: str = Form("mppe-custom-set"), 
    version: str = Form("4"), 
    epochs: str = Form("1"), 
    batch_size: str = Form("8"),
):  
    task_id = str(uuid.uuid4())
    os.makedirs("models", exist_ok=True)
    
    tasks[task_id] = {
        "status": "running",
        "logs": [],
        "result": None
    }
    
    if model_name not in TRAINING_FUNCTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Selected model not implemented yet"
        )

    def training_process():
        try:
            logs = []
            def progress_callback(message):
                logger.info(f"[{task_id}] {message}")
                logs.append(message)
                tasks[task_id]["logs"] = logs.copy()

            train_func = TRAINING_FUNCTIONS[model_name]
            
            status, metrics = train_func(
                api_key, workspace, project, version, 
                model_name, model_size, epochs, batch_size,
                progress_callback=progress_callback
            )
            
            tasks[task_id].update({
                "status": "completed" if status else "failed",
                "result": str(metrics) if status else f"Training failed: {metrics}"
            })
            
        except Exception as e:
            error_msg = f"Critical error: {str(e)}"
            logger.error(error_msg)
            tasks[task_id].update({
                "status": "failed",
                "result": f"Train error: {str(e)}"
            })
    
    background_tasks.add_task(training_process)
    return {"task_id": task_id}

# @app.post("/start_inference")
# async def start_inference(
#     background_tasks: BackgroundTasks,
#     model_name: str = Form("yolov11"), 
#     model_size: str = Form("nano"), 
#     api_key: str = Form("0000"), 
#     workspace: str = Form("safemacsws"), 
#     project: str = Form("mppe-custom-set"), 
#     version: str = Form("4"), 
#     epochs: str = Form("1"), 
#     batch_size: str = Form("8"),
# ):  
#     task_id = str(uuid.uuid4())
#     os.makedirs("models", exist_ok=True)
    
#     tasks[task_id] = {
#         "status": "running",
#         "logs": [],
#         "result": None
#     }
    
#     if model_name not in TRAINING_FUNCTIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Selected model not implemented yet"
#         )

#     def training_process():
#         try:
#             logs = []
#             def progress_callback(message):
#                 logger.info(f"[{task_id}] {message}")
#                 logs.append(message)
#                 tasks[task_id]["logs"] = logs.copy()

#             train_func = TRAINING_FUNCTIONS[model_name]
            
#             status, metrics = train_func(
#                 api_key, workspace, project, version, 
#                 model_name, model_size, epochs, batch_size,
#                 progress_callback=progress_callback
#             )
            
#             tasks[task_id].update({
#                 "status": "completed" if status else "failed",
#                 "result": str(metrics) if status else f"Training failed: {metrics}"
#             })
            
#         except Exception as e:
#             error_msg = f"Critical error: {str(e)}"
#             logger.error(error_msg)
#             tasks[task_id].update({
#                 "status": "failed",
#                 "result": f"Train error: {str(e)}"
#             })
    
#     background_tasks.add_task(training_process)
#     return {"task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    return tasks.get(task_id, {"error": "Task not found"})