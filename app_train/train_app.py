import gradio as gr
import time
import os
import requests
from pathlib import Path
from typing import Dict
from roboflow_pipes import *

filepath = "./files/"

result_filenames = []

custom_theme = gr.themes.Monochrome(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('JetBrains Mono'), gr.themes.GoogleFont('Limelight'), 'sans-serif'],
).set(
    block_radius='*radius_xxl',
    button_large_radius='*radius_xl',
    button_large_text_size='*text_md',
    button_small_radius='*radius_xl',
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")

def run_training(model_name, model_size, api_key, workspace, project, version, epochs, batch_size):
    
    request_data = {
        "model_name": model_name,
        "model_size": model_size,
        "api_key": api_key,
        "workspace": workspace,
        "project": project,
        "version": version,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    
    if not all([api_key, workspace, project, version]):
        yield "Missing Roboflow credentials", {}
        return
    
    response = requests.post(
        f"{BACKEND_URL}/start_training",
        data=request_data,
    )
    
    response_data = response.json()
    task_id = response_data.get("task_id")

    if not task_id:
        yield "Error starting task", ""
        return
    
    while True:
        status_response = requests.get(f"{BACKEND_URL}/status/{task_id}")
        if status_response.status_code != 200:
            yield task_id, f"Backend error: {status_response.text}"
            return
            
        data = status_response.json()
        status = data.get("status")
        logs = data.get("logs", [])
        result = data.get("result")

        output = []
        if logs:
            output.extend(logs)
            
        if status == "completed":
            output.append("\nTraining Result:\n" + str(result))
            yield task_id, "\n".join(output)
            return
        elif status == "failed":
            output.append("\nTraining Failed:\n" + str(result))
            yield task_id, "\n".join(output)
            return
        else:
            yield task_id, "Training started! Task ID: " + task_id + "\n\nWaiting for updates..."
            
        time.sleep(20)

def run_inference(model_type, model_size, checkpoint_path, threshold, input_image):
    
    request_data = {
        "model_name": model_type,
        "model_size": model_size,
        "check_path": checkpoint_path,
        "conf_threshold": threshold,
    }
    
    response = requests.post(
        f"{BACKEND_URL}/start_inference",
        data=request_data,
    )
    
    response_data = response.json()
    return response_data

MODEL_CONFIG = {
    "yolov12": {"sizes": ["nano", "small", "medium", "large"]},
    "yolov11": {"sizes": ["nano", "small", "medium", "large", "xlarge"]},
    "yolov10": {"sizes": ["nano", "small", "medium", "large", "xlarge"]},
    "rt-detr v1": {"sizes": ["large", "xlarge"]},
    "rt-detr v2": {"sizes": ["r18vd_coco_o365", "r50vd_coco_o365", "r101vd_coco_o365"]},
    "rf-detr": {"sizes": ["base", "large"]},
    # "rtmdet": {"sizes": ["tiny", "small", "medium", "large", "xlarge"]}
}

def update_model_sizes(model_name):
    return gr.Dropdown(choices=MODEL_CONFIG[model_name]["sizes"], value=MODEL_CONFIG[model_name]["sizes"][0])


with gr.Blocks(theme=custom_theme, title='CVM-TS') as demo:
    gr.Markdown("""<a name="readme-top"></a>
                    <h1 style="text-align:center;line-height: 0.3;" ><font size="30px"><strong style="font-family: Limelight">CVM-TS</strong></font></h1>
                    <p style="text-align:center;color:#0FC28F;line-height: 0.1;font-weight: bold;">Computer Vision Model Training Studio</p>
                    <p align="center"><font size="4px">Application for training models for computer vision tasks using Roboflow datasets<br></font></p>
                    <p align="center"></p>""")
    
    with gr.Tabs():
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column():
                    # Models' Settings
                    with gr.Accordion("Model Settings", open=True):
                        model_name = gr.Dropdown(
                            label="Model Architecture",
                            choices=list(MODEL_CONFIG.keys()),
                            value="yolov11"
                        )
                        model_size = gr.Dropdown(
                            label="Model Size",
                            choices=MODEL_CONFIG["yolov11"]["sizes"],
                            value="nano"
                        )
                    
                    # Roboflow Settings
                    with gr.Accordion("Roboflow Dataset Settings", open=False):
                        api_key = gr.Textbox(label="API Key", type="password", value="pEbpvVmHCmE4sFlvI8Og")
                        workspace = gr.Textbox(label="Workspace", value="safemacsws")
                        project = gr.Textbox(label="Project Name", value="mppe-custom-set")
                        version = gr.Textbox(label="Version Number", value="4")
                        formats = gr.Textbox(label="Export option", value="yolov11", interactive=True)
                        with gr.Row(equal_height=True):
                            dataset_btn = gr.Button('Prepare Dataset', variant="primary" )
                            status = gr.Textbox(value="Not Prepared", show_label=False, container=False)
                            tmp = gr.Textbox(render=False)
                    
                    # Hyperparameters
                    with gr.Accordion("Training Hyperparameters (WIP)", open=False):
                        epochs = gr.Slider(
                            label="Epochs", 
                            minimum=1, 
                            maximum=200, 
                            value=1,
                            step=1
                        )
                        batch_size = gr.Slider(
                            label="Batch Size", 
                            minimum=1, 
                            maximum=256, 
                            value=8, 
                            step=1
                        )
                    
                    train_btn = gr.Button("Start Training", variant="primary")
                
                with gr.Column():
                    progress_output = gr.Code(
                        label="Training Progress", 
                        interactive=False
                    )
                    metrics_output = gr.Code(
                        label="Training Metrics",
                        interactive=False
                    )
    
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Model Settings", open=True):
                        inf_model_name = gr.Dropdown(
                            label="Model Architecture",
                            choices=list(MODEL_CONFIG.keys()),
                            value="yolov11"
                        )
                        inf_model_size = gr.Dropdown(
                            label="Model Size",
                            choices=MODEL_CONFIG["yolov11"]["sizes"],
                            value="medium"
                        )
                        checkpoint = gr.Textbox(
                            label="Checkpoint Path", 
                            value="models/checkpoint.pth",
                            placeholder="Path to model checkpoint"
                        )
                        threshold = gr.Slider(
                            label="Prediction Value Limit", 
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.5, 
                            step=0.05
                        )
                    infer_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column():    
                    image_input = gr.Image(label="Input Image", type="filepath")
                    image_output = gr.Image(label="Detection Results")
    
    with gr.Row():
        gr.Markdown("""<p align="center"><a href="https://github.com/AlexeyLunyakov"><nobr>Created solo by Alexey Lunyakov</nobr></a></p>""")
    
    # Event handlers
    model_name.change(
        update_model_sizes,
        inputs=model_name,
        outputs=model_size
    )
    
    inf_model_name.change(
        update_model_sizes,
        inputs=inf_model_name,
        outputs=inf_model_size
    )
    
    dataset_btn.click(
        dataset_prepare,
        inputs=[
            api_key,
            workspace,
            project,
            version,
            formats
        ],
        outputs=[status,tmp]
    )
    
    train_btn.click(
        run_training,
        inputs=[
            model_name,
            model_size,
            api_key,
            workspace,
            project,
            version,
            epochs,
            batch_size
        ],
        outputs=[progress_output, metrics_output]
    )
    
    infer_btn.click(
        run_inference,
        inputs=[
            inf_model_name,
            inf_model_size,
            checkpoint,
            threshold,
            image_input
        ],
        outputs=image_output
    )

demo.launch(allowed_paths=["/assets/"])