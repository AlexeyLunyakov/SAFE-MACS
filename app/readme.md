<a name="readme-top"></a>  
<div align="center"><img width="100%" src="https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/app/assets/safe-macs-h.png" alt="product banner"></div>
<div align="center">
  <p align="center">
    <h1 align="center">Safety Automated Medical Control System (SAFE-MACS)</h1>
  </p>

  <p align="center">
    <p><strong>Application for digital automatic control of the use of PPE for medical personnel using computer vision.</strong></p>
    <br /><br />
  </p>
</div>

<h4 align="start"><a>Machine Learning</a></h4>

Architecture with multiple models has been implemented, allowing dynamic selection between, YOLOv11-L, RT-DETR-L, RF-DETR-BASE (or their ONNX versions) based on deployment requirements.

<h4 align="start"><a>Metrics</a></h4>

The full testing table can be found [here](https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/model/results.md)

Model | prms | epochs | F1-w | mAP50 | mAP50-95 |
--- |:---:|:---:|:---:|:---:|:---:|
yolo11-l | 25.3 M | 75 | 0.81 at 0.308 | 0.836 | 0.551 |
rt-detr-v1-l | 32 М | 50 | 0.84 at 0.403 | 0.863 | 0.567 |
rf-detr-base | 29 M | 25 | 0.892 at 0.550 | 0.901 | 0.676 |


Model-mAP50 | Coverall | Face_Shield | Gloves | Goggles | Mask |
--- |:---:|:---:|:---:|:---:|:---:|
yolo11-l | 0.96 | 0.686 | 0.808 | 0.782 | 0.945 |
rt-detr-v1-l | 0.954 | 0.72 | 0.859 | 0.826 |0.957 |
rf-detr-base | 0.967 | 0.888 | 0.896 | 0.801 | 0.955 |

<h4 align="start"><a></a>Module Testing</h4>

<details>
  <summary> <strong><i>Testing models with an app on Gradio:</i></strong> </summary>
  
  - In Visual Studio Code (**Windows-PowerShell recommended**) through the terminal, run the following commands sequentially:

    - Clone the repository:
    ```
    git clone https://github.com/AlexeyLunyakov/SAFE-MACS.git
    ```
    - Creating and activating a virtual environment:
    ```
    cd ./SAFE-MACS
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Installing dependencies (CUDA 12.4 required):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip3 install -r requirements.txt
    ```
    - After installing the dependencies (3-5 minutes), you can run Gradio:
    ```
    cd ./app/
    python app.py
    ```
    or with the ability to automatically restart if errors occur:
    ```
    cd ./app/
    gradio app.py
    ```
</details> 