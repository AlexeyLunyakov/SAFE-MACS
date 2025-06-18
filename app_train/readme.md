<a name="readme-top"></a>  
<div align="center"><img width="100%" src="https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/app_train/assets/cvm-ts-h.png" alt="product banner"></div>
<div align="center">
  <p align="center">
    <h1 align="center">Computer Vision Model Training Studio (CVM-TS)</h1>
  </p>

  <p align="center">
    <p><strong>Application for training models for computer vision tasks using Roboflow datasets.</strong></p>
    <br /><br />
  </p>
</div>

<h4 align="start"><a>Machine Learning</a></h4>

Currently, Ultralytics, Baidu (PekingU), Roboflow and OpenMMLab models are trained using unification of the model training code. The solution can be deployed both locally and remotely, the backend part of the solution is integrated with FastAPI

<h4 align="start"><a>Train algorithms</a></h4>

Model training in Jupyter Notebooks can be found [here](https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/model/)

The full testing table can be found [here](https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/model/results.md)


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
    cd ./app_train/
    python train_app.py
    ```
    or with the ability to automatically restart if errors occur:
    ```
    cd ./app_train/
    gradio train_app.py
    ```
</details> 