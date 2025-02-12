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

<center>

**Contents** |
:---:|
[Abstract](#title1) |
[Analog Analysis](#title2) |
[Description](#title3) |
[Testing and Deployment](#title4) |
[Updates](#title5) |

</center>

## <h3 align="start"><a id="title1">Abstract</a></h3> 
**In response to the need for enhanced safety measures in healthcare, a solution was developed to automate the monitoring of personal protective equipment (PPE) use by healthcare personnel.**

The relevance of this topic is associated with the disappointing statistics of the increase in morbidity during the COVID-19 pandemic, when the lack of a modern solution in the field of human health safety led to monstrous consequences. The rapid spread of infectious diseases, especially those transmitted through contact, due to a lack of adequate control, highlights the need to protect both patients and healthcare workers through innovative non-invasive methods.

**The developed solution can be used to:**
* automate the process of monitoring compliance with safety regulations in healthcare facilities;
* optimize business processes in the field of medical logistics and personnel management.

**The task involves working with real images of medical workers in a working environment, so the solution includes several key tasks for training models:**
* accurate determination of the presence of PPE on a person and warning about its absence;
* identification of the presence of important types of PPE necessary to protect exposed areas of skin, respiratory tract, eyes, including: Coveralls, Face Shields, Gloves, Goggles and Masks.


<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title2">Analog Analysis</a></h3>

After conducting a preliminary analysis of analogues, it was revealed that there are no complete analogues of the developed system, however, the following methods of monitoring the wearing of PPE in medical institutions exist:
- **Local visual monitoring**, the disadvantages of which are the human factor, significant time costs;
- **Analysis using a single observer and video surveillance systems**, the obvious disadvantages of which are the high cost of maintenance, as well as low efficiency due to the presence of a person;
- **Pre-trained models for monitoring masks** on the street and in public places, developed mainly during the COVID pandemic.

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title3">Description</a></h3>

<h4 align="start"><a>FrontEnd & BackEnd</a></h4>

**Gradio** was chosen as the main application development stack, as it provides the solution with:
- Easy scaling of the system for growing data volumes;
- Cross-platform, easy deployment right out of the box;
- Quick replacement of deep learning models if necessary;
- Variability of data analysis with pandas, numpy and others;

[![Gradio Badge](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=fff&style=flat-square)](https://www.gradio.app)

<h4 align="start"><a>Machine Learning</a></h4>

The future architecture of the system was chosen to be an ensemble of **YOLOv11** model as a detector and the introduction of an additional model (EfficientNet or similar) as a classifier of objects of interest. 

> *During the design of the system, other models were also trained (RT-DETR, CLIP), so in the final version, the user was provided with a choice of pipelines in the user interface. Model training, as well as weights, which GitHub allows place, are located in the corresponding directories.* 

[![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square)](https://www.python.org/)
[![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat-square)](https://pytorch.org/)

**Object Detection & Classification**:

Model | Page |
:---:|:---:|
ultralytics/YOLOv11-M | [![Ultralytics Badge](https://img.shields.io/badge/Ultralytics-111F68?logo=ultralytics&logoColor=fff&style=flat-square)](https://github.com/ultralytics/ultralytics) |
Baidu/RT-DETR-L | [![Baidu Badge](https://img.shields.io/badge/Baidu-2932E1?logo=baidu&logoColor=fff&style=flat-square)](https://github.com/lyuwenyu/RT-DETR) |
new pipelines | &#x2610; |
coming soon | &#x2610; |

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


## <h3 align="start"><a id="title4">Testing and Deployment</a></h3> 

  <br />

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
    python ./app/app.py
    ```
    or with the ability to automatically restart if errors occur:
    ```
    cd ./app/
    gradio app.py
    ```

</details> 

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title5">Updates</a></h3> 

***TODO list***
TODO | WIP | DONE |
--- |:---:|:---:|
selection of models and approaches (and their use based on the server configuration where the solution is deployed) | &#x2611; | &#x2610; | 
optimization and quantization of the solution for use on mobile devices and low-power configurations (ZEUS, ONNX, TensorRT) | &#x2611; | &#x2610; | 
integration of the database and multi-threaded processing of images, video, streams for full integration into the real conditions | &#x2610; | &#x2610; | 
models integration: CLIP, CLIP+EfficientNet, YOLO+CLIP and other ensembles | &#x2611; | &#x2610; | 

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


<a name="readme-top"></a>
