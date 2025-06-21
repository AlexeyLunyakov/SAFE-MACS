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
* identification of the presence of important types of PPE necessary to protect exposed areas of skin, respiratory tract, eyes.

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title2">Analog Analysis</a></h3>

After conducting a preliminary analysis of analogues, it was revealed that there are no complete analogues of the developed system, however, the following methods of monitoring the wearing of PPE in medical institutions exist:
- **Local visual monitoring**, the disadvantages of which are the human factor, significant time costs;
- **Analysis using a single observer and video surveillance systems**, the obvious disadvantages of which are the high cost of maintenance, as well as low efficiency due to the presence of a person;
- **Pre-trained models for monitoring masks** on the street and in public places, developed mainly during the COVID pandemic.

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title3">Description</a></h3>

<h4 align="start"><a>Solution Architecture</a></h4>

<div align="center"><img width="100%" src="https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/app/assets/safe-macs-arch.png" alt="product architecture"></div>

<h4 align="start"><a>FrontEnd & BackEnd</a></h4>

**Gradio** was chosen as the main application development stack, as it provides the solution with:
- Easy scaling of the system for growing data volumes;
- Cross-platform, easy deployment right out of the box;
- Quick replacement of deep learning models if necessary;
- Variability of data analysis with pandas, numpy and others.

[![Gradio Badge](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=fff&style=flat-square)](https://www.gradio.app)

<h4 align="start"><a>Data Processing</a></h4>

Custom Dataset of 4050 annotated photographs was collected for training computer vision models, augmented with geometric (rotation, scaling) and photometric (lighting adjustment, noise introduction) transformations to improve model robustness.

Priority classes - **Masks, Gloves, and Coveralls** - were identified through visual as well as spatial-frequency analysis of operating room images. However, to improve the robustness of the models, the resulting dataset includes five classes of PPE: **Coveralls, Face Shields, Gloves, Goggles and Masks**.

> You can find this dataset on Roboflow (click on Badge below)

[![Roboflow Badge](https://img.shields.io/badge/Roboflow-6706CE?logo=roboflow&logoColor=fff&style=flat-square)](https://app.roboflow.com/safemacsws/mppe-custom-set/4)


<h4 align="start"><a>Machine Learning</a></h4>

Architecture with multiple models has been implemented, allowing dynamic selection between, for example, YOLOv11-L, RT-DETR-L and RF-DETR based on deployment requirements and one's own choice. This approach balances detection accuracy and processing speed while minimizing hardware dependencies. For ease of data presentation, the ability to predict only those classes that the user selects in the interface has been added.

> *During the design of the system, other models were also trained (RT-DETRv2, RTMDet, RF-DETR), so in the final version, the user was provided with a choice of pipelines in the user interface. Model training, as well as weights, which GitHub allows place, are located in the corresponding directories.* 

[![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat-square)](https://www.python.org/)
[![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat-square)](https://pytorch.org/)

<h4 align="start"><a>Results</a></h4>

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

Testing has shown high metrics for priority classes with little variance in performance between models under different lighting conditions and number of employees. The configurability of the system allows it to be integrated into existing surveillance infrastructure, providing continuous compliance analytics without the need for specialized equipment.

<h4 align="start"><a>Models and authors</a></h4>

Model | Page |
:---:|:---:|
ultralytics/YOLOv11 | [![Ultralytics Badge](https://img.shields.io/badge/Ultralytics-111F68?logo=ultralytics&logoColor=fff&style=flat-square)](https://github.com/ultralytics/ultralytics) |
Baidu/RT-DETR | [![Baidu Badge](https://img.shields.io/badge/Baidu-2932E1?logo=baidu&logoColor=fff&style=flat-square)](https://github.com/lyuwenyu/RT-DETR) |
Roboflow/RF-DETR | [![Roboflow Badge](https://img.shields.io/badge/Roboflow-6706CE?logo=roboflow&logoColor=fff&style=flat-square)](https://github.com/roboflow/rf-detr) |
open-mmlab/RTMDet | [![Github Badge](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=fff&style=flat-square)](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) |
new pipelines | &#x2610; |
coming soon | &#x2610; |

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


## <h3 align="start"><a id="title4">Testing and Deployment</a></h3> 

  <br />

<details>
  <summary> <strong><i>Testing models with an Inference app:</i></strong> </summary>
  
  - In Visual Studio Code (**Windows-PowerShell recommended**) through the terminal, run the following commands sequentially:

    - Clone the repository:
    ```
    git clone https://github.com/AlexeyLunyakov/SAFE-MACS.git
    ```
    - Create your parent directory for docker-machine results output:
    ```
    mkdir -p docker_files
    ```
    - Image build:
    ```
    docker build -t cv-app -f Dockerfile.gpu .
    ```
    - After installing the dependencies (3-5 minutes), you can run Container with GPU:
    ```
    docker run -d --gpus all -p 7860:7860 -v "$(pwd)/docker_files:/app/files" --name cv-container cv-app
    ```
    or with CPU-only:
    ```
    docker build -t cv-app -f Dockerfile.cpu .
    docker run -p 7861:7860 -v "$(pwd)/docker_files:/app/files" --name cv-container cv-app
    ```
</details> 

Additional instructions for installation and use can be found [here](https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/app/readme.md) and [there](https://github.com/AlexeyLunyakov/SAFE-MACS/blob/main/app_train/readme.md)

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>

## <h3 align="start"><a id="title5">Updates</a></h3> 

***ToDo list***
New feature | WIP | Done |
--- |:---:|:---:|
selection of models and approaches (and their use based on the server configuration where the solution is deployed) | &#x2611; | &#x2610; | 
optimization and quantization of the solution for use on mobile devices and low-power configurations (ZEUS, ONNX, TensorRT) | &#x2611; | &#x2610; | 
integration of the database and multi-threaded processing of images, video, streams for full integration into the real conditions | &#x2610; | &#x2610; | 
pipelines integration: RT-DETRv2, RF-DETR, RTMDET and others | &#x2611; | &#x2610; | 

<p align="right">(<a href="#readme-top"><i>Back to top</i></a>)</p>


<a name="readme-top"></a>
