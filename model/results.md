## Final metrics of the models:

Model | params | epochs | F1 | map50 | map50-95 | data |
--- |:---:|:---:|:---:|:---:|:---:|:---:|
yolo11-l | 25.3 M | 20 | 0.71 at 0.305 | 0.74 | 0.447 | 1017
yolo11-l | 25.3 M | 75 | 0.81 at 0.308 | 0.836 | 0.551 | 4055 wp
yolo11-x | 56.9 M | 100 | 0.81 at 0.385 | 0.828 | 0.556 | 44055 np
yolo11-x | 56.9 M | 75 | 0.81 at 0.492 | 0.832 | 0.551 | 4055 wp
rt-detr-v1-l | 32 М | 10 | 0.75 at 0.409 | 0.77 | 0.467 | 1017
rt-detr-v1-l | 32 М | 50 | 0.84 at 0.403 | 0.863 | 0.567 | 4055 wp
rt-detr-v2-l | 42 М | 50 | 0.82 at 0.550 | 0.823 | 0.606 | 4055 wp

## Mean Average Precision for all classes:

Model-mAP50 | Coverall | Face Shield | Gloves | Goggles | Mask | data |
--- |:---:|:---:|:---:|:---:|:---:|:---:|
yolo11-l | 0.511 | 0.48 | 0.41 | 0.16 | 0.728 | (1017 val on new data)
yolo11-l | 0.96 | 0.686 | 0.808 | 0.782 | 0.945 | (4055 wp)
yolo11-x | 0.959 | 0.657 | 0.815 | 0.764 | 0.946 | (4055 np)
yolo11-x | 0.896 | 0.485 | 0.708 | 0.618 | 0.913 | (4055 np val on new data)
yolo11-x | 0.957 | 0.659 | 0.803 | 0.792 | 0.949 | (4055 wp)
rt-detr-v1-l | 0.511 | 0.485 | 0.522 | 0.209 | 0.742 | (1017 val on new data)
rt-detr-v1-l | 0.954 | 0.72 | 0.859 | 0.826 | 0.957 | (4055 wp)
rt-detr-v2-l |0.954 | 0.716 | 0.838 | 0.699 | 0.938 | (4055 wp)


## Designation of datasets:
* 1017 - cppe5 dataset
* 4055 - mppe-custom-set (own)
    * 4055 wp - with preprocess
    * 4055 np - no preprocess (tried for the sake of experiment)