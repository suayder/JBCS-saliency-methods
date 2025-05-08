# JBCS-video-saliency [REPO UNDER CONSTRUCTION]

**Salience prediction methods for video cropping in sidewalk footage**

The condition of urban infrastructure is an important aspect in ensuring the safety and well-being of pedestrians. This is especially important around public health facilities, such as sidewalks surrounding hospitals. Computational tools have already demonstrated their potential in this context, including surface material classification and obstacle detection; however, most solutions require labeled data, which is costly and time-consuming. To address this gap, we propose two strategies for salience prediction in videos that reduce the dependence of manual labeling. The first leverages human visual attention, converting user clicks into attention maps. The second employs the SAM2 model to generate labeled video data more efficiently. The outputs of this process are used to train specialized saliency detectors to identify general cracks, surface defects, and key sections of tactile paving, such as directional changes. Also, we apply these saliency models to video cropping in order to highlight the most relevant areas within each frame. This approach enables content-aware video retargeting, supports object-focused attention, and facilitates sidewalk condition analysis by emphasizing defects and potential hazards. This work extends our previous  study [costa2024videocropping](https://repositorio.usp.br/directbitstream/6a94f30c-3267-4f77-8b84-042cd0eecc96/3225831.pdf) by (1) developing a click-based video annotation tool, (2) developing two saliency detection strategies for sidewalks video cropping, (3) training and evaluating saliency models for sidewalk structure analysis, and (4) applying these models within a video cropping framework. Our experimental results showed that saliency models were able to highlight relevant information in urban environments, achieving an AUC of 0.582 in the best case for human-based attention and 0.914 for tactile-based attention, thereby enhancing assistive technologies for visually impaired individuals.

## VIDEO DEMO

[![](assets/first_page.png)](https://drive.google.com/file/d/1fe7JgEsmDxfiSc1JlJBSFe1ezeB4KgNb/view?usp=drive_link)

## Experiments

To train the models we used the original implementations:

- ViNet - https://github.com/samyak0210/ViNet
- TMFI-Net - https://github.com/wusonghe/TMFI-Net
- STSANet - https://github.com/WZq975/STSANet

### Annotation

For this project an annotation tool was developed, it can be found at https://github.com/suayder/VideoClickCapture

## Scripts

### Human-based click processing

#### Generate attention maps and fixation maps

Attention maps is the density map that is used for training. Fixation maps is only a binary map with the pixel position as value 255, it is used for metrics computation.

- Modify the paths in `config.yaml`
- Run `human-based-click-processing/click2attention_maps.py`, pay attention to the argparse argument. You are able to renderize and a single click as argument or you generate all attention maps if in a folder structure described in the begining of the script file.
- Run `human-based-click-processing/click2fixation.py`, again, pay attention to the arguments in the scripts.

### Cropping

- There are two cropping scripts `cropping_app.py` ans `crop_interpolated.py`
- `cropping_app.py` just maximizes the attention inside the frame given the expected desired final dimmentions
- `crop_interpolated.py` maximizes and interpolate the frames to make the transition between the frames smoothed
- The parameters and paths are changed inside each script
- The aspect ratios for reframing is configured in the constant variables in the beggining of the code
- Both receive the input folder with images and the folder with the attention maps to serve as reference to the cropping
- The outputs are the corresponding cropped images saved in an output folder (configured in the parameters)

### Other scripts

There are some adicional scripts used to help during the development

- `single_image_procedure.ipynb` exemplifies the creation of a single attention map for **tactile paving**
- `data.py` contains some classes to help in data loading, for instance you can easily iterate over videos or generated clicks.


## evaluation

The individual evaluations can be done inside `saliency/` folder, which will have a notebook computing the mean of the saliency.

This code is a version of the original one: https//github.com/herrlich10/saliency


