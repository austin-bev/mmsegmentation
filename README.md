# Honours research
## Austin Bevacqua (20162896) 
Huge thanks to Bradley for starting work on this

## Setup
For setup, I used WSL2 and installed miniconda. From there, I created a new conda environment with Python 3.8, Pytorch 2.0 and CUDA 11.7.
From there I basically followed this: https://mmsegmentation.readthedocs.io/en/latest/get_started.html to install all of the dependencies

I did initally have a problem where mmseg wouldn't work due to `Could not load library libcudnn_cnn_infer.so.8`, but installing nvidia-cudnn seemed to fix it.

## Datasets
The first step to training and testing the models is to get the right dataset. This model trains on the CamVid11 dataset.
First, download the CamVid32 dataset https://www.kaggle.com/datasets/carlolepelaars/camvid then follow dataset_utils.ipynb to convert this into the CamVid11 dataset.

Next step is to grab `extra/camvid.py` and insert it into your local mmsegmentation install (`mmseg/datasets`).
After that, edit the `__init__.py` inside of `mmseg/datasets` to include: `from .camvid import Camvid11Dataset` and edit `__all__` to include `Camvid11Dataset` in the list.
