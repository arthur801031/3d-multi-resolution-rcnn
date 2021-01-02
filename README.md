
# 3D Instance Segmentation Framework for Cerebral Microbleeds using 3D Multi-Resolution R-CNN

Official PyTorch implementaiton of the paper "3D Instance Segmentation Framework for Cerebral Microbleeds using 3D Multi-Resolution R-CNN" by I-Chun Arthur Liu, Chien-Yao Wang, Jiun-Wei Chen, Wei-Chi Li, Feng-Chi Chang, Yi-Chung Lee, Yi-Chu Liao, Chih-Ping Chung, Hong-Yuan Mark Liao, Li-Fen Chen. Paper is currently under review.

Keywords: 3D instance segmentation, 3D object detection, cerebral microbleeds, convolutional neural networks (CNNs), susceptibility weighted imaging (SWI), 3D Mask R-CNN, magnetic resonance imaging (MRI), medical imaging, pytorch.


## Usage Instructions

### Requirements
- Linux (tested on Ubuntu 16.04 and Ubuntu 18.04)
- Conda or Miniconda
- Python 3.4+
- PyTorch 1.0
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv)

### Tested on CUDAs
- CUDA 11.0 with Nvidia Driver 450.80.02
- CUDA 10.0 with Nvidia Driver 410.78
- CUDA 9.0 with Nvidia Driver 384.130

### Installation

1. Cloen the repository.

```shell
git clone https://github.com/arthur801031/3d-multi-resolution-rcnn.git
```

2. Install Conda or Miniconda.
[Miniconda installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)


3. Create a conda environment from `conda.yml`.

```shell
cd 3d-multi-resolution-rcnn/
conda env create --file conda.yml
```

4. Install pip packages.

```shell
pip install -r requirements.txt
```

5. Compile cuda extensions.

```shell
./compile.sh
```

6. Create a "data" directory and move your dataset to this directory.

```shell
mkdir data
```

## Training Commands
```bash
# single GPU training with validation during training
clear && python setup.py install && CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/3d-multi-resolution-rcnn.py 1 --validate

# multi-GPU training with validation during training
clear && python setup.py install && CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/3d-multi-resolution-rcnn.py 2 --validate

# resume training from checkpoint
clear && python setup.py install && CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/3d-multi-resolution-rcnn.py 2 --validate --resume_from work_dirs/checkpoints/3d-multi-resolution-rcnn/latest.pth
```

## Testing Commands
```bash
# perform evaluation on bounding boxes only
clear && python setup.py install && CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/3d-multi-resolution-rcnn.py work_dirs/checkpoints/3d-multi-resolution-rcnn/latest.pth --gpus 1 --out results.pkl --eval bbox

# perform evaluation on bounding boxes and segmentations
clear && python setup.py install && CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/3d-multi-resolution-rcnn.py work_dirs/checkpoints/3d-multi-resolution-rcnn/latest.pth --gpus 1 --out results.pkl --eval bbox segm
```

## Test Image(s) 
Refer to `test_images.py` for details.

## COCO Annotation Format
```json
{
    "info": {
        "description": "Dataset",
        "url": "https://",
        "version": "0.0.1",
        "year": 2019,
        "contributor": "arthur",
        "date_created": "2020-10-29 17:12:12.838644"
    },
    "licenses": [
        {
            "id": 1,
            "name": "E.g. Attribution-NonCommercial-ShareAlike License",
            "url": "http://"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "microbleed",
            "supercategory": "COCO"
        }
    ],
    "images": [
        {
            "id": 1,
            "file_name": "A002-26902603_instance_v1.npy",
            "width": 512,
            "height": 512,
            "date_captured": "2020-10-29 15:31:32.060574",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        },
        {
            "id": 2,
            "file_name": "A003-1_instance_v1.npy",
            "width": 512,
            "height": 512,
            "date_captured": "2020-10-29 15:31:32.060574",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "area": 196,
            "bbox": [
                300,
                388,
                7,
                7,
                65,
                4
            ],
            "segmentation": "data/Stroke_v4/COCO-full-vol/train/annotations_full/A002-26902603_instance_v1_1.npy",
            "segmentation_label": 1,
            "width": 512,
            "height": 512
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "iscrowd": 0,
            "area": 1680,
            "bbox": [
                334,
                360,
                15,
                14,
                33,
                8
            ],
            "segmentation": "data/Stroke_v4/COCO-full-vol/train/annotations_full/A003-1_instance_v1_1.npy",
            "segmentation_label": 1,
            "width": 512,
            "height": 512
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "iscrowd": 0,
            "area": 486,
            "bbox": [
                380,
                244,
                9,
                9,
                51,
                6
            ],
            "segmentation": "data/Stroke_v4/COCO-full-vol/train/annotations_full/A003-1_instance_v1_10.npy",
            "segmentation_label": 10,
            "width": 512,
            "height": 512
        },
        {
            "id": 4,
            "image_id": 2,
            "category_id": 1,
            "iscrowd": 0,
            "area": 256,
            "bbox": [
                340,
                300,
                8,
                8,
                61,
                4
            ],
            "segmentation": "data/Stroke_v4/COCO-full-vol/train/annotations_full/A003-1_instance_v1_11.npy",
            "segmentation_label": 11,
            "width": 512,
            "height": 512
        },
        {
            "id": 5,
            "image_id": 2,
            "category_id": 1,
            "iscrowd": 0,
            "area": 550,
            "bbox": [
                367,
                196,
                10,
                11,
                65,
                5
            ],
            "segmentation": "data/Stroke_v4/COCO-full-vol/train/annotations_full/A003-1_instance_v1_12.npy",
            "segmentation_label": 12,
            "width": 512,
            "height": 512
        }
    ]
}
```

This codebase is based on [OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection).