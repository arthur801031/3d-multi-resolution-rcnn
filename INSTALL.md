## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 3.4+
- PyTorch 1.0
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv)

### Tested on CUDAs

- CUDA 11.0 with Nvidia Driver 450.80.02
- CUDA 10.0 with Nvidia Driver 410.78
- CUDA 9.0 with Nvidia Driver 384.130

### Install mmdetection

a. Clone the mmdetection repository.

```shell
git clone https://github.com/arthur801031/mmdetection-arthur.git
```

b. Install Conda and create an environment from `pytorch10.yml`

c. Compile cuda extensions.

```shell
cd mmdetection-arthur
pip install -r requirements.txt # Make sure `Pytorch (torch==1.0.1.post2)` and `torchvision==0.2.2` are installed in pip
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

d. Install mmdetection (other dependencies will be installed automatically).

```shell
# outdated...
# python(3) setup.py install  # add --user if you want to install it locally
# or "pip install -e ."
```

Refer to `./working_pip.md` for detail pip list (workign as of March 4th, 2020).

Note: You need to run the last step each time you pull updates from github.
The git commit id will be written to the version number and also saved in trained models.

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```

### Scripts
Just for reference, [Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Notice
You can run `python(3) setup.py develop` or `pip install -e .` to install mmdetection if you want to make modifications to it frequently.

If there are more than one mmdetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
