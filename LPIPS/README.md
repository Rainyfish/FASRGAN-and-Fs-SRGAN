
## Perceptual Similarity Metric and Dataset [[Project Page]](http://richzhang.github.io/PerceptualSimilarity/)

This repository contains the **(1) Learned Perceptual Image Patch Similarity (LPIPS) metric** and **(2) Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset** proposed in the following paper:

**The Unreasonable Effectiveness of Deep Networks as a Perceptual Metric**  
[Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Oliver Wang](http://www.oliverwang.info/).  
In [ArXiv, 2018](https://arxiv.org/abs/1801.03924).  

<img src='https://richzhang.github.io/PerceptualSimilarity/index_files/fig1_v2.jpg' width=1200>

## (0) Dependencies

This repository uses Python 2 or 3, with the following libraries: [PyTorch](www.pytorch.org), numpy, scipy, skimage.

## (1) Learned Perceptual Image Patch Similarity (LPIPS) metric

### About the metric

We found that deep network activations work surprisingly well as a perceptual similarity metric. This was true across network architectures (SqueezeNet [2.8 MB], AlexNet [9.1 MB], and VGG [58.9 MB] provided similar scores) and supervisory signals (unsupervised, self-supervised, and supervised all perform strongly). We slightly improved scores by linearly "calibrating" networks - adding a linear layer on top of off-the-shelf classification networks. We provide 3 variants, using linear layers on top of the SqueezeNet, AlexNet (default), and VGG networks. Using this code, you can simply call `model.forward(im0,im1)` to evaluate the distance between two image patches.

### Using the metric

File [`test_network.py`](test_network.py) contains example usage. Running `test_network.py` will take the distance between example reference image [`ex_ref.png`](./imgs/ex_ref.png) to distorted images [`ex_p0.png`](./imgs/ex_p0.png) and [`ex_p1.png`](./imgs/ex_p1.png). Before running it - which do you think *should* be closer?

Load a model with the following commands.

```python
from models import dist_model as dm
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True)
```

Variable `net` can be `squeeze`, `alex`, `vgg`. Network `alex` is fastest and performs the best (not specifying the `net` will default to `alex`). Set `model=net` for an uncalibrated off-the-shelf network (taking cos distance).

To call the model, run ```model.forward(im0,im1)```, where ```im0, im1``` are PyTorch tensors with shape ```python Nx3xHxW``` (```N``` patches of size ```HxW```, RGB images scaled in `[-1,+1]`).

## (2) Berkeley Adobe Perceptual Patch Similarity (BAPPS) dataset

### Downloading the dataset

Run `bash ./scripts/get_dataset.sh` to download and unzip the dataset. Dataset will appear in directory `./dataset`. Dataset takes [6.6 GB] total.
- 2AFC train [5.3 GB]
- 2AFC val [1.1 GB]
- JND val [0.2 GB]  
Alternatively, run `bash ./scripts/get_dataset_valonly.sh` to only download the validation set (no training set).

### Evaluating a perceptual similarity metric on a dataset

Script `test_dataset_model.py` evaluates a perceptual model on a subset of the dataset.

**Dataset flags**
- `dataset_mode`: `2afc` or `jnd`, which type of perceptual judgment to evaluate
- `datasets`: list the datasets to evaluate
    - if `dataset_mode` was `2afc`, choices are [`train/traditional`, `train/cnn`, `val/traditional`, `val/cnn`, `val/superres`, `val/deblur`, `val/color`, `val/frameinterp`]
    - if `dataset_mode` was `jnd`, choices are [`val/traditional`, `val/cnn`]
    
**Perceptual pimilarity model flags**
- `model`: perceptual similarity model to use
    - `net-lin` for our LPIPS learned similarity model (linear network on top of internal activations of pretrained network)
    - `net` for a classification network (uncalibrated with all layers averaged)
    - `l2` for Euclidean distance
    - `ssim` for Structured Similarity Image Metric
- `net`: choices are [`squeeze`,`alex`,`vgg`] for the `net-lin` and `net` models (ignored for `l2` and `ssim` models)
- `colorspace`: choices are [`Lab`,`RGB`], used for the `l2` and `ssim` models (ignored for `net-lin` and `net` models)

**Misc flags**
- `batch_size`: evaluation batch size (will default to 1 )
- `--use_gpu`: turn on this flag for GPU usage

An example usage is as follows: `python ./test_dataset_model.py --dataset_mode 2afc --datasets val/traditional val/cnn --model net-lin --net alex --use_gpu --batch_size 50`. This would evaluate our model on the "traditional" and "cnn" validation datasets.

### About the dataset

The dataset contains two types of perceptual judgements: **Two Alternative Forced Choice (2AFC)** and **Just Noticeable Differences (JND)**.

**(1) Two Alternative Forced Choice (2AFC)** - Data is contained in the `2afc` subdirectory. Evaluators were given a reference patch, along with two distorted patches, and were asked to select which of the distorted patches was "closer" to the reference patch.

Training sets contain 2 human judgments/triplet.
- `train/traditional` [56.6k triplets]
- `train/cnn` [38.1k triplets]
- `train/mix` [56.6k triplets]

Validation sets contain 5 judgments/triplet.
- `val/traditional` [4.7k triplets]
- `val/cnn` [4.7k triplets]
- `val/superres` [10.9k triplets]
- `val/deblur` [9.4k triplets]
- `val/color` [4.7k triplets]
- `val/frameinterp` [1.9k triplets]

Each 2AFC subdirectory contains the following folders:
- `ref` contains the original reference patches
- `p0,p1` contain the two distorted patches
- `judge` contains what the human evaluators chose - 0 if all humans preferred p0, 1 if all humans preferred p1

**(2) Just Noticeable Differences (JND)** - Data is contained in the `jnd` subdirectory. Evaluators were presented with two patches - a reference patch and a distorted patch - for a limited time, and were asked if they thought the patches were the same (identically) or difference. 

Each set contains 3 human evaluations/example.
- `val/traditional` [4.8k patch pairs]
- `val/cnn` [4.8k patch pairs]

Each JND subdirectory contains the following folders:
- `p0,p1` contain the two patches
- `same` contains fraction of human evaluators who thought the patches were the same (0 if all humans thought patches were different, 1 if all humans thought patches were the same)

## Acknowledgements

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
