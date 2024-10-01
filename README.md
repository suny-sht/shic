# SHIC: Shape-Image Correspondences with no Keypoint Supervision

ECCV 2024

## Authors

- **Aleksandar (Suny) Shtedritski**
- **Christian Rupprecht**
- **Andrea Vedaldi**

Contact: {suny, chrisr, vedaldi}@robots.ox.ac.uk

## Website

For more detailed information about SHIC, visit our [project website](https://www.robots.ox.ac.uk/~vgg/research/shic/).

## Demo

A demo of SHIC is available on Hugging Face. You can try it out [here](https://huggingface.co/spaces/suny-sht/shic).

## Code 

### Installation 
```
conda create -n shicenv python=3.9
conda activate shicenv
pip install torch==1.12.0 torchvision==0.13.0 einops spharapy==1.1.2 trimesh==3.23.5 matplotlib numpy scipy Pillow jupyter
```
### Weights

Download pretrained the weights here [here](https://drive.google.com/file/d/1xj8NyPcJIZsZjU-ewvVmEm1rKwxgBa-x/view?usp=sharing).

Unzip and put in this folder. Should look like


    .
    ├── ...
    ├── models                 
    │   ├── shapes          # obj shape files
    │   └── weights         # pth weights for shape features and image models
    ├── README.md
    └── ...


### Demo notebook
Run the `shic.ipynb` notekbook for a more custom demo of the model. The notebook runs very quickly on a CPU, which is used by default

### Training code
Coming soon!

## BibTeX

```
@inproceedings{shtedritski2024SHIC,
      title={SHIC: Shape-Image Correspondences with no Keypoint Supervision}, 
      author={Shtedritski, Aleksandar and Rupprecht, Christian and Vedaldi, Andrea},
      year={2024},
      booktitle={ECCV},
}
```
