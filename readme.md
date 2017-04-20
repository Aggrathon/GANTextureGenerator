# Texture Generator using Generative Advisary Networks
[ INSERT DESCRIPTION HERE ]


## Examples
![Example Image](/examples/example_01.png)  
Here are 16 generated stone walls after 112 000 training iterations. However the quality didn't seem to improve much past 20 000 iterations, meaning the model still has to be refined.

## Usage

1. Put your real images in the ```input``` folder

2. Train a GAN using:  
```python train.py network_name [iterations]```

3. Generate textures using:  
```python generate.py network_name [num_images]```


## Requirements
 - Python 3
 - Tensorflow
 - Pillow
 - Numpy


## TODO
 - Better readme
 - Generate example
 - Try bigger sizes than 64x64
    - maybe through an upscaling GAN
 - Try making repeatable textures