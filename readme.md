# Texture Generator using Generative Advisary Networks
This is a neural network for generating images using the GAN (Generative Advisary Network) arcitechture. 
It is initially setup for creating more texture-like images where some cropping, rotation and filtering can be used to increase the amount training data (input images).


## Examples
### Sonewalls
![Example Image](/examples/example_01.png)
![Example Image](/examples/example_02.png)

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


## Inspirations
 - [Generative Adversarial Nets in TensorFlow](http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
 - [DCGAN](https://arxiv.org/abs/1511.06434)


## TODO
 - Fix the stalled learning
    - only small improvement beyond 20 000 iterations (mostly shuffling)
    - Better optimization?
        - Better cost function?
        - Better network configuration?
 - Try bigger sizes than 64x64
    - maybe through an upscaling GAN
 - Try making repeatable textures