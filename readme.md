# Texture Generator using Generative Advisary Networks
This is a neural network for generating images using the GAN (Generative Advisary Network) arcitechture. 
It is initially setup for creating more texture-like images where some cropping, rotation and filtering can be used to increase the amount training data (input images).


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


## Inspirations
 - [Generative Adversarial Nets in TensorFlow](http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
 - [DCGAN](https://arxiv.org/abs/1511.06434)


## TODO
 - Fix the stalled learning
    - no improvement beyond 20 000 iterations
    - Better optimization?
        - Better cost function?
        - Better network configuration?
            - Add autoencoder to setup better generator
                - Run each log5/2 iterations
 - Try bigger sizes than 64x64
    - maybe through an upscaling GAN
 - Try making repeatable textures