# Texture Generator using Generative Advisary Networks
[ INSERT DESCRIPTION HERE ]

## Usage
Edit the config.py file for quick edits of the GAN layout and hyperparameters

Train a GAN using:  
```python train.py network_name [iterations]```

Generate a texture using:  
```python generate.py network_name [num_images]```

## Requirements
 - Python 3
 - Tensorflow
 - Pillow
 - Numpy

## TODO
 - Better readme
 - instead of conv2d + pool try conv2d + lrelu in the discriminator
 - generate example
 - try colors
 - try bigger sizes than 32x32
    - maybe through an upscaling GAN
 - Try making repeatable textures