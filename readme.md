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
 - Generate example
 - Try colors
 - Try bigger sizes than 64x64
    - maybe through an upscaling GAN
 - Try batch normalization
 - Try making repeatable textures