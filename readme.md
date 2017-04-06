# Texture Generator using Generative Advisary Networks
[ INSERT DESCRIPTION HERE ]


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
 - Try colors
 - Try bigger sizes than 64x64
    - maybe through an upscaling GAN
 - Try making repeatable textures
 - Denoise? https://www.tensorflow.org/api_docs/python/tf/image/total_variation