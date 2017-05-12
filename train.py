
import os
from generator_gan import GANetwork
from image import ImageVariations


CONFIG = {
    'colors': 3,
    'batch_size': 192,
    'generator_base_width': 32,
    'image_size': 64,
    'input_size': 128,
    'discriminator_convolutions': 4,
    'generator_convolutions': 4,
    'learning_rate': 0.0003,
    'learning_momentum': 0.8,
    'learning_momentum2': 0.95
}

IMAGE_CONFIG = {
    'pool_renew': 4,
    'rotation_range': (-30, 30),
    'brightness_range': (0.7, 1.2),
    'saturation_range': (0.7, 1.2),
    'contrast_range': (0.8, 1.2),
    'size_range': (1.0, 0.95)
}

def get_network(name, **config):
    return GANetwork(name, image_manager=ImageVariations(**IMAGE_CONFIG), **config)


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        print('Usage:')
        print('  python %s network_name [num_iterations]\t- Trains a network on the images in the input folder'%os.sys.argv[0])
    elif len(os.sys.argv) < 3:
        get_network(os.sys.argv[1], **CONFIG).train()
    else:
        get_network(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
