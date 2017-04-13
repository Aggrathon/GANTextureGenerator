
import os
from network import GANetwork


CONFIG = {
    'colors': 3,
    'batch_size': 64,
    'generator_base_width': 16,
    'image_size': 64,
    'discriminator_convolutions': 4,
    'generator_convolutions': 5,
    'input_size': 128
}


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        print('Usage:')
        print('  python %s network_name [num_iterations]\t- Trains a network on the images in the input folder'%os.sys.argv[0])
    elif len(os.sys.argv) < 3:
        GANetwork(os.sys.argv[1], **CONFIG).train()
    else:
        GANetwork(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
