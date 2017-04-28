
import os
from network import GANetwork
from autogan import AutoGanGenerator


CONFIG = {
    'colors': 3,
    'batch_size': 192,
    'generator_base_width': 32,
    'image_size': 64,
    'discriminator_convolutions': 5,
    'generator_convolutions': 5,
}

def get_network(name, **config):
    #return GANetwork(name, **config)
    return AutoGanGenerator(name=name, **config)


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        print('Usage:')
        print('  python %s network_name [num_iterations]\t- Trains a network on the images in the input folder'%os.sys.argv[0])
    elif len(os.sys.argv) < 3:
        get_network(os.sys.argv[1], **CONFIG).train()
    else:
        get_network(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
