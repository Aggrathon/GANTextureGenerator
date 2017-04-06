
import os
from network import GANetwork


CONFIG = {
    'colors': 3,
    'batch_size': 32,
    'generator_base_width': 32,
    'image_size': 128,
    'discriminator_convolutions': 5,
    'generator_convolutions': 6,
    'input_size': 256,
    'learning_rate': 1.0
}


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        GANetwork('default', **CONFIG).train()
    elif len(os.sys.argv) < 3:
        GANetwork(os.sys.argv[1], **CONFIG).train()
    else:
        GANetwork(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
