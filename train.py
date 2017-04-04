import os
from network import GANetwork


CONFIG = {
    'colors': 1,
    'batch_size': 32
}


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        GANetwork('default', **CONFIG).train()
    elif len(os.sys.argv) < 3:
        GANetwork(os.sys.argv[1], **CONFIG).train()
    else:
        GANetwork(os.sys.argv[1], **CONFIG).train(int(os.sys.argv[2]))
