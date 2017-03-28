import os
from network import GANetwork
from image import ImageManager


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        GANetwork('default').train(1000)
    elif len(os.sys.argv) < 3:
        GANetwork(os.sys.argv[1]).train()
    else:
        GANetwork(os.sys.argv[1]).train(int(os.sys.argv[2]))
