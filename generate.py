import os
from network import GANetwork


if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        GANetwork('default').generator.save_images()
    elif len(os.sys.argv) < 3:
        GANetwork(os.sys.argv[1]).generator.save_images()
    else:
        GANetwork(os.sys.argv[1]).generator.save_images(int(os.sys.argv[2]))

