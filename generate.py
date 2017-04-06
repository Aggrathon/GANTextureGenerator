
import os

import tensorflow as tf

from network import GANetwork
from train import CONFIG


def generate(name, amount=1):
    CONFIG['batch_size'] = amount
    gan = GANetwork(name, **CONFIG)
    session, _, iter = gan.get_session(create=False)
    if iter == 0:
        print("No already trained network found (%s)"%name)
        return
    print("Generating %d images using the %s network"%(amount, name))
    gan.generate(session, gan.name, amount)

def generate_grid(name):
    gan = GANetwork(name, **CONFIG)
    session, _, iter = gan.get_session(create=False)
    if iter == 0:
        print("No already trained network found (%s)"%name)
        return
    print("Generating a image grid using the %s network"%name)
    gan.generate_grid(session, gan.name)

if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        generate('default')
    elif len(os.sys.argv) < 3:
        generate(os.sys.argv[1])
    else:
        if os.sys.argv[2] == 'grid':
            generate_grid(os.sys.argv[1])
        else:
            generate(os.sys.argv[1], int(os.sys.argv[2]))
