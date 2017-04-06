
import os

import tensorflow as tf

from network import GANetwork
from train import CONFIG


def create_session(name):
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    try:
        saver.restore(session, os.path.join('network', name))
    except Exception as e:
        print(e)
        print("No already trained network found (%s)"%os.path.join('network', name))
        exit()
    return session

def generate(name, amount=1):
    CONFIG['batch_size'] = amount
    gan = GANetwork(name, **CONFIG)
    session = create_session(name)
    print("Generating %d images using the %s network"%(amount, name))
    gan.generate(session, gan.name, amount)

def generate_grid(name):
    gan = GANetwork(name, **CONFIG)
    session = create_session(name)
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
