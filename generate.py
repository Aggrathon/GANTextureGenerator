import os
import tensorflow as tf
from network import GANetwork
from config import NETWORK_FOLDER


def create_session(name):
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    try:
        saver.restore(session, os.path.join(NETWORK_FOLDER, name))
    except Exception as e:
        print(e)
        print("No already trained network found (%s)"%os.path.join(NETWORK_FOLDER, name))
        exit()
    return session

def generate(name, amount=1):
    gan = GANetwork(name)
    session = create_session(name)
    print("Generating %d images using the %s network"%(amount, name))
    for i in range(amount):
        gan.generator.generate(session, gan.name+'-'+str(i))

if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        generate('default')
    elif len(os.sys.argv) < 3:
        generate(os.sys.argv[1])
    else:
        generate(os.sys.argv[1], int(os.sys.argv[2]))

