#Folders
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
NETWORK_FOLDER = 'network'

#Image
IMAGE_SIZE = 32
COLORED = False

#Network
BATCH_SIZE = 64
DEFAULT_NAME = 'default'


class GeneratorConfig():
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.colors = 3 if COLORED else 1
        self.expand_layers = 2
        self.conv_layers = 3
        self.conv_size = 32
        self.input_size = 128
        self.dropout = 0.4
        self.batch_size = BATCH_SIZE

class DiscriminatorConfig():
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.colors = 3 if COLORED else 1
        self.conv_layers = 3
        self.conv_size = 32
        self.class_layers = 2
        self.dropout = 0.4
        self.batch_size = BATCH_SIZE

