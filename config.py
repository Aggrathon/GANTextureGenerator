#Folders
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
NETWORK_FOLDER = 'network'

#Image
IMAGE_SIZE = 64
COLORED = False

#Network
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class GeneratorConfig():
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.colors = 3 if COLORED else 1
        self.conv_layers = 5
        self.conv_size = 32
        self.input_size = 128
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE

class DiscriminatorConfig():
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.colors = 3 if COLORED else 1
        self.conv_layers = 4
        self.conv_size = 32
        self.class_layers = 1
        self.dropout = 0.4
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE

