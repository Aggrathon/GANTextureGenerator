import sys, os, string, random, math
from PIL import Image, ImageEnhance
import numpy as np
from config import INPUT_FOLDER, OUTPUT_FOLDER, IMAGE_SIZE, BATCH_SIZE


class ImageManager():

    def __init__(self, image_size=IMAGE_SIZE, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, keep_in_memory=True):
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        #Attributes
        self.keep_in_memory = keep_in_memory
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_size = image_size
        #Config
        self.max_rotation = 30
        self.brightness_range = (0.7, 1.1)
        self.saturation_range = (0.7, 1.)
        self.contrast_range = (0.8, 1.2)
        #File reading
        if keep_in_memory:
            self.images = [self.read(f) for f in os.listdir(input_folder)]
            for i in self.images:
                i.crop((1, 1, 2, 2))


    def get_batch(self, batch_size=BATCH_SIZE):
        if self.keep_in_memory:
            return np.array([self.get_array(random.choice(self.images)) for _ in range(batch_size)])
        else:
            files = os.listdir(self.input_folder)
            return np.array([self.get_array_from_file(random.choice(files)) for _ in range(batch_size)])


    def get_array(self, image):
        arr = np.asarray(self.get_variation(image))
        arr.shape = self.image_size*self.image_size*3
        return arr

    def get_array_from_file(self, file):
        return self.get_array(self.read(file))

    def get_variation(self, image):
        #Crop
        min_dim = min(image.size)
        scale = math.sqrt(2)*self.image_size*1.1
        if scale*3 < min_dim:
            scale = random.uniform(0.3, 0.7)
        elif scale*2 < min_dim:
            scale = random.uniform(min_dim, min_dim - scale)/min_dim
        else:
            scale = random.uniform(scale, min_dim)/min_dim
        size = int(random.random()*min_dim*(1-scale)+min_dim*scale)
        pos = (random.randrange(0, image.size[0]-size), random.randrange(0, image.size[1]-size))
        image = image.crop((pos[0], pos[1], pos[0]+size, pos[1]+size))
        #Rotate
        size = image.size[0]/math.sqrt(2)
        offset = (image.size[0]-size)/2
        rotation = random.randint(-self.max_rotation, self.max_rotation)
        image = image.rotate(rotation).crop((offset, offset, offset+size, offset+size))
        #Transpose
        if random.random() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
        #Variation
        brightness = random.uniform(*self.brightness_range)
        if np.abs(brightness-1.0) > 0.05:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        saturation = random.uniform(*self.saturation_range)
        if np.abs(saturation-1.0) > 0.05:
            image = ImageEnhance.Color(image).enhance(saturation)
        contrast = random.uniform(*self.contrast_range)
        if np.abs(contrast-1.0) > 0.05:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        return image.resize((self.image_size, self.image_size), Image.LANCZOS)


    def read(self, file):
        img = Image.open(os.path.join(self.input_folder, file))
        return img

    def write(self, image, name=None):
        image.shape = self.image_size, self.image_size, 3
        img = Image.fromarray(np.array(image, dtype=np.uint8))
        if name is None:
            fullname = "test.png"
        else:
            fullname = name+".png"
        img.save(os.path.join(self.output_folder, fullname), "PNG")


if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        img = ImageManager(keep_in_memory=False)
        for i in range(int(os.sys.argv[1])):
            img.write(img.get_batch(1)[0],"variant_%d"%i)
        print("Generated %s image variations as they are when fed to the network"%os.sys.argv[1])
    else:
        print("Testing memory requiremens")
        img = ImageManager()
        input("Press Enter to continue... (all images loaded)")
        iml1 = img.get_batch(BATCH_SIZE*2)
        input("Press Enter to continue... (also one double batch)")
