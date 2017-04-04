
import math
import os
import random
import time
from threading import Event, Thread

import numpy as np
from PIL import Image, ImageEnhance

from queue import Queue


class ImageVariations():
    def __init__(self, image_size=64, batch_size=64, colored=True,
                 in_memory=True, in_directory='input', out_directory='output',
                 rotation_range=(-20, 20), brightness_range=(0.7, 1.2),
                 saturation_range=(0.7, 1.), contrast_range=(0.9, 1.3)):
        #Parameters
        self.image_size = image_size
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.in_directory = in_directory
        self.out_directory = out_directory
        #Variation Config
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.colored = colored
        #Thread variables
        self.queue = Queue()
        self.files = []
        self.threads = []
        self.event = Event()
        self.closing = True

    def start_threads(self):
        """Start the threads that are generating image variations"""
        self.closing = True
        self.event.set()
        self.files = os.listdir(self.in_directory)
        num_threads = os.cpu_count()
        if num_threads is None:
            num_threads = 4
        self.threads = [Thread(target=self.__thread__, args=(self.files[i::num_threads],), daemon=True)
                        for i in range(num_threads)]
        self.event.clear()
        self.closing = False
        for t in self.threads:
            t.start()

    def stop_threads(self):
        """Stop the threads that are generating image variations (freeing memory)"""
        self.closing = True
        self.event.set()

    def get_batch(self):
        """Get a batch of images as arrays"""
        if self.closing:
            self.start_threads()
        self.event.set()
        images = [self.queue.get() for i in range(self.batch_size)]
        self.event.clear()
        return images

    def __thread__(self, files):
        if self.in_memory:
            if self.colored:
                images = [Image.open(os.path.join(self.in_directory, file)) for file in files]
            else:
                images = [Image.open(os.path.join(self.in_directory, file)).convert("L") for file in files]
        while True:
            if self.in_memory:
                image = random.choice(images)
            else:
                image = Image.open(os.path.join(self.in_directory, random.choice(files)))
                if not self.colored:
                    image = image.convert("L")
            arr = np.asarray(self.get_variation(image), dtype=np.float)
            if not self.colored:
                arr.shape = arr.shape+(1,)
            self.queue.put(arr)
            if self.closing:
                return
            if self.queue.qsize() >= self.batch_size*3:
                self.event.wait()

    def get_variation(self, image):
        """Get an variation of the image according to the object config"""
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
        rotation = random.randint(*self.rotation_range)
        image = image.rotate(rotation).crop((offset, offset, offset+size, offset+size))
        #Transpose
        if random.random() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
        #Variation
        brightness = random.uniform(*self.brightness_range)
        if np.abs(brightness-1.0) > 0.05:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if self.colored:
            saturation = random.uniform(*self.saturation_range)
            if np.abs(saturation-1.0) > 0.05:
                image = ImageEnhance.Color(image).enhance(saturation)
        contrast = random.uniform(*self.contrast_range)
        if np.abs(contrast-1.0) > 0.05:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        return image.resize((self.image_size, self.image_size), Image.LANCZOS)


    def save_image(self, image, name=None):
        os.makedirs(self.out_directory, exist_ok=True)
        if self.colored:
            image.shape = self.image_size, self.image_size, 3
            img = Image.fromarray(np.array(image, dtype=np.uint8), "RGB")
        else:
            image.shape = self.image_size, self.image_size
            img = Image.fromarray(np.array(image, dtype=np.uint8), "L")
        add_time = time.time() - 1500000000
        if name is None:
            path = os.path.join(self.out_directory, "%d_test.png"%add_time)
        else:
            path = os.path.join(self.out_directory, '%d_%s.png'%(add_time, name))
        img.save(path, 'PNG')


if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        imgvariations = ImageVariations(in_memory=False, batch_size=int(os.sys.argv[1]))
        imgvariations.start_threads()
        images_batch = imgvariations.get_batch()
        imgvariations.stop_threads()
        for i in range(int(os.sys.argv[1])):
            imgvariations.save_image(images_batch[i], name="variant_%d"%i)
        print("Generated %s image variations as they are when fed to the network"%os.sys.argv[1])
    else:
        print("Testing memory requiremens")
        imgvariations = ImageVariations()
        input("Press Enter to continue... (all images loaded)")
        iml1 = imgvariations.get_batch()
        iml2 = imgvariations.get_batch()
        iml3 = imgvariations.get_batch()
        iml4 = imgvariations.get_batch()
        input("Press Enter to continue... (also four batches)")
