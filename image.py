
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageEnhance


class ImageVariations():
    def __init__(self, image_size=64, colored=True, pool_size=10000,
                 in_directory='input', out_directory='output',
                 rotation_range=(-15, 15), brightness_range=(0.7, 1.2),
                 saturation_range=(0.7, 1.), contrast_range=(0.9, 1.3),
                 size_range=(0.6, 0.8)):
        #Parameters
        self.image_size = image_size
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.images_count = pool_size
        #Variation Config
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.size_range = size_range
        self.colored = colored
        #Generate Images
        self.index = 0
        if self.images_count > 0:
            if self.images_count > 20:
                print("Processing Images")
            files = [f for f in os.listdir(self.in_directory) if os.path.isfile(os.path.join(self.in_directory, f))]
            np.random.shuffle(files)
            mp = self.images_count//len(files)
            rest = self.images_count%len(files)
            if mp > 0:
                pool = Pool()
                images = pool.starmap(self.__generate_images__, [(f, mp) for f in files])
                self.pool = [img for sub in images for img in sub]
                pool.close()
            else:
                self.pool = []
            self.pool += [img for sub in [self.__generate_images__(f, 1) for f in files[:rest]] for img in sub]
            np.random.shuffle(self.pool)

    def __generate_images__(self, image_file, iterations):
        if self.colored:
            image = Image.open(os.path.join(self.in_directory, image_file))
        else:
            image = Image.open(os.path.join(self.in_directory, image_file)).convert("L")
        def variation_to_numpy():
            arr = np.asarray(self.get_variation(image), dtype=np.float)
            if not self.colored:
                arr.shape = arr.shape+(1,)
            return arr
        return [variation_to_numpy() for _ in range(iterations)]


    def get_batch(self, count):
        """Get a batch of images as arrays"""
        if self.index + count < len(self.pool):
            batch = self.pool[self.index:self.index+count]
            self.index += count
            return batch
        else:
            batch  = self.pool[self.index:]
            self.index = 0
            np.random.shuffle(self.pool)
            return batch + self.get_batch(count - len(batch))

    def get_rnd_batch(self, count):
        if count > len(self.pool):
            return self.get_batch(count)
        index = np.random.randint(0, len(self.pool)-count)
        return self.pool[index:index+count]

    def get_variation(self, image):
        """Get an variation of the image according to the object config"""
        #Crop
        min_dim = min(image.size)
        scale = random.uniform(*self.size_range)
        size = int(random.random()*min_dim*(1-scale)+min_dim*scale)
        pos = (random.randrange(0, image.size[0]-size), random.randrange(0, image.size[1]-size))
        image = image.crop((pos[0], pos[1], pos[0]+size, pos[1]+size))
        #Rotate
        rotation = random.randint(*self.rotation_range)
        sina = np.abs(np.sin(np.deg2rad(rotation)))
        b = size / (1+sina)
        a = sina*size / (1+sina)
        size = int(np.sqrt(a*a + b*b))-1
        offset = (image.size[0]-size)/2
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
        add_time = time.time() - 1490000000
        if name is None:
            path = os.path.join(self.out_directory, "%d_test.png"%add_time)
        else:
            path = os.path.join(self.out_directory, '%d_%s.png'%(add_time, name))
        img.save(path, 'PNG')


if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        num_imgs = int(os.sys.argv[1])
        imgvariations = ImageVariations(pool_size=num_imgs)
        images_batch = imgvariations.get_batch(num_imgs)
        for variant_id in range(num_imgs):
            imgvariations.save_image(images_batch[variant_id], name="variant_%d"%variant_id)
        print("Generated %i image variations as they are when fed to the network"%num_imgs)
    else:
        print("Testing memory requiremens")
        num_imgs = 10000
        imgvariations = ImageVariations(pool_size=num_imgs)
        input("Press Enter to continue... (Pool countains %i images)"%num_imgs)
