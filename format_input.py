import sys, os, string, random, math
from PIL import Image, ImageEnhance


RESULT_FOLDER = "input"
RESULT_SIZE = 512
RESULT_CULLING = 0.3


def remove_files_from_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_random_name(folder, extension, length=8):
    pool = string.ascii_letters + string.digits
    while True:
        rnd = [random.choice(pool) for _ in range(length)]
        name = os.path.join(folder, ''.join(rnd)+extension)
        if not os.path.exists(name):
            return name


def process_image(infile, output_folder, size, culling):
    if os.path.isfile(infile):
        try:
            image = Image.open(infile)
            for im in transform_image(image, size):
                if random.random() > culling:
                    output_image(im, output_folder)
            print("Processed", os.path.basename(infile))
        except IOError:
            print("Could not process", os.path.basename(infile))


def transform_image(image, size):
    def get_variations(image):
        yield image
        yield ImageEnhance.Brightness(image).enhance(0.8)
        yield ImageEnhance.Color(image).enhance(0.8)
    def get_crops(image, size, num_var=5):
        min_dim = min(image.size)
        scale = max(0.75, math.sqrt(2)*size/min_dim*1.1)
        for _ in range(num_var):
            size = int(random.random()*min_dim*(1-scale)+min_dim*scale)
            pos = (random.randrange(0,image.size[0]-size), random.randrange(0,image.size[1]-size))
            yield image.crop((pos[0], pos[1], pos[0]+size, pos[1]+size))
    def get_transforms(image):
        yield image
        yield image.copy().transpose(Image.FLIP_LEFT_RIGHT)
    def get_rotations(image, max_rot=30):
        yield image
        size = image.size[0]/math.sqrt(2)
        offset = (image.size[0]-size)/2
        yield image.copy().rotate(-random.randint(-max_rot, max_rot)).crop((offset, offset, offset+size, offset+size))
    for im1 in get_variations(image):
        for im2 in get_transforms(im1):
            for im3 in get_crops(im2, size):
                for im4 in get_rotations(im3):
                    yield im4

def output_image(image, folder):
    outfile = get_random_name(folder, ".png")
    size = min(image.size)
    pos = ((image.size[0]-size)/2, (image.size[1]-size)/2)
    image = image.crop((pos[0], pos[1], pos[0]+size, pos[1]+size))
    image.thumbnail((RESULT_SIZE, RESULT_SIZE), Image.LANCZOS)
    image.save(outfile, "PNG")


def process_folder(folder):
    print("Cleaning Result Folder")
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    remove_files_from_folder(RESULT_FOLDER)
    print("Processing images")
    for file_name in os.listdir(folder):
        process_image(os.path.join(folder, file_name), RESULT_FOLDER, RESULT_SIZE, RESULT_CULLING)


# Requires folder
if len(sys.argv) < 2:
    print("No folder specified!")
    print("Usage: %s <directory/with/images>"%sys.argv[0])
    print()
    print("This script takes an directory of images and processes them.")
    print(" The results are saved in the '%s' directory."%RESULT_FOLDER)
else:
    process_folder(sys.argv[1])
