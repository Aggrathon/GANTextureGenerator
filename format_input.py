import sys, os
from PIL import Image, ImageFilter


RESULT_FOLDER = "input"
RESULT_SIZE = 512


def remove_files_from_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def process_image(file, input_folder, output_folder, size):
    infile = os.path.join(input_folder, file)
    if os.path.isfile(infile):
        f, e = os.path.splitext(file)
        outfile = os.path.join(output_folder, f+".png")
        try:
            image = Image.open(infile)
            size = min(image.size)
            pos = ((image.size[0]-size)/2, (image.size[1]-size)/2)
            image = image.crop((pos[0], pos[1], pos[0]+size, pos[1]+size))
            image.thumbnail((RESULT_SIZE, RESULT_SIZE), Image.LANCZOS)
            image.save(outfile+".png", "PNG")
            print("Processed", file)
        except IOError:
            print("Could not process", file)


def process_folder(folder):
    print("Cleaning Result Folder")
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    remove_files_from_folder(RESULT_FOLDER)
    print("Processing images")
    for file_name in os.listdir(folder):
        process_image(file_name, folder, RESULT_FOLDER, RESULT_SIZE)


# Requires folder
if len(sys.argv) < 2:
    print("No folder specified!")
    print("Usage: %s <directory/with/images>"%sys.argv[0])
    print()
    print("This script takes an directory of images and processes them.")
    print(" The results are saved in the '%s' directory."%RESULT_FOLDER)
else:
    process_folder(sys.argv[1])
