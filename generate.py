
import os

from train import CONFIG, get_network, IMAGE_CONFIG


def get_config(batch):
    config = CONFIG
    config['batch_size'] = batch
    config['log'] = False
    config['grid_size'] = int(batch**0.5)
    return config

def generate(name, amount=1):
    gan = get_network(name, **get_config(amount))
    session, _, iteration = gan.get_session()
    if iteration == 0:
        print("No already trained network found (%s)"%name)
        return
    print("Generating %d images using the %s network"%(amount, name))
    gan.generate(session, gan.name, amount)

def generate_grid(name, size=5):
    gan = get_network(name, **get_config(size*size))
    session, _, iteration = gan.get_session()
    if iteration == 0:
        print("No already trained network found (%s)"%name)
        return
    print("Generating a image grid using the %s network"%name)
    gan.generate_grid(session, gan.name)

if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        print('Usage:')
        print('  python %s network_name [num_images]\t- Generates images to the output folder'%os.sys.argv[0])
        print('  python %s network_name [grid]\t- Generates an image grid to the output folder'%os.sys.argv[0])
        print('  python %s images [num_images]\t- Processes input images to the output folder'%os.sys.argv[0])
    elif os.sys.argv[1] == 'images':
        from image import ImageVariations
        conf = IMAGE_CONFIG
        conf['pools'] = 1
        if len(os.sys.argv) == 3:
            conf['batch_size'] = int(os.sys.argv[2])
        else:
            conf['batch_size'] = 1
        imgs = ImageVariations(image_size=CONFIG['image_size'], **conf)
        imgs.start_threads()
        images_batch = imgs.get_batch()
        imgs.stop_threads()
        for variant_id in range(conf['batch_size']):
            imgs.save_image(images_batch[variant_id], name="variant_%d"%variant_id)
        print("Generated %s image variations as they are when fed to the network"%os.sys.argv[1])
    elif len(os.sys.argv) < 3:
        generate(os.sys.argv[1])
    else:
        if os.sys.argv[2] == 'grid':
            generate_grid(os.sys.argv[1])
        else:
            generate(os.sys.argv[1], int(os.sys.argv[2]))
