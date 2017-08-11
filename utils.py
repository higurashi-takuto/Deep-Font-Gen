import glob

import numpy as np

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e


def read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


def preprocess(li):
    path_inp, path_out, char_out = li
    img_inp = read_image_as_array(path_inp)
    img_out = read_image_as_array(path_out)
    return img_inp, img_out, char_out


def make_list():
    image_list = []
    fonts = glob.glob('images/*')
    for fontname in fonts:
        for input_image in range(218):
            for target_image in range(218):
                if input_image != target_image:
                    image_list.append(
                        ['{}/{}.png'.format(fontname, input_image),
                         '{}/{}.png'.format(fontname, target_image),
                         target_image])
    return image_list
