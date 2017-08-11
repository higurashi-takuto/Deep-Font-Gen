import glob
import os

import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e


def read_image_as_array(path):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=np.float32)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return np.expand_dims(image, 0)


def preprocess(li):
    path_inp, path_out, char_out = li
    img_inp = read_image_as_array(path_inp)
    img_out = read_image_as_array(path_out)

    img_inp *= 2 / 255
    img_out *= 2 / 255

    img_inp -= 1
    img_out -= 1

    return img_inp, img_out, np.array(char_out, dtype=np.int32)


def make_list(root):
    image_list = []
    fontnames = glob.glob(os.path.join(root, '*'))
    for fontname in fontnames:
        images = glob.glob(os.path.join(fontname, '*.png'))
        for target_image in images:
            target_chr = os.path.basename(target_image).replace('.png', '')
            for input_image in images:
                if input_image != target_image:
                    image_list.append([input_image, target_image, target_chr])
    return image_list


def set_opt(model, optimizer, *hooks):
    optimizer.setup(model)
    for hook in hooks:
        optimizer.add_hook(hook)
    return optimizer


def generate_sample(generator, iterator, device):
    trigger = (100, 'iteration')

    @chainer.training.make_extension(trigger)
    def generate(trainer):
        if trigger[1] == 'epoch':
            n = trainer.updater.epoch
        else:
            n = trainer.updater.iteration

        iterator.reset()
        converter = chainer.dataset.convert.concat_examples
        out = os.path.join(trainer.out,
                           '{}{}'.format(trigger[1], n))
        try:
            os.makedirs(out)
        except OSError:
            pass

        for i, batch in enumerate(iterator):
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('train', False):
                    x, x_real, char = converter(batch, device)

                    # generate
                    x_fake = generator(x, char).data

                    # to cpu
                    x = chainer.cuda.to_cpu(x).reshape((-1, 64))
                    x_real = chainer.cuda.to_cpu(x_real).reshape((-1, 64))
                    x_fake = chainer.cuda.to_cpu(x_fake).reshape((-1, 64))

                    # save spectrogram
                    result = np.hstack((x, x_real, x_fake))
                    plt.figure()
                    plt.imshow(result, cmap='gray', vmin=-1, vmax=1)
                    plt.savefig(os.path.join(out, '{}.png'.format(i)))
                    plt.close()

    return generate
