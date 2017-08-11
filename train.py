import argparse
import datetime
import os
import shutil

import pandas as pd
import numpy as np
import chainer
from chainer.training import extensions

from models import Generator
from models import Discriminator
from updater import DFGUpdater
from utils import make_list
from utils import preprocess
from utils import set_opt
import opt

# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU1 ID (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='number of process(es)')
args = parser.parse_args()

# load data
train = make_list(opt.root)
# valid = pd.read_csv()

# setup dataset iterator
train_dataset = chainer.datasets.TransformDataset(
    train, preprocess)
# valid_dataset = chainer.datasets.TransformDataset(
#     valid, preprocess)

if args.process > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, opt.batchsize, n_processes=arg.process)
    # valid_iter = chainer.iterators.MultiprocessIterator(
    #     valid_dataset, , opt.batchsize,
    #     repeat=False, shuffle=False, n_processes=arg.process)
else:
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, opt.batchsize)
    # valid_iter = chainer.iterators.SerialIterator(
    #     valid_dataset, , opt.batchsize,
    #     repeat=False, shuffle=False)

# make result directory
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, os.path.join(result, 'train.py'))
shutil.copy('utils.py', os.path.join(result, 'utils'))
shutil.copy('models.py', os.path.join(result, 'models.py'))
shutil.copy('updater.py', os.path.join(result, 'updater.py'))
shutil.copy('opt.py', os.path.join(result, 'opt.py'))

# setup models
generator = Generator(opt.n_charactor)
discriminator = Discriminator()
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    generator.to_gpu()
    discriminator.to_gpu()

# setup optimizers
opt_g = set_opt(generator, chainer.optimizers.Adam(opt.lr_gen, beta1=0.5),
                chainer.optimizer.GradientClipping(10),
                chainer.optimizer.WeightDecay(0.0001))
opt_d = set_opt(discriminator, chainer.optimizers.Adam(opt.lr_dis, beta1=0.5),
                chainer.optimizer.GradientClipping(10),
                chainer.optimizer.WeightDecay(0.0001))

# setup trainer
updater = DFGUpdater(opt_g, opt_d, train_iter, args.gpu)
trainer = chainer.training.Trainer(updater, opt.trigger, out=result)

# setup extensions
trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
trainer.extend(extensions.PrintReport(
    ['iteration', 'loss/recon']),
    trigger=(10, 'iteration'))
trainer.extend(extensions.ProgressBar(update_interval=10))

trainer.run()
