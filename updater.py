import chainer
from chainer import functions as F


class DFGUpdater(chainer.training.StandardUpdater):
    def __init__(self, opt_g, opt_d, iterator, device):
        self._iterators = {'main': iterator}
        self.generator = opt_g.target
        self.discriminator = opt_d.target
        self._optimizers = {'generator': opt_g, 'discriminator': opt_d}
        self.device = device
        self.converter = chainer.dataset.convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # read data
        batch = self._iterators['main'].next()
        x, x_real, char = self.converter(batch, self.device)
        iteration = self.iteration

        # forward
        x_fake = self.generator(x, char)

        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)

        h_real = self.generator.encode(x_real)
        h_fake = self.generator.encode(x_fake)

        # compute loss
        loss_recon = F.mean_absolute_error(x_real, x_fake)

        # update
        self.generator.cleargrads()
        loss_recon.backward()
        self._optimizers['generator'].update()

        # report
        chainer.reporter.report({
            'loss/recon': loss_recon
            })
