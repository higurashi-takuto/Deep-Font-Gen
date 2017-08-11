import chainer
from chainer import links as L
from chainer import functions as F
from chainer import reporter


class Generator(chainer.Chain):
    def __init__(self, n_charactor):
        super(Generator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, 128, 4, 2, 1)
            self.conv2 = L.Convolution2D(
                None, 256, 4, 2, 1)
            self.conv3 = L.Convolution2D(
                None, 512, 4, 2, 1)
            self.conv4 = L.Convolution2D(
                None, 1024, 4, 2, 1)
            self.embed = L.EmbedID(n_charactor, 256)
            self.deconv1 = L.Deconvolution2D(
                None, 512, 4, 2, 1)
            self.deconv2 = L.Deconvolution2D(
                None, 256, 4, 2, 1)
            self.deconv3 = L.Deconvolution2D(
                None, 128, 4, 2, 1)
            self.deconv4 = L.Deconvolution2D(
                None, 1, 4, 2, 1)

    def encode(self, x, char=None):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        if char is None:
            return h4
        emb_char = self.embed(char).reshape((-1, 16, 4, 4))
        h = F.concat((h4, emb_char))
        return h

    def decode(self, h):
        h1 = F.relu(self.deconv1(h))
        h2 = F.relu(self.deconv2(h1))
        h3 = F.relu(self.deconv3(h2))
        y = F.tanh(self.deconv4(h3))
        return y

    def __call__(self, x, char):
        h = self.encode(x, char)
        y = self.decode(h)
        return y


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, 128, 4, 2, 1)
            self.conv2 = L.Convolution2D(
                None, 256, 4, 2, 1)
            self.conv3 = L.Convolution2D(
                None, 512, 4, 2, 1)
            self.conv4 = L.Convolution2D(
                None, 1024, 4, 2, 1)
            self.fc = L.Linear(None, 1)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        y = self.fc(h4)
        return y
