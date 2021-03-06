from unittest import TestCase
from linear_model import LinearModel
from chainer.optimizers import AdaDelta
from chainer.testing import attr

class TestAdaDelta(TestCase):
    def setUp(self):
        self.optimizer = AdaDelta(eps=1e-5)
        self.model = LinearModel(self.optimizer)

    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy(False), 0.75)

    @attr.gpu
    def test_linear_model_gpu(self):
        self.assertGreater(self.model.accuracy(True), 0.75)
