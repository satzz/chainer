from chainer import caffe, Function, FunctionSet, Variable
import chainer.functions as F

class CaffeModel(FunctionSet):
    insize = 224
    def __init__(self, model_path, arch):
        super(CaffeModel, self).__init__(
            func=caffe.CaffeFunction(model_path))

        if arch == 'nin':
            self.func.forward = self.forward_nin
        elif arch == 'googlenet':
            self.func.forward = self.forward_googlenet
        else:
            raise RuntimeError('Unsupported caffe model')

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)
        return self.func.forward(x, t, train=train)

    def forward_nin(self, x, t, train):
        y, = self.func(inputs={'data': x}, outputs=['pool4'], train=train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def forward_googlenet(self, x, t, train):
        l1, l2, l3, y = self.func(inputs={'data': x, 'label': t},
                                  outputs=['loss1/loss1',
                                           'loss2/loss1',
                                           'loss3/loss3',
                                           'loss3/classifier'],
                                  train=train)
        return (l1 + l2) * 0.3 + l3, F.accuracy(y, t)
