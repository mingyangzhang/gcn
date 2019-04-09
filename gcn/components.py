from inits import *
from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def multiply(x, y, sparse=False):
    """Wrapper for tf.math.multiply. """

    if sparse:
        raise NotImplementedError
    else:
        res = tf.math.multiply(x, y)
    return res

class Component(object):
    """ Prototype class for component that consists of layers. """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

    def _call(self, inputs):
        """ Pass input to layers and return final output. """

        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class MultiViewGCN(Component):
    """ MultiView GCN. """

    def __init__(self, view_num, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 featureless=False, **kwargs):

        super(MultiViewGCN, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.view_num = view_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.layers = []
        with tf.variable_scope(self.name + '_vars'):
            for v in range(self.view_num):
                self.vars['weights-{}'.format(v)] = glorot([self.output_dim, ], name='weights-{}'.format(v))

        if self.logging:
            self._log_vars()

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def _call(self, inputs):

        x = inputs
        assert len(x) == self.view_num, "Expect input view {}, got {}.".format(self.view_num, len(x))

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = dropout(x, 1-self.dropout)

        # compute GCN outputs of all views
        view_outputs = []
        for i in range(self.view_num):
            z = x[i]
            for layer in self.layers:
                z = layer(z)
            view_outputs.append(z)

        # fuse
        weighted_outputs = []
        for i, output in enumerate(view_outputs):
            weighted_outputs.append(
                multiply(output, self.vars['weights-{}'.format(i)], sparse=self.sparse_inputs))

        return tf.math.add_n(weighted_outputs)


class RGCN(Component):
    pass