import collections

import numpy as np
import tensorflow as tf


class StyleTransfer:
    def __init__(self, content_layer_ids, style_layer_ids, init_image, content_image,
                 style_image, session, net, num_iter, loss_ratios, content_loss_norm_type, content_mask=None):
        self.net = net
        self.sess = session

        # sort layers info
        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

        # preprocess input images
        self.p0 = np.float32(self.net.preprocess(content_image))
        self.a0 = np.float32(self.net.preprocess(style_image))
        self.x0 = np.float32(self.net.preprocess(init_image))
        self.content_mask = tf.Variable(content_mask) if content_mask is not None else None

        # parameters for optimization
        self.content_loss_norm_type = content_loss_norm_type
        self.num_iter = num_iter
        self.loss_ratios = loss_ratios

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):
        self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

        # graph input
        self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
        self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

        # get content-layer-feature for content loss
        content_layers = self.net.feed_forward(self.p, scope='content')
        style_layers = self.net.feed_forward(self.a, scope='style')
        self.Ps = {}
        self.Qs = {}
        self.As = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]
            self.Qs[id] = style_layers[id]

        for id in self.STYLE_LAYERS:
            self.As[id] = self._gram_matrix(style_layers[id])

        # get layer-values for x

        self.Fs = self.net.feed_forward(self.x, scope='mixed')
        if self.content_mask is not None:
            self.x_mask = self.x * tf.stop_gradient(self.content_mask)
            self.Fs_mask = self.net.feed_forward(self.x_mask, scope='mixed_mask')

        """ compute loss """
        L_content = 0
        L_style = 0
        for id in self.Fs:
            if id in self.CONTENT_LAYERS:
                F = self.Fs[id]  # content feature of x
                P = self.Ps[id]  # content feature of p

                _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
                N = h.value * w.value  # product of width and height
                M = d.value  # number of filters

                w = self.CONTENT_LAYERS[id]  # weight for this layer

                # You may choose different normalization constant
                if self.content_loss_norm_type == 1:
                    L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))
                if self.content_loss_norm_type == 2:
                    L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(
                        tf.pow((self.transform_fm(self.Qs[id], self.Ps[id]) - F), 2))

            if id in self.STYLE_LAYERS:
                if self.content_mask is not None:
                    F = self.Fs_mask[id]
                else:
                    F = self.Fs[id]
                _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
                N = h.value * w.value  # product of width and height
                M = d.value  # number of filters
                w = self.STYLE_LAYERS[id]  # weight for this layer
                G = self._gram_matrix(F)  # style feature of x
                A = self.As[id]  # style feature of a
                for g in G:
                    for a in A:
                        L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((g - a), 2)) / len(G)

        # fix beta as 1
        alpha = self.loss_ratios[0]
        beta = 1
        gamma = self.loss_ratios[1]

        self.L_content = L_content
        self.L_style = L_style
        self.L_tv = self.total_variation_loss(self.x)
        self.L_total = alpha * L_content + beta * L_style + gamma * self.L_tv

    def update(self):
        global _iter
        _iter = 0

        def callback(tl, cl, sl, tv):
            global _iter
            print('iter : %4d, ' % _iter,
                  'L_total : %g, L_content : %g, L_style : %g, L_total_var: %g' % (tl, cl, sl, tv))
            _iter += 1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.L_total, method='L-BFGS-B',
                                                           options={'maxiter': self.num_iter})

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        optimizer.minimize(self.sess, feed_dict={self.a: self.a0, self.p: self.p0},
                           fetches=[self.L_total, self.L_content, self.L_style, self.L_tv], loss_callback=callback)

        final_image = self.sess.run(self.x)
        final_image = np.clip(self.net.undo_preprocess(final_image), 0.0, 255.0)

        return final_image

    def _gram_matrix(self, tensor):
        shape = tensor.get_shape()
        num_channels = int(shape[3])
        gram = []
        for i in range(shape[0]):
            matrix = tf.reshape(tensor[i], shape=[-1, num_channels])
            gram.append(tf.matmul(tf.transpose(matrix), matrix))
        return gram

    def total_variation_loss(self, x):
        a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
        b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def transform_fm(self, featureE, featureI):
        gmax = 5
        gmin = 0.7
        G = featureE / (featureI + 1e-4)
        G = tf.maximum(tf.minimum(G, gmax), gmin)
        return G
