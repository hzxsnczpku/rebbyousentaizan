import argparse
import os
import numpy as np
import tensorflow as tf
from style_transfer import StyleTransfer
import utils
import vgg19

used_gpu = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu


def parse_args():
    desc = "Tensorflow implementation of 'Image Style Transfer Using Convolutional Neural Networks"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_path', type=str, default='pre_trained_model',)
    parser.add_argument('--content', type=str, help='File path of content image', required=True)
    parser.add_argument('--style', type=str, nargs='+', help='File path of style image', required=True)
    parser.add_argument('--output', type=str, help='File path of output image', required=True)
    parser.add_argument('--loss_ratio_c', type=float, default=1e-3, help='Weight of content-loss relative to style-loss')
    parser.add_argument('--loss_ratio_tv', type=float, default=1e-2,
                        help='Weight of total-variance relative to style-loss')
    parser.add_argument('--content_layers', nargs='+', type=str, default=['conv3_2', 'conv4_2', 'conv5_2'])
    parser.add_argument('--style_layers', nargs='+', type=str,
                        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[.3, .4, .3])
    parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2, .2, .2, .2, .2])
    parser.add_argument('--initial_type', type=str, default='content', choices=['random', 'content', 'style'])
    parser.add_argument('--max_size', type=int, default=512, help='The maximum width or height of input images')
    parser.add_argument('--content_loss_norm_type', type=int, default=1, choices=[1, 2])
    parser.add_argument('--num_iter', type=int, default=1500)
    parser.add_argument('--content_mask', type=str, help='File path of content mask image', default=None)
    return parser.parse_args()


def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)


def main():
    args = parse_args()
    model_file_path = args.model_path + '/' + vgg19.MODEL_FILE_NAME
    vgg_net = vgg19.VGG19(model_file_path)

    content_image = utils.load_image(args.content, max_size=args.max_size)

    style_image = []
    for style_image_path in args.style:
        style_image.append(utils.load_image(style_image_path, shape=(content_image.shape[1], content_image.shape[0])))
    style_image = np.array(style_image)

    content_mask = None
    if args.content_mask is not None:
        content_mask = utils.load_image(args.content_mask, shape=(content_image.shape[1], content_image.shape[0]))
        content_mask = content_mask/255.

    # initial guess for output
    if args.initial_type == 'content':
        init_image = content_image
    elif args.initial_type == 'style':
        init_image = style_image
    elif args.initial_type == 'random':
        init_image = np.random.normal(size=content_image.shape, scale=np.std(content_image))

    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight

    # open session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # build the graph
    st = StyleTransfer(session=sess, content_layer_ids=CONTENT_LAYERS, style_layer_ids=STYLE_LAYERS,
                       init_image=add_one_dim(init_image), content_image=add_one_dim(content_image),
                       style_image=style_image, net=vgg_net, num_iter=args.num_iter,
                       loss_ratios=[args.loss_ratio_c, args.loss_ratio_tv],
                       content_loss_norm_type=args.content_loss_norm_type, content_mask=content_mask)
    result_image = st.update()
    sess.close()

    shape = result_image.shape
    result_image = np.reshape(result_image, shape[1:])
    utils.save_image(result_image, args.output)


if __name__ == '__main__':
    main()
