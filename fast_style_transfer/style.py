import os
from optimize import optimize
from argparse import ArgumentParser
from utils import get_img, list_files

used_gpu = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

CHECKPOINT_DIR = 'checkpoints'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', required=True)
    parser.add_argument('--style', type=str, dest='style', help='style image path', required=True)
    parser.add_argument('--train-path', type=str, dest='train_path', help='path to training images folder',
                        default=TRAIN_PATH)
    parser.add_argument('--epochs', type=int, dest='epochs', help='num epochs', default=2)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=4)
    parser.add_argument('--checkpoint-iterations', type=int, dest='checkpoint_iterations', help='checkpoint frequency',
                        default=500)
    parser.add_argument('--vgg-path', type=str, dest='vgg_path', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float, dest='content_weight', default=7.5e0)
    parser.add_argument('--style-weight', type=float, dest='style_weight', default=1e2)
    parser.add_argument('--tv-weight', type=float, dest='tv_weight', default=2e2)
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=1e-3)
    return parser


def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def main():
    parser = build_parser()
    opts = parser.parse_args()

    style_target = get_img(opts.style)
    content_targets = _get_files(opts.train_path)

    kwargs = {"epochs": opts.epochs, "print_iterations": opts.checkpoint_iterations, "batch_size": opts.batch_size,
              "save_path": os.path.join(opts.checkpoint_dir, 'fns.ckpt'), "learning_rate": opts.learning_rate}
    args = [content_targets, style_target, opts.content_weight, opts.style_weight, opts.tv_weight, opts.vgg_path]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses
        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)

if __name__ == '__main__':
    main()
