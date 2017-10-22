import PIL.Image
import numpy as np


def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)
    if max_size is not None:
        factor = float(max_size) / np.max(image.size)
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS)

    return np.float32(image)


def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
