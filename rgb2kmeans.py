import PIL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
from skimage import color
from enum import Enum

# TODO: 
# [ ] In gradient mode, keep the pixels for the top N layers, and discard the rest (set to white)
# [ ] Normalize layer weights so that overlapping layers sum to 1. This should apply after the pixel discard pass

class Mode(Enum):
    Gradient = 0
    Bucket   = 1

class Options:
    def __init__(self, k: int, mode: Mode, max_layers: int, alpha: float = 1., invert: bool = False):
        self.k = k
        self.mode = mode
        if self.mode == Mode.Gradient:
            self.max_layers = max_layers
            self.alpha = alpha
            self.invert = invert

def load_image(path):
    img = PIL.Image.open(path)
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], len(img.mode))
    if img.shape[-1] > 3:
        img = img[:,:,:-1]
    return img

def cosine_similarity(X, Y):
    XdotY = np.sum(X * Y, axis=-1, keepdims=True)
    normX = np.linalg.norm(X, axis=-1, keepdims=True)
    normY = np.linalg.norm(Y, axis=-1, keepdims=True)
    sim = XdotY / (normX * normY + 1e-5)
    return sim

def distance(X, Y):
    # diff = Y - X
    # return np.sum(diff * diff, axis=-1, keepdims=True)
    return np.linalg.norm(Y - X, axis=-1, keepdims=True)

def normalize_weight(weight):
    return (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)

def save_image(layer, path):
    img = (layer * 255).astype(np.uint8)
    img = PIL.Image.fromarray(np.squeeze(img, axis=-1))
    img.save(path)

# TODO: This isn't working the way I need it to
def keep_top_n_layers(layers: list[np.ndarray], n: int, renormalize: bool) -> list[np.ndarray]:
    print(f'Keep top {n} weights')
    # (K, H, W)
    weights = np.stack(layers, axis=0).squeeze(-1)
    print(f'Shape of weights: {weights.shape}')
    # Find the top N layer indices for each pixel
    top = np.argpartition(weights, n - 1, axis=0)[:n]
    # Discard bottom k - n layers
    filtered = np.ones_like(weights)
    y, x = np.indices(weights.shape[1:])
    np.put_along_axis(filtered, top, np.take_along_axis(weights, top, axis=0), axis=0)
    if renormalize:
        filtered /= np.maximum(filtered.sum(axis=0, keepdims=True), 1e-8)
    # Restore the trailing singleton dimension
    filtered = np.expand_dims(filtered, axis=-1)
    return list(filtered)

def rgb2kmeans(input_path, options: Options, output_prefix):
    img = load_image(input_path)
    img = (img / 255.).astype(np.float32)
    # Use LAB color space, as that maps better to Euclidean distance than RGB 
    lab = color.rgb2lab(img)
    # rgb2 = color.lab2rgb(lab)
    # plt.imshow(rgb2)
    # plt.show()
    pixels = lab.reshape(-1, 3)
    model = sklearn.cluster.KMeans(n_clusters=options.k, random_state=0).fit(pixels)
    centroids = model.cluster_centers_
    white = np.array([1., 1., 1.])
    white_lab = np.array([100., 0.01, -0.01])
    # print(f'Centroids: {centroids}')
    # print(f'Buckets: {buckets}')
    # clamped = np.take(centroids, buckets, axis=0)
    layers = []
    if options.mode == Mode.Gradient:
        for i in range(options.k):
            l = distance(pixels, centroids[i])
            # In LAB colorspace the distance isn't in the range 0:1 so it's not suitable for interpolation
            # Divide by an arbitrary max value to "normalize"
            l /= 200.
            l = np.clip(l * options.alpha, 0, 1)
            layers.append(l.reshape((img.shape[0], img.shape[1], 1)))
        # Keep top N layers only and renormalize the weights of the remaining layers
        layers = keep_top_n_layers(layers, options.max_layers, False)
        # Invert weights
        if options.invert:
            for l in layers:
                l = 1 - l
    elif options.mode == Mode.Bucket:
        labels = model.predict(pixels)
        for i in range(options.k):
            l = np.where(labels == i, 0, 1).astype(np.float32)
            layers.append(l.reshape((img.shape[0], img.shape[1], 1)))
    else:
        raise ValueError(f'Unexpected value for mode {options.mode}')
    # Display layers in grid
    cols = math.ceil(math.sqrt(options.k))
    rows = math.ceil(options.k / cols)
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten() if options.k > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < options.k:
            preview = (1 - layers[i]) * centroids[i] + layers[i] * white_lab
            preview_rgb = color.lab2rgb(preview)
            ax.imshow(preview_rgb)
            # ax.imshow(layers[i], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{i + 1} - {centroids[i]}')
            ax.axis("off")
        else:
            # Hide unused subplot cells
            ax.axis("off")
    plt.tight_layout()
    plt.show()
    # Save layers
    for i in range(options.k):
        save_image(layers[i], f'{output_prefix}layer_{i + 1}.png')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert RGB images to layers using K-means algorithm')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-p', '--prefix', default='./', help='Output path prefix')
    parser.add_argument('-k', '--num_centroids', type=int, default=2, help='Number of layers')
    parser.add_argument('-i', '--inverse', action='store_true', help='Invert layer weights')
    parser.add_argument('-a', '--alpha', type=float, default=1., help='Scaling factor for distance function. Used with gradient mode.')
    parser.add_argument('-b', '--bucket', action='store_true', help='Use bucketing mode instead of gradient. Results in distinct layers with flat colors instead of overlapping gradients.')
    parser.add_argument('-l', '--max_overlapping_layers', default=2, help='Maximum number of overlapping gradients per pixel.')
    args = parser.parse_args()
    if args.bucket:
        options = Options(args.num_centroids, Mode.Bucket)
    else:
        options = Options(args.num_centroids, Mode.Gradient, args.max_overlapping_layers, args.alpha, args.inverse)
    rgb2kmeans(args.input, options, args.prefix)