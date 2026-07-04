import PIL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os, math
from skimage import color
from enum import Enum
from scipy.spatial.distance import cdist

# TODO: 
# When applying filter, keep only the top layer, if it's the clear winner. Only allow blending 
# when the top two layers are relatively close to each other.
# Improve centroid selection. Here are some ideas:
# - FPS (Farthest Point Sampling)
# - K-means + centroid pruning

class Mode(Enum):
    Gradient = 0
    Bucket   = 1

class Options:
    def __init__(self, k: int, mode: Mode, max_layers: int, alpha: float = 1., invert: bool = False):
        self.k = k
        self.num_initial_k = k
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
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
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

def prune_model(pixels: np.ndarray, model: sklearn.cluster.KMeans, k: int) -> list:
    centroids = model.cluster_centers_
    labels = model.predict(pixels)
    # Step 1: Gather statistics
    K = len(centroids)
    counts = np.bincount(labels, minlength=K)
    sums = np.zeros((K, 3), dtype=np.float64)
    np.add.at(sums, labels, pixels)
    # Recompute centroids from the actual assigned pixels
    centroids = sums / counts[:, None]
    while len(centroids) > k:
        # Step 2: Find the closest pair
        # TODO: Use score instead of just distance
        dist = cdist(centroids, centroids)
        # Ignore self-distances
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        # Step 3: Merge them
        counts[i] += counts[j]
        sums[i] += sums[j]
        centroids[i] = sums[i] / counts[i]
        # Step 4: Update labels
        # All j pixels are reassigned to i
        labels[labels == j] = i
        # Step 5: Remove the old cluster
        keep = np.arange(K) != j
        # print(f'keep: {keep}')
        # print(f'counts: {counts}')
        counts = counts[keep]
        # print(f'sums: {sums}')
        sums = sums[keep]
        # print(f'centroids: {centroids}')
        centroids = centroids[keep]
        # Shift all indices after j by 1
        labels[labels > j] -= 1
        K -= 1
    return centroids, labels

def rgb2kmeans(input_path, options: Options):
    img = load_image(input_path)
    img = (img / 255.).astype(np.float32)
    # Use LAB color space, as that maps better to Euclidean distance than RGB 
    lab = color.rgb2lab(img)
    pixels = lab.reshape(-1, 3)
    model = sklearn.cluster.KMeans(n_clusters=options.num_initial_k, random_state=0).fit(pixels)
    centroids, labels = prune_model(pixels, model, options.k)
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
        for i in range(options.k):
            l = np.where(labels == i, 0, 1).astype(np.float32)
            layers.append(l.reshape((img.shape[0], img.shape[1], 1)))
    else:
        raise ValueError(f'Unexpected value for mode {options.mode}')
    return layers, centroids

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert RGB images to layers using K-means algorithm')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-p', '--prefix', default='./', help='Output path prefix')
    parser.add_argument('-k', '--num_centroids', type=int, default=2, help='Number of layers')
    parser.add_argument('--initial_centroids', type=int, default=20, help='Initial number of centroids before pruning')
    parser.add_argument('-i', '--inverse', action='store_true', help='Invert layer weights')
    parser.add_argument('-a', '--alpha', type=float, default=1., help='Scaling factor for distance function. Used with gradient mode.')
    parser.add_argument('-b', '--bucket', action='store_true', help='Use bucketing mode instead of gradient. Results in distinct layers with flat colors instead of overlapping gradients.')
    parser.add_argument('-l', '--max_overlapping_layers', default=2, help='Maximum number of overlapping gradients per pixel.')
    parser.add_argument('--preview', action='store_true', help='Display color previews of the separated layers.')
    args = parser.parse_args()
    if args.bucket:
        options = Options(args.num_centroids, Mode.Bucket)
    else:
        options = Options(args.num_centroids, Mode.Gradient, args.max_overlapping_layers, args.alpha, args.inverse)
    options.num_initial_k = args.initial_centroids
    layers, centroids = rgb2kmeans(args.input, options)
    # Layer previews
    if args.preview:
        white_lab = np.array([100., 0.01, -0.01])
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
        save_image(layers[i], f'{args.prefix}layer_{i + 1}.png')