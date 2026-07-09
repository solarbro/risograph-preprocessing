import PIL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os, math
from skimage import color
from enum import Enum
from scipy.spatial.distance import cdist

class Mode(Enum):
    Gradient = 0
    Bucket   = 1

class Options:
    def __init__(self, k: int, mode: Mode, max_layers: int = 20, gamma: float = 1., invert: bool = False):
        self.k = k
        self.num_initial_k = k
        self.mode = mode
        if self.mode == Mode.Gradient:
            self.max_layers = max_layers
            self.winner_threshold = None
            self.gamma = gamma
            self.invert = invert
            self.chroma_cutoff = 20.0

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

def ink_weight(X, Y, gamma, chroma_factor):
    # Convert to LCH
    XL, XC, XH = lab_to_lch(X)
    YL, YC, YH = lab_to_lch(np.expand_dims(Y, axis=0))
    # Compute hue similarity against centroid
    dH = circular_hue_diff(YH[:, None], XH[None,:])
    hue_similarity = (1.0 - dH / np.pi) ** gamma
    # At this point 1 = similar hue, 0 = opposite hue
    # Compute darkness. 0 = white, 1 = black
    darkness = 1.0 - XL / 100.0
    # Handle gray tones
    chroma_confidence = np.clip(XC / chroma_factor, 0.0, 1.0)
    # Final score
    score = hue_similarity * darkness[None, ...] * chroma_confidence[None, ...]
    # We want to use black ink 
    # neutrality = 1.0 - chroma_confidence 
    # Darkening factor is based on how much darker the pixel is relative to the centroid.
    darkening_factor = np.clip(XL / YL, 0.0, 1.0)
    b = darkening_factor
    # Apply gamma to black layer as well
    b = b ** gamma
    # Invert weight - 0 = full ink, 1 = no ink
    return 1.0 - score, 1.0 - b

def normalize_weight(weight):
    return (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)

def save_image(layer, path):
    img = (layer * 255).astype(np.uint8)
    img = PIL.Image.fromarray(np.squeeze(img, axis=-1))
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    img.save(path)
    print(f'Saved layer: {path}')

def keep_top_n_layers(layers: list[np.ndarray], n: int, renormalize: bool) -> list[np.ndarray]:
    if n >= len(layers):
        # No need to filter
        return layers
    # print(f'Keep top {n} weights')
    # (K, H, W)
    weights = np.stack(layers, axis=0).squeeze(-1)
    # print(f'Shape of weights: {weights.shape}')
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

def keep_clear_winner(layers: list[np.ndarray], margin: float = 0.1) -> list[np.ndarray]:
    # (K, H, W)
    dist = np.stack(layers, axis=0).squeeze(-1)
    # Find the closest and second-closest surviving layers
    order = np.argsort(dist, axis=0)
    best = order[0]
    second = order[1]
    best_dist = np.take_along_axis(dist, order[:1], axis=0)[0]
    second_dist = np.take_along_axis(dist, order[1:2], axis=0)[0]
    clear = (second_dist - best_dist) > margin
    filtered = dist.copy()
    H, W = best.shape
    y, x = np.indices((H, W))
    # Remove all surviving layers at clear-winner pixels
    filtered[:, clear] = 1.0
    # Restore only the winner
    filtered[best[clear], y[clear], x[clear]] = best_dist[clear]
    # Split layers back
    filtered = np.expand_dims(filtered, axis=-1)
    return list(filtered)

# Hue difference with wraparound
def circular_hue_diff(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 2*np.pi - d)

def lab_to_lch(value):
    L = value[:, 0]
    a = value[:, 1]
    b = value[:, 2]
    C = np.sqrt(a*a + b*b)
    H = np.arctan2(b, a) # [-pi, pi]
    return L, C, H

def get_scores(centroids, counts):
    L, C, H = lab_to_lch(centroids)
    dH = circular_hue_diff(H[:,None], H[None,:])
    dC = np.abs(C[:, None] - C[None, :])
    dL = np.abs(L[:, None] - L[None, :])
    # For gray tones, hue is undefined so it can end up being whatever value
    # To avoid the hue tone jumping around, apply an additional weight for low chroma colors
    chroma_weight = np.minimum(C[:, None], C[None, :]) / 20.0
    chroma_weight = np.clip(chroma_weight, 0.0, 1.0)
    score = chroma_weight * dH + 1.0 * (dC / 100.0) + 0.10 * (dL / 100.0)
    return score
    # dist = cdist(centroids, centroids)
    # Note: including counts in the score made the selection worse
    # size = counts[:, None] + counts[None, :]
    # return dist * size
    # return dist

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
    while K > k:
        # Step 2: Find the closest pair
        score = get_scores(centroids, counts)
        # Ignore self-distances
        np.fill_diagonal(score, np.inf)
        i, j = np.unravel_index(np.argmin(score), score.shape)
        # Step 3: Merge them
        # ----------------------------------------
        # Method 1
        counts[i] += counts[j]
        sums[i] += sums[j]
        centroids[i] = sums[i] / counts[i]
        # Method 2. More or less the same output as 1.
        # ci = counts[i]
        # cj = counts[j]
        # t = 1.0 / (ci + cj)
        # wi = ci * t
        # wj = cj * t
        # centroids[i] = wi * centroids[i] + wj * centroids[j]
        # ----------------------------------------
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
        # Update stats
        K -= 1
    return centroids, labels

def rgb2kmeans(img, options: Options):
    # Use LAB color space, as that maps better to Euclidean distance than RGB 
    lab = color.rgb2lab(img)
    pixels = lab.reshape(-1, 3)
    model = sklearn.cluster.KMeans(n_clusters=options.num_initial_k, random_state=0).fit(pixels)
    centroids, labels = prune_model(pixels, model, options.k - 1)
    layers = []
    # print(pixels.shape)
    if options.mode == Mode.Gradient:
        black = np.zeros((pixels.shape[0],))
        for i in range(len(centroids)):
            # l = distance(pixels, centroids[i])
            # In LAB colorspace the distance isn't in the range 0:1 so it's not suitable for interpolation
            # Divide by an arbitrary max value to "normalize"
            # l /= 200.
            # l = np.clip(l * options.alpha, 0, 1)
            l, b = ink_weight(pixels, centroids[i], options.gamma, options.chroma_cutoff)
            black += b
            layers.append(l.reshape((img.shape[0], img.shape[1], 1)))
        black /= options.k
        layers.append((1 - black).reshape(img.shape[0], img.shape[1], 1))
        bc = np.array([[0.0, 0.0, 0.0]])
        centroids = np.vstack((centroids, bc))
        # centroids.append(np.zeros((3)))
        # Keep top N layers only and renormalize the weights of the remaining layers
        layers = keep_top_n_layers(layers, options.max_layers, False)
        if options.winner_threshold is not None:
            layers = keep_clear_winner(layers, options.winner_threshold)
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
    parser.add_argument('-g', '--gamma', type=float, default=1., help='Gain factor used with hue similarity score in gradient mode.')
    parser.add_argument('-b', '--bucket', action='store_true', help='Use bucketing mode instead of gradient. Results in distinct layers with flat colors instead of overlapping gradients.')
    parser.add_argument('-l', '--max_overlapping_layers', default=2, help='Maximum number of overlapping gradients per pixel.')
    parser.add_argument('--preview', action='store_true', help='Display color previews of the separated layers.')
    args = parser.parse_args()
    if args.bucket:
        options = Options(args.num_centroids, Mode.Bucket)
    else:
        options = Options(args.num_centroids, Mode.Gradient, args.max_overlapping_layers, args.gamma, args.inverse)
    options.num_initial_k = args.initial_centroids
    img = load_image(args.input)
    img = (img / 255.).astype(np.float32)
    layers, centroids = rgb2kmeans(img, options)
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