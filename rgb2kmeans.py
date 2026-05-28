import PIL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
from skimage import color

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

def rgb2kmeans(input_path, num_buckets, alpha, invert, output_prefix):
    img = load_image(input_path)
    img = (img / 255.).astype(np.float32)
    # Use LAB color space, as that maps better to Euclidean distance than RGB 
    lab = color.rgb2lab(img)
    # rgb2 = color.lab2rgb(lab)
    # plt.imshow(rgb2)
    # plt.show()
    pixels = lab.reshape(-1, 3)
    model = sklearn.cluster.KMeans(n_clusters=num_buckets, random_state=0).fit(pixels)
    centroids = model.cluster_centers_
    white = np.array([1., 1., 1.])
    white_lab = np.array([100., 0.01, -0.01])
    # print(f'Centroids: {centroids}')
    # buckets = model.predict(pixels)
    # print(f'Buckets: {buckets}')
    # clamped = np.take(centroids, buckets, axis=0)
    layers = []
    for i in range(num_buckets):
        l = distance(pixels, centroids[i])
        # In LAB colorspace the distance isn't in the range 0:1 so it's not suitable for interpolation
        # Divide by an arbitrary max value to "normalize"
        l /= 200.
        l = np.clip(l * alpha, 0, 1)
        if invert:
            l = 1 - l
        layers.append(l.reshape((img.shape[0], img.shape[1], 1)))
    # Display layers in grid
    cols = math.ceil(math.sqrt(num_buckets))
    rows = math.ceil(num_buckets / cols)
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten() if num_buckets > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < num_buckets:
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
    for i in range(num_buckets):
        save_image(layers[i], f'{output_prefix}layer_{i + 1}.png')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert RGB images to layers using K-means algorithm')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-p', '--prefix', default='./', help='Output path prefix')
    parser.add_argument('-k', '--buckets', type=int, default=2, help='Number of layers')
    parser.add_argument('-i', '--inverse', action='store_true', help='Invert layer weights')
    parser.add_argument('-a', '--alpha', type=float, default=1., help='Scaling factor for distance function')
    args = parser.parse_args()
    rgb2kmeans(args.input, args.buckets, args.alpha, args.inverse, args.prefix)