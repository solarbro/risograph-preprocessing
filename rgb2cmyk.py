import PIL
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    img = PIL.Image.open(path)
    # Convert to numpy
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], len(img.mode))
    # Discard alpha channel
    if img.shape[-1] > 3:
        img = img[:,:,:-1]
    return img

def visualize_cmyk(img):
    c = img[:,:,0]
    m = img[:,:,1]
    y = img[:,:,2]
    k = img[:,:,3]
    fig, ax = plt.subplots(2, 2)
    def display_ax(ax, data, title):
        ax.imshow(data, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(title)
    display_ax(ax[0][0], c, 'C')
    display_ax(ax[0][1], m, 'M')
    display_ax(ax[1][0], y, 'Y')
    display_ax(ax[1][1], k, 'K')
    plt.show()

def save_image(layer, path):
    img = PIL.Image.fromarray(layer)
    img.save(path)

def rgb2cmyk(input_path, output_prefix):
    img = load_image(input_path)
    cmy = 1 - img / 255.
    k = np.min(cmy, axis=-1, keepdims=True)
    cmy = (cmy - k) / (1 - k)
    csum = np.sum(cmy, axis=-1, keepdims=True)
    cmy /= csum
    inv_cmy = 1 - cmy
    cmyk = np.concatenate((inv_cmy, 1-k), axis=-1)
    visualize_cmyk(cmyk)
    # Save layers
    save_image(cmyk[:,:,0], f'{output_prefix}c.png')
    save_image(cmyk[:,:,1], f'{output_prefix}m.png')
    save_image(cmyk[:,:,2], f'{output_prefix}y.png')
    save_image(cmyk[:,:,3], f'{output_prefix}k.png')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert RGB images to CMYK layers')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-p', '--prefix', default='./', help='Output path prefix')
    args = parser.parse_args()
    rgb2cmyk(args.input, args.prefix)
