from rgb2kmeans import rgb2kmeans, Options, Mode, save_image
import PIL
import numpy as np
import streamlit as st
import math
from skimage import color

def resize_image(img: PIL.Image, size: int) -> PIL.Image:
    w, h = img.size
    scale = size / max(w, h)
    if scale >= 1.0:
        return img # Already smaller than preview
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, PIL.Image.BILINEAR)

def pil_to_np(img: PIL.Image) -> np.ndarray:
    np_img = np.array(img.getdata()).reshape(img.size[1], img.size[0], len(img.mode))
    if np_img.shape[-1] > 3:
        np_img = np_img[:,:,:-1]
    np_img = (np_img / 255.).astype(np.float32)
    return np_img

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    st.title('Riso Layer Generator')
    initial_k = st.sidebar.slider("Initial K", 2, 128, 32)
    final_k = st.sidebar.slider("Final K", 1, 16, 4)
    max_blend = st.sidebar.slider("Max Overlapped Layers", 1, 8, 2)
    winner_margin = st.sidebar.slider('Winner Margin', 0.0, 1.0, 0.5, 0.01)
    gamma = st.sidebar.slider('Gamma', 0.0, 10.0, 4.0, 0.01)
    chroma = st.sidebar.slider('Chroma Cutoff', 1.0, 100.0, 20.0, 0.1)
    uploaded = st.file_uploader('Image', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded is None:
        st.stop()
    src_image = PIL.Image.open(uploaded)
    src_preview = resize_image(src_image, 512)
    src_preview = pil_to_np(src_preview)
    options = Options(final_k, Mode.Gradient, max_blend, gamma, False) # TODO: invert option
    options.num_initial_k = initial_k
    options.winner_threshold = winner_margin
    options.chroma_cutoff = chroma
    layers, centroids = rgb2kmeans(src_preview, options)
    st.image(src_preview, caption="Original")
    # Display layers in grid
    white_lab = np.array([100., 0.01, -0.01])
    num_cols = math.ceil(math.sqrt(final_k))
    cols = st.columns(num_cols)
    for i, layer in enumerate(layers):
        with cols[i % num_cols]:
            preview = (1 - layers[i]) * centroids[i] + layers[i] * white_lab
            preview_rgb = color.lab2rgb(preview)
            st.image(preview_rgb, caption=f'Layer {i}')
    # TODO: Composite image
    prefix = st.text_input('Export prefix', './')
    if st.button('Export'):
        full_res = pil_to_np(src_image)
        layers, centroids = rgb2kmeans(full_res, options)
        for i in range(final_k):
            save_image(layers[i], f'{prefix}layer_{i + 1}.png')