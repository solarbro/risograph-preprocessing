# risograph-preprocessing
Scripts from preprocessing images for risograph prints

## K-Means 
Uses K-Means algorithm to split the input image into the desired number of color layers. It actually generates N-1 color layers, and a complimentary black layer. The predicted layer colors look pretty washed out, but the intention is to replace them with a color palette of your choice.

Run `streamlit run layer_preview.py` in a terminal to launch the web tool.

# Preview
You can use this online [RISO Print Simulator](https://risosim.dybsort.dk/) to preview what the layers will look like with different colors.

# Other resources:
Try [RGB Separation Tool](https://www.risoseparator.tools/) for a RGB to CMY (no K) separation. It gets much closer to the original colors, if you're not interested in playing around with color palattes.