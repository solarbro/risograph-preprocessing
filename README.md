# risograph-preprocessing
Scripts from preprocessing images for risograph prints

## K-Means 
Uses K-Means algorithm to split the input image into the desired number of color layers. For each layer, a weight is assigned based on the "distance" of that pixel from the layer's centroid. Provides the following arguments:

| Arg | Description |
|:--|:--|
| -k NUMBER | The number of layers. Defaults to 2, if left unspecified. |
| -a ALPHA | A scaling factor for the distance function. This allows tweaking the thresholds between layers. Defaults to 1.0, if left unspecified. |
| -i | Invert the layer weights. |
