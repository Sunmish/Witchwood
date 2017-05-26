# Witchwood
A simple source-finding script for use on FITS images with radio data. The intended use is for measuring the flux densities of extended radio sources.  The code uses a naive flood-fill algorithm checked against a reference array so that it only makes one pass (against the actual array, and the reference array) over each array. Sources are considered as clustering of pixels, and their flux densities are measured. The code is still in progress, but at present works well to measure the flux density of all sources in NVSS, TGSS, SUMSS, GLEAM, and VLSSr images.

Source-finding and measuring can be done to find a single source by providing a coordinate pair to measure_tree or by simply finding all sources in an image with measure_forest.  Largest angular scale (LAS) calculations are made if selected, by measuring the greatest angular distance between any two boundary pixels of a detected source. At present this LAS measuring only works with FITS images that have NAXIS = 2, but could be changed to allow NAXIS = 3/4 by adding a few lines to incorporate those extra axes.


Witchwood will be merged into AstroTools as part of ``fluxtools``. Most the functionality will be the same, with the possibility of additional command-line compatible scripts to use some functions standalone. 
