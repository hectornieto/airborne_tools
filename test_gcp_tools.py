from pathlib import Path
import numpy as np
from airborne_tools import image_preprocessing as img
from airborne_tools import gcp_tools as gcp

transform = 0  # Set 0 for thin plate spline transformation,
# not recommended unless homogeneous, high quality and dense enough GCPs are obtained,
# otherwise severe distortions might occur.
# Set a positive value instead for a polynomial transformation of order equal to such value

match_factor = 0.80  # matching factor between keypoint descriptors,
# set a lower value for more restrictive match search, but fewer potential GCPs
# Set to 0 to get the best matches in two images instead of pair of best matches

search_window_size = 250  # GCPs search and some filtering is done at smaller windows, to ensure distribution
# of GCPs and comply with the assumption of several GCPs filters

dist_threshold = 10  # Expected maximum translation/error, in georreference units, of the mosaic
warp_threshold = 0.025  # Maximum error accepted, in georeference units, when fitting the warp polynomial
# Usually the objective is to have an errof of half-pixel,
# so use one half of the slave pixel resolution
# current work directory
workdir = Path()
test_dir = workdir / 'test'

tir_image = test_dir / 'tir_odm_20220916.tif'
vnir_image = test_dir / 'Sequoia_vnir_20220916.tif'
master_image = test_dir / 'Sequoia_vnir_20220916_PC1.tif'
collocated_image = test_dir / 'tir_odm_20220916_collocated.tif'

# optional, inclue a list of GCPs that were selected manually, otherwise set to None
manual_gcp_file = None

if not master_image.exists():
    pass
    # We need to reduce the dimensionality of the master image to a single grayscale band.
    # We therefore apply a PCA reduction to get a grayscale image combining all spectral bands
    img.pca(vnir_image,
            no_data=4294967296,
            use_bands=[0, 1, 2, 3],
            pca_components=1,
            outfile=master_image,
            normalize=True)

gcp.collocate_image(master_image,
                    tir_image,
                    collocated_image,
                    slave_bands=[0],
                    master_no_data=np.nan,
                    slave_no_data=65535,
                    dist_threshold=dist_threshold,
                    pixel_threshold=5,
                    angle_threshold=45,
                    warp_threshold=warp_threshold,
                    filter_intersect=True,
                    search_window_size=search_window_size,
                    manual_gcp_file=manual_gcp_file,
                    transform=transform,
                    use_sift=True,
                    match_factor=match_factor,
                    image_to_georeference=None,
                    bands_to_georeference=[0])

