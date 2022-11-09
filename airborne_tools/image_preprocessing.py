import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from osgeo import gdal
import cv2

MIN_MAX_STRETCH = 0

def pca(image_file,
        no_data=0,
        use_bands=None,
        pca_components=None,
        outfile=None,
        normalize=False):

    driver = gdal.GetDriverByName('GTiff')

    fid = gdal.Open(str(image_file), gdal.GA_ReadOnly)
    if use_bands:
        bands = list(use_bands)
        nbands = len(bands)
    else:
        nbands = fid.RasterCount
        bands = range(nbands)

    if not pca_components:
        pca_components = nbands

    cols = fid.RasterXSize
    rows = fid.RasterYSize
    hyper_geo = fid.GetGeoTransform()
    hyper_prj = fid.GetProjection()
    input_array = np.full((rows * cols, nbands), np.nan)
    for i, band in enumerate(bands):
        print('Reading band %s' % band)
        array = fid.GetRasterBand(band + 1).ReadAsArray().astype(float)
        mask = array != no_data
        if normalize:
            scaler_input = StandardScaler()
            scaler_input.fit(array[mask].reshape(-1, 1))
            input_array[:, i] = scaler_input.transform(array.reshape(-1, 1)).reshape(-1)
            input_array[:, i] *= mask.reshape(-1)
        else:
            input_array[:, i] = array.reshape(-1)
        del array
    del fid
    pca = PCA(n_components=pca_components)
    input_array = np.ma.masked_array(input_array, mask=input_array==no_data)
    pca.fit(input_array)
    print(f'Explained variance per component: ' + ", ".join(np.round(pca.explained_variance_ratio_, 2).astype(str)))
    print(f'Explained variance total: {np.sum(np.asarray(pca.explained_variance_ratio_)):4.2f}')
    output_array = pca.transform(input_array)
    output_array = output_array.reshape((rows, cols, pca_components))
    output_array[~mask] = np.nan
    if outfile:
        ds = driver.Create(str(outfile), cols, rows, pca_components, gdal.GDT_Float32)
        ds.SetGeoTransform(hyper_geo)
        ds.SetProjection(hyper_prj)
        for band in range(pca_components):
            ds.GetRasterBand(band + 1).WriteArray(output_array[:, :, band])
            ds.FlushCache()
        del ds

    return output_array, pca.explained_variance_ratio_


def scale_grayscale_image(image, no_data=None, stretch=MIN_MAX_STRETCH):
    if isinstance(no_data, type(None)):
        index = np.logical_and(image != no_data, np.isfinite(image))
    else:
        index = np.ones(image.shape, dtype=bool)

    if stretch == 0:
        min_val = np.nanmin(image[index])
        max_val = np.nanmax(image[index])
    elif stretch > 0:
        min_val = np.nanpercentile(image[index], stretch)
        max_val = np.nanpercentile(image[index], 1 - stretch)
    else:
        mean = np.nanmean(image[index])
        std = np.nanstd(image[index])
        min_val = mean + stretch * std
        max_val = mean - stretch * std

    image[~index] = 0
    if np.sum(index) > 30:
        image[index] = (2**8 - 1) * ((image[index] - min_val) / (max_val - min_val))
        image = cv2.equalizeHist(image.astype(np.uint8))
    else:
        image *= 0
    return image


def get_map_coordinates(row, col, gt):
    x = gt[0] + gt[1] * col + gt[2] * row
    y = gt[3] + gt[4] * col + gt[5] * row
    return x, y


def get_pixel_coordinates(x, y, gt):
    row = (y - gt[3]) / gt[5]
    col = (x - gt[0]) / gt[1]
    return row, col
