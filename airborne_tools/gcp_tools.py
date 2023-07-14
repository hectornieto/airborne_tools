# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:13:51 2016

@author: hector
"""
import os
from pathlib import Path
from osgeo import gdal, ogr, osr
import numpy as np
import cv2
from airborne_tools import image_preprocessing as img
from scipy import stats

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

def collocate_image(master_file,
                    slave_file,
                    output_file,
                    slave_bands=None,
                    master_no_data=0,
                    slave_no_data=0,
                    dist_threshold=50,
                    pixel_threshold=20,
                    angle_threshold=90,
                    filter_intersect=True,
                    warp_threshold=0,
                    search_window_size=1000,
                    manual_gcp_file=None,
                    transform=0,
                    use_sift=True,
                    match_factor=0.75,
                    image_to_georeference=None,
                    bands_to_georeference=None):
    """
    Warps an image following a list of Ground Control Points.

    Parameters
    ----------
    master_file : str or Path object
        Path to the input GDAL compatible image that will be
        used as reference/master
    slave_file : str or Path object
        Path to the input GDAL compatible image that will be used
        for collocation with respect to the reference/master.
    output_file : str or Path object
        Path to the output collocated image.
    slave_bands : list, optional
        List of bands of `slave_file` that will be used for feature matching.
        If None all bands of `slave_file` will be used.
    master_no_data : float, optional
        NoData value for the master image.
    src_no_data : float, optional
        NoData value for the slave image.
    dist_threshold : float, optional
        Maximum translation distance, in destination georrefence units, allowed for each GCP.
        If zero this test will be omitted.
    pixel_threshold : float, optional
        Minimum distance in pixel units allowed between two GCPs
        If zero this test will be omitted.
    angle_threshold : float, optional
        Maximum angular difference (degrees) allowed from the average transformation direction.
        If zero this test will be omitted.
    filter_intersect : bool, optional
        Filter GCPs based on whether they intersect with other GCPs
        If False this test will be omitted.
    warp_threshold : float, optional
        Maximum error accepted, in georeference units, when fitting a 3-degree warp polynomial.
        Usually the objective is to have an error of half-pixel,
        so it is recommended to use one half of the slave pixel resolution
        If zero this test will be omitted.
    search_window_size : int, optional
        Feature matching will be performed in tiled-windows, within the slave image, of this size in pixels.
    manual_gcp_file : str or Path object
        Path to an ASCII table containing already (manually) assigned GCPs.
        The ASCII file must contain at least four named columns with headers ["Row, "Col", "X", "Y"].
    transform : int, default
        Transformation method to use.
        If `transform=0` a Thin Plate Spline transformation will be used.
        Use with caution unless homogeneous, high quality and dense enough GCPs are obtained,
        otherwise severe distortions might occur.
        Set a positive value instead for a polynomial transformation of order equal to such value.
    use_sift : bool
        Flag whether to use SIFT detector and descriptor [Lowe2004]_.
        If use_sift=False it will use ORB detector and descriptor [Rublee2011]_.
    match_factor : float
        Ratio test filter, as explained by [Lowe2004]_. Set a lower value for more restrictive match search,
        but fewer potential GCPs
        If match_factor=0 only best matches will be selected [CV2docs]_.
    image_to_georeference : str or Path object
        Path to the image that will be collocated.
        If None [Default] the `slave_image` will be collocated
    bands_to_georeference : list
        List of bands from the image to collocate that will be saved as georreferenced.
        If None all bands from `image_to_georeference` or `slave_image` will be used.

    Returns
    -------
    None
    """


    output_file = Path(output_file)
    outdir = output_file.parent
    # Read the master TIR mosaic
    masterfid = gdal.Open(str(master_file), gdal.GA_ReadOnly)
    master_geo = masterfid.GetGeoTransform()
    master_xsize = masterfid.RasterXSize
    master_ysize = masterfid.RasterYSize

    # Set the extent for the output image equal as the master image
    xmin, ymax = img.get_map_coordinates(0, 0, master_geo)
    xmax, ymin = img.get_map_coordinates(master_xsize, master_ysize, master_geo)
    output_extent = (xmin, ymin, xmax, ymax)

    slavefid = gdal.Open(str(slave_file), gdal.GA_ReadOnly)
    slave_cols = slavefid.RasterXSize
    slave_rows = slavefid.RasterYSize
    slave_geo = slavefid.GetGeoTransform()
    slave_prj = slavefid.GetProjection()
    if not slave_bands:
        slave_bands = range(slavefid.RasterCount)

    # Get the upper and lower bounds for each tile in the input image
    upper_rows = range(0, slave_rows, search_window_size)
    upper_cols = range(0, slave_cols, search_window_size)
    n_xwindows = len(upper_cols)
    n_ywindows = len(upper_rows)
    # Create the empty list with valid GCPs and count for valid GCPs
    gcp_valid = []
    total = 0
    distance = 0
    azimuth = 0
    intersect = 0
    proximity = 0
    # Loop all the tiles to get the GCPs
    for i, upper_row in enumerate(upper_rows):
        # print('Searching tiles in row %s,'%(upperRow))
        if i >= n_ywindows - 1:
            lower_row = slave_rows
            win_ysize = lower_row - upper_row
            # Last tile must fit the image size
        else:
            lower_row = upper_rows[i + 1]
            win_ysize = search_window_size

        for j, upper_col in enumerate(upper_cols):
            print('Searching tile row %s, col %s' % (upper_row, upper_col))
            if j >= n_xwindows - 1:
                lowerCol = slave_cols  # Last tile must fit the image size
                win_xsize = lowerCol - upper_col
            else:
                lowerCol = upper_cols[j + 1]
                win_xsize = search_window_size

            xmin, ymax = img.get_map_coordinates(upper_row, upper_col, slave_geo)
            xmax, ymin = img.get_map_coordinates(lower_row, lowerCol, slave_geo)
            # Get the pixel coordinates of the master image
            # The search window has a buffer equal to the distance threshold
            xmin -= dist_threshold
            ymin -= dist_threshold
            xmax += dist_threshold
            ymax += dist_threshold

            ul_master_row, ul_master_col = np.floor(img.get_pixel_coordinates(xmin,
                                                                              ymax,
                                                                              master_geo)
                                                    ).astype(np.int16)

            lr_master_row, lr_master_col = np.ceil(img.get_pixel_coordinates(xmax,
                                                                             ymin,
                                                                             master_geo)
                                                   ).astype(np.int16)

            # Avoid negative pixel coordinates and beyond the image extent
            ul_master_row = int(np.clip(ul_master_row, 0, master_ysize))
            ul_master_col = int(np.clip(ul_master_col, 0, master_xsize))
            lr_master_row = int(np.clip(lr_master_row, 0, master_ysize))
            lr_master_col = int(np.clip(lr_master_col, 0, master_xsize))

            win_master_xsize = int(lr_master_col - ul_master_col)
            win_master_ysize = int(lr_master_row - ul_master_row)
            # Read the master image array and subset
            master_scaled = masterfid.GetRasterBand(1).ReadAsArray(ul_master_col,
                                                                   ul_master_row,
                                                                   win_master_xsize,
                                                                   win_master_ysize).astype(float)

            if np.all(master_scaled == master_no_data):  # If all pixels have no data skip tile
                continue

            master_scaled = img.scale_grayscale_image(master_scaled, no_data=master_no_data)

            # separate into 10 x 10 blocks and get mean values for each block
            # to be used to check correlation with slave blocks (positive or negative relation)
            master_blocks = split_blocks(master_scaled, 10)

            if np.all(master_scaled == 0):
                continue

            ul_x, ul_y = img.get_map_coordinates(ul_master_row, ul_master_col, master_geo)
            # Get the master subset geotransform
            master_window_geo = (ul_x, master_geo[1], master_geo[2], ul_y, master_geo[4], master_geo[5])

            # Loop all the Principal Componets
            gcp_region = []
            for band in slave_bands:
                slave_scaled = slavefid.GetRasterBand(band + 1).ReadAsArray(upper_col, upper_row,
                                                                            win_xsize, win_ysize).astype(float)
                # check if relation is direct or inverted
                # separate into 10 x 10 blocks and get mean values for each block
                slave_blocks = split_blocks(slave_scaled, 10)
                # mask to eliminate blocks with nodata
                mask_blocks = ~np.logical_or(np.isnan(slave_blocks), np.isnan(master_blocks))

                # calculate spearman's correlation to check if correlation is positive or negative
                r_spearman = stats.spearmanr(slave_blocks[mask_blocks], master_blocks[mask_blocks])[0]
                print(f'r_cor = {r_spearman}, N = {slave_blocks[mask_blocks].size}')

                # inverse the values if expected relationship is negatively related
                # (low value features become high value features)
                if r_spearman < 0:
                    print('Negative relation found between master and slave')
                    slave_scaled *= -1

                if np.all(slave_scaled == slave_no_data):  # If all pixels have no data skip tile
                    continue

                slave_scaled = img.scale_grayscale_image(slave_scaled, no_data=slave_no_data)


                if np.all(slave_scaled == 0):
                    continue

                # Find features and matches
                gcps = find_gcps(master_scaled,
                                 slave_scaled,
                                 master_window_geo,
                                 ul_offset=(upper_row, upper_col),
                                 match_factor=match_factor,
                                 use_sift=use_sift)
                total += len(gcps)
                print(f'Found {total} valid GPCs')
                if len(gcps) > 0 and dist_threshold > 0:
                    gcps = filter_gcp_by_translation(gcps, slave_geo, dist_threshold)
                    print(f'Got {len(gcps)} valid GPCs with a translation lower than {dist_threshold}m')
                    distance += len(gcps)

                for gcp in gcps:
                    gcp_region.append(gcp)

            if len(gcp_region) > 0:
                if angle_threshold > 0:
                    # Filter GCPs based on angular deviations from the mean translation direction
                    gcp_region = filter_gcp_by_azimuth(gcp_region, slave_geo, angle_threshold)
                    print(f'Filtered {len(gcp_region)} GPCs by angular theshold')
                    azimuth += len(gcp_region)

                if pixel_threshold > 0:
                    # Filter GCPs based on image proximity
                    gcp_region = filter_gcp_by_gcp_proximity(gcp_region, pixel_threshold)
                    print(f'Got {len(gcp_region)} GCPs separated enough')
                    proximity += len(gcp_region)

                for gcp in np.asarray(gcp_region).tolist():
                    gcp_valid.append(tuple(gcp))

    n_gcps = {'total': total, 'distance': distance, 'azimuth': azimuth,
              'proximity': proximity}

    print(f"Got {proximity} candidate GCPs for the whole scene")

    if filter_intersect:
        # Filter GCPs based on whether they intersec with other GCPs
        gcp_valid = filter_gcp_by_intersection(gcp_valid, slave_geo)
        print(f'Got {len(gcp_valid)} GCPs that do not intersect')
        n_gcps["intersect"] = len(gcp_valid)

    # Remove GCPs with exactly the same map coordinates
    gcp_valid = filter_gcp_by_unique_coordinates(gcp_valid)
    print(f'Got {len(gcp_valid)} GCPs having different coordinates')
    n_gcps['coordinates'] = len(gcp_valid)

    if warp_threshold > 0:
        # Filter GCPs based on whether they intersec with other GCPs
        gcp_valid = filter_gcp_by_warp_error(gcp_valid, warp_threshold)
        print(f'Got {len(gcp_valid)} GCPs with a warp error lower than {warp_threshold}m')
        n_gcps["warp"] = len(gcp_valid)


    # Add manual GCPs
    if manual_gcp_file:
        print(f"Adding manual GCPs")
        gcp_valid = np.asarray(gcp_valid)[:, :4]
        gcp_valid = ascii_to_gcp(manual_gcp_file, gcps=gcp_valid.tolist())

    slave_xcoord, slave_yCoord = img.get_map_coordinates(np.asarray(gcp_valid)[:, 2],
                                                         np.asarray(gcp_valid)[:, 3],
                                                         slave_geo)

    if not (outdir / "GCPs").is_dir():
        (outdir / "GCPs").mkdir(parents=True)

    outshapefile = outdir / 'GCPs' / f"{output_file.name[:-4]}_Transform.shp"
    _write_transformation_vector((slave_xcoord, slave_yCoord),
                                 (np.asarray(gcp_valid)[:, 0], np.asarray(gcp_valid)[:, 1]),
                                 outshapefile,
                                 slave_prj)

    # Write the GCP to ascii file
    outtxtfile = outdir / 'GCPs' / f"{output_file.name[:-4]}_GCPs.txt"
    gcps_to_ascii(gcp_valid, outtxtfile)
    # Reproject image
    if image_to_georeference:
        slave_file = image_to_georeference

    warp_image_with_gcps(slave_file,
                         gcp_valid,
                         output_file,
                         output_extent=None,
                         src_no_data=slave_no_data,
                         transform=transform,
                         data_type=gdal.GDT_Float32,
                         subset_bands=bands_to_georeference)

def warp_image_with_gcps(input_file,
                         gcp_list,
                         outfile,
                         output_extent=None,
                         src_no_data=0,
                         subset_bands=None,
                         transform=0,
                         data_type=gdal.GDT_UInt16,
                         resolution=None):
    """
    Warps an image following a list of Ground Control Points.

    Parameters
    ----------
    input_file : str or Path object
        Path to the input GDAL compatible image that will be georeferenced/warped.
    gcp_list : list of tuples
        List of GCPs, with tuples of map and image coordinates (x, y, row, col).
    outfile : str or Path object
        Path to the output georreferenced image.
    output_extent : list or tuple, optional
        Bounds in georreferenced units (xmin, ymin, xmax, ymax).
        If None the extent will be computed by GDAL Warp.
    src_no_data : float, optional
        NoData value for the input image
    subset_bands : list, optional
        List of bands of `input_file` that will be georeferenced.
        If None all bands of `input_file` will be used.
    transform : int, default
        Transformation method to use.
        If `transform=0` a Thin Plate Spline transformation will be used.
        Use with caution unless homogeneous, high quality and dense enough GCPs are obtained,
        otherwise severe distortions might occur.
        Set a positive value instead for a polynomial transformation of order equal to such value.
    data_type : int, optional
        GDAL output data type, see more info at https://naturalatlas.github.io/node-gdal/classes/Constants%20(GDT).html
        Default: Unsigned 16bit
    resolution : tuple, optional
        Output image resolution (xres, yres) in georrefenced units.
        If None the resolution of the `input_image` will be used.

    Returns
    -------
    None
    """
    infid = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    prj = infid.GetProjection()
    if not resolution:
        geo = infid.GetGeoTransform()
        xres, yres = geo[1], geo[5]
    else:
        xres, yres = resolution

    rows = infid.RasterYSize
    cols = infid.RasterXSize
    driver = gdal.GetDriverByName('GTiff')
    tempfile = input_file.parent / 'temp.tif'
    if tempfile.exists():
        [os.remove(i) for i in Path(input_file.name.rstrip(".")[0]).glob('.*')]


    nbands = infid.RasterCount
    if not subset_bands:
        subset_bands = range(nbands)

    nbands = len(subset_bands)
    ds = driver.Create(str(tempfile), cols, rows, nbands, data_type)
    for i, band in enumerate(subset_bands):
        print('Saving Band ' + str(band))
        array = infid.GetRasterBand(band + 1).ReadAsArray()
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(array)
        band.SetNoDataValue(src_no_data)
        band.FlushCache()
        del band, array

    del infid
    gcp_list = _create_gdal_gcps(gcp_list)
    ds.SetGCPs(gcp_list, prj)
    ds.SetProjection(prj)
    ds.FlushCache()
    del ds
    # Run GDAL Warp
    if not outfile:
        outfile = input_file.name.rstrip(".")[0] + '_Georef.tif'

    if outfile.exists():
        [os.remove(i) for i in Path(input_file.name.rstrip(".")[0]).glob('.*')]

    warp_opts = {"outputBounds": output_extent, "xRes": xres, "yRes": yres,
                 "resampleAlg": "bilinear", "srcNodata": src_no_data, "dstNodata": 0,
                 "multithread": True}

    if transform == 0:
        warp_opts["tps"] = True

    else:
        warp_opts["polynomialOrder"] = transform

    gdal.Warp(str(outfile), str(tempfile), **warp_opts)
    os.remove(tempfile)


def find_gcps(master_image,
              slave_image,
              master_gt,
              ul_offset=(0, 0),
              match_factor=0.75,
              use_sift=False):
    """
    Find potential GCPs between two images by finding and matching features.

    Parameters
    ----------
    master_image : 2D-array
        Image or subset array that will be use as reference or master
    slave_image : 2D-array
        Image or subset array that will be collocated over the master
    master_gt : list or tuple
        GDAL geotransform for the master image
    ul_offset : tuple
        Offset, in pixel units, to the upper left slave image coordinate when using a subset of the slave image
    match_factor : float
        Ratio test filter, as explained by [Lowe2004]_. Set a lower value for more restrictive match search,
        but fewer potential GCPs
        If match_factor=0 only best matches will be selected [CV2docs]_.
    use_sift : bool
        Flag whether to use SIFT detector and descriptor [Lowe2004]_.
        If use_sift=False it will use ORB detector and descriptor [Rublee2011]_.

    Returns
    -------
    gcps : list of tuple
        List of GCPs, with tuples of map and image coordinates (x, y, row, col)

    References
    ----------
    ..[Lowe2004] Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints.
        International Journal of Computer Vision 60, 91–110 (2004).
        DOI: 10.1023/B:VISI.0000029664.99615.94.
    ..[CV2docs] <https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html>.
    ..[Rublee2011] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski.
        Orb: an efficient alternative to sift or surf.
        In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2564–2571. IEEE, 2011.
    """
    gcp_list = []
    # Create the feature detector/descriptor and matching objects
    if use_sift:
        # Initiate SIFT detector
        detector = cv2.SIFT_create()
        # We use NORM distance measurement for SIFT
        norm_type = cv2.NORM_L1
    else:
        # Initiate ORB detector
        detector = cv2.ORB.create()
        norm_type = cv2.NORM_HAMMING

    # Find features and their descriptors
    kp_master, des_master = detector.detectAndCompute(master_image, None)
    kp_slave, des_slave = detector.detectAndCompute(slave_image, None)
    if len(kp_master) < 2 or len(kp_slave) < 2:
        return gcp_list

    # We use Brute Force algorithm to find matches
    if match_factor > 0:
        cross_check = False
        matcher = cv2.BFMatcher(norm_type)
        # Get the 2 best matches per feature
        matches = matcher.knnMatch(des_master, des_slave, k=2)

        for i, (m, n) in enumerate(matches):
            if m.distance < match_factor * n.distance:
                master_pt = np.float32(kp_master[m.queryIdx].pt)
                x_master, y_master = img.get_map_coordinates(float(master_pt[1]),
                                                             float(master_pt[0]),
                                                             master_gt)

                slave_pt = np.float32(kp_slave[m.trainIdx].pt)
                gcp_list.append((x_master,
                                 y_master,
                                 ul_offset[0] + float(slave_pt[1]),
                                 ul_offset[1] + float(slave_pt[0])))

    else:
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = matcher.match(des_master, des_slave)
        for i, m in enumerate(matches):
            master_pt = np.float32(kp_master[m.queryIdx].pt)
            x_master, y_master = img.get_map_coordinates(float(master_pt[1]),
                                                         float(master_pt[0]),
                                                         master_gt)

            slave_pt = np.float32(kp_slave[m.trainIdx].pt)
            gcp_list.append((x_master,
                             y_master,
                             ul_offset[0] + float(slave_pt[1]),
                             ul_offset[1] + float(slave_pt[0])))

    return gcp_list


def filter_gcp_by_translation(gcps, slave_gt, dist_thres):
    """ Remove GCPs which translation distance is too large

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)
    slave_gt : tuple or list
        GDAL geotransform for the slave image
    dist_thres : float
        Maximum translation distance, in destination georrefence units, allowed for each GCP

    Returns
    -------
    gcps : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """
    gcps = np.asarray(gcps)
    if len(gcps.shape) == 1:
        gcps = gcps.reshape(1, -1)
    x_slave, y_slave = img.get_map_coordinates(gcps[:, 2], gcps[:, 3], slave_gt)
    dist = np.sqrt((x_slave - gcps[:, 0]) ** 2 + (y_slave - gcps[:, 1]) ** 2)
    gcps = gcps[dist <= dist_thres]
    return gcps.tolist()


def filter_gcp_by_gcp_proximity(gcps, pixel_thres):
    """ Remove GCPs are two proximal to other GCPs

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)
    pixel_thres : float
        Minimum distance in pixel units allowed between two GCPs

    Returns
    -------
    gcps_good : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """
    gcps_good = []
    for i, gcp_test in enumerate(gcps):
        good = True
        if i == len(gcps) - 2:
            continue
        for j in range(i + 1, len(gcps)):
            dist = np.sqrt((gcp_test[2] - gcps[j][2]) ** 2 + (gcp_test[3] - gcps[j][3]) ** 2)
            if dist < pixel_thres:  # GCPs closer to each other are discarded to avoid overfitting
                good = False
                break
        if good:
            gcps_good.append(gcp_test)

    return gcps_good


def filter_gcp_by_intersection(gcps, slave_gt):
    """ Remove GCPs that have a transformation vector direction
    significantly different to the average transformation direction

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)
    slave_gt : tuple or list
        GDAL geotransform for the slave image
    angle_thres : float
        Maximum angular difference (degrees) from the average transformation direction

    Returns
    -------
    gcps : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """
    gcps_good = []
    gcps = np.asarray(gcps)
    if len(gcps.shape) == 1:
        gcps = gcps.reshape(1, -1)

    else:
        x_slave, y_slave = img.get_map_coordinates(gcps[:, 2], gcps[:, 3], slave_gt)
        obs_vec = (gcps[:, 0] - x_slave, gcps[:, 1] - y_slave)
        azimuth = calc_azimuth(obs_vec)
        cos_azimuth = np.cos(np.radians(azimuth))
        sin_azimuth = np.sin(np.radians(azimuth))
        mean_azimuth = np.degrees(np.arctan2(np.mean(sin_azimuth), np.mean(cos_azimuth)))
        diff = np.abs(calc_azimuth_difference(azimuth, mean_azimuth))
        indices = diff.argsort()[::-1]
        for i, index in enumerate(indices):
            good = True
            if i == len(indices) - 2:
                continue
            for j in range(i + 1, len(indices)):
                intercept = _calc_vector_intersection((x_slave[index], y_slave[index]),
                                                      (gcps[index, 0], gcps[index, 1]),
                                                      (x_slave[indices[j]], y_slave[indices[j]]),
                                                      (gcps[indices[j], 0], gcps[indices[j], 1]))
                if intercept:  # GCPs closer to each other are discarded to avoid overfitting
                    good = False
                    break
            if good:
                gcps_good.append(gcps[index])

    return gcps_good


def _calc_vector_intersection(start1, end1, start2, end2):
    """
    Checks whether two segments/vectors instersect each other.
    Parameters
    ----------
    start1 : tuple
        coordinates (x, y) of starting point of first segment
    end1 : tuple
        coordinates (x, y) of ending point of first segment
    start2 : tuple
        coordinates (x, y) of starting point of second segment
    end2 : tuple
        coordinates (x, y) of ending point of second segment

    Returns
    -------
    intersect : bool
        True if the two segments intersect, False otherwise
    """
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    intersect = _ccw(start1, start2, end2) != _ccw(end1, start2, end2) \
                and _ccw(start1, end1, start2) != _ccw(start1, end1, end2)
    return intersect


def filter_gcp_by_azimuth(gcps, slave_gt, angle_thres):
    """ Remove GCPs that have a transformation vector direction
    significantly different to the average transformation direction

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)
    slave_gt : tuple or list
        GDAL geotransform for the slave image
    angle_thres : float
        Maximum angular difference (degrees) from the average transformation direction

    Returns
    -------
    gcps : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """

    gcps = np.asarray(gcps)
    if len(gcps.shape) == 1:
        gcps = gcps.reshape(1, -1)

    x_slave, y_slave = img.get_map_coordinates(gcps[:, 2], gcps[:, 3], slave_gt)
    obs_vec = (gcps[:, 0] - x_slave, gcps[:, 1] - y_slave)
    azimuth = calc_azimuth(obs_vec)
    cos_azimuth = np.cos(np.radians(azimuth))
    sin_azimuth = np.sin(np.radians(azimuth))
    # Compute the mean azimuth transformation direction
    mean_azimuth = np.degrees(np.arctan2(np.mean(sin_azimuth), np.mean(cos_azimuth)))
    # Compute the angular differences to the mean direction and filter
    diff = np.abs(calc_azimuth_difference(azimuth, mean_azimuth))
    gcps = gcps[diff <= angle_thres]
    return gcps.tolist()


def filter_gcp_by_unique_coordinates(gcps):
    """ Remove GCPs that have exactly the same destionation coordinates

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)

    Returns
    -------
    gcps : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """
    gcps = np.asarray(gcps)
    coord_good = []
    gcps_good = []
    for gcp in gcps:
        if (gcp[0], gcp[1]) not in coord_good:
            coord_good.append((gcp[0], gcp[1]))
            gcps_good.append(gcp)
    return gcps_good


def filter_gcp_by_warp_error(gcps, error_thres):
    """ Remove GCPs with errors larger than a giving threshold
    after a fitting a 3-degree polynomial transformation

    Parameters
    ----------
    gcps : list of tuple
        List of GCPs that be evaluated, with tuples of map and image coordinates (x, y, row, col)
    error_thres : float
        Maximum error accepted, in georeference units, when fitting the warp polynomial
        Usually the objective is to have an error of half-pixel,
        so it is recommended to use one half of the slave pixel resolution

    Returns
    -------
    gcps : list of tuple
        List of filtered GCPs, with tuples of map and image coordinates (x, y, row, col)
    """
    def _fit_polynomial_warp(gcps):

        gcps = np.asarray(gcps)
        if gcps.shape[0] < 15:
            return None, None
        rows = gcps[:, 2]
        cols = gcps[:, 3]
        rows2 = rows ** 2
        cols2 = cols ** 2
        rowscols = rows * cols
        rows2cols = rows ** 2 * cols
        rowscols2 = rows * cols ** 2
        rows3 = rows ** 3
        cols3 = cols ** 3

        x = np.matrix([np.ones(rows.shape), rows, cols, rowscols, rows2, cols2, rows2cols, rowscols2, rows3, cols3]).T
        map_x = gcps[:, 0].reshape(-1, 1)
        map_y = gcps[:, 1].reshape(-1, 1)
        theta_x = (x.T * x).I * x.T * map_x
        theta_y = (x.T * x).I * x.T * map_y
        return np.asarray(theta_x).reshape(-1), np.asarray(theta_y).reshape(-1)

    def _calc_warp_erors(gcps, theta_x, theta_y):
        def _polynomial_warp(rows, cols, theta_x, theta_y):
            x = theta_x[0] + theta_x[1] * rows + theta_x[2] * cols + theta_x[3] * rows * cols + theta_x[4] * rows ** 2 + \
                theta_x[5] * cols ** 2 + theta_x[6] * rows ** 2 * cols + theta_x[7] * rows * cols ** 2 + \
                theta_x[8] * rows ** 3 + theta_x[9] * cols ** 3
            y = theta_y[0] + theta_y[1] * rows + theta_y[2] * cols + theta_y[3] * rows * cols + theta_y[4] * rows ** 2 + \
                theta_y[5] * cols ** 2 + theta_y[6] * rows ** 2 * cols + theta_y[7] * rows * cols ** 2 + \
                theta_y[8] * rows ** 3 + theta_y[9] * cols ** 3
            return x, y

        gcps = np.asarray(gcps)
        if len(gcps.shape) == 1:
            gcps = gcps.reshape(1, -1)

        rows = gcps[:, 2]
        cols = gcps[:, 3]
        x_model, y_model = _polynomial_warp(rows, cols, theta_x, theta_y)
        error = np.sqrt((x_model - gcps[:, 0]) ** 2 + (y_model - gcps[:, 1]) ** 2)
        return error

    gcps = np.asarray(gcps)
    if gcps.shape[0] < 15:
        return gcps

    theta_x, theta_y = _fit_polynomial_warp(gcps)
    error = _calc_warp_erors(gcps, theta_x, theta_y)
    while np.max(error) > error_thres and len(error) > 30:
        index = error.argsort()[::-1]
        gcps = gcps[index[1:]]
        theta_x, theta_y = _fit_polynomial_warp(gcps)
        error = _calc_warp_erors(gcps, theta_x, theta_y)

    return gcps


def calc_azimuth(vector):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    vector : tuple
        Vector coordinates (x, y)
    
    Returns
    -------
    azimuth : float
        Azimutal heading (degrees from North)'''

    # Get the view azimuth angle
    azimuth = np.degrees(np.arctan2(vector[0], vector[1]))
    return azimuth


def calc_azimuth_difference(angle_1, angle_2):
    ''' Calculates the angle difference between two angles

    Parameters
    ----------   
    angle_1 : float or array
        First angle (degrees)
    angle_2 : float or array
        Second angle (degrees)
    
    Returns
    -------
    angle : Angle difference (degrees)'''

    # Get the view azimuth angle
    angle_1 = np.radians(angle_1)
    angle_2 = np.radians(angle_2)
    angle = np.degrees(np.arctan2(np.sin(angle_1 - angle_2),
                                  np.cos(angle_1 - angle_2)))
    return angle


def _create_gdal_gcps(gcps):
    "Prepares the GCPs to be stored in GDAL format"
    gdalgcp = []
    for i, gcp in enumerate(gcps):
        gdalgcp.append(gdal.GCP(gcp[0], gcp[1], 0, gcp[3], gcp[2], '', str(i)))
    return gdalgcp


def gcps_to_ascii(gcps, outfile):
    """ Saves an ASCII file with a list of GCPs.

    Parameters
    ----------
    gcps : list of tuples, optional
        List of GCPs  with tuples of map and image coordinates (x, y, row, col)
    outfile : str or Path object
        Output ASCII file, it will contain 5 columns ["ID", "X", "Y", "Row, "Col"]
        > ID: GCP id
        > X, Y: destination map coordinates
        > Row, Col: origin image coordinates

    Returns
    -------
    None
    """

    header = 'ID \t X \t Y \t Row \t Col'
    fid = open(outfile, 'w')
    fid.write(header + '\n')
    for i, gcp in enumerate(gcps):
        fid.write(f'{i}\t{gcp[0]}\t{gcp[1]}\t{gcp[2]}\t{gcp[3]}\n')
    fid.flush()
    fid.close()
    return


def ascii_to_gcp(infile, gcps=[]):
    """ Reads ASCII file with GCPs.
    Optionally adds these GCPs to an exsiting list of GCPs

    Parameters
    ----------
    infile : str or Path object
        Input ASCII file, it must contain at least 4 columns ["Row, "Col", "X", "Y"]
        > Row, Col: origin image coordinates
        > X, Y: destination map coordinates
    gcps : list of tuples, optional
        List of existing GCPs in which the GCP in the ASCII table will be appended

    Returns
    -------
    gcps : list of tuples
        List of GCPs  with tuples of map and image coordinates (x, y, row, col)
    """
    indata = np.genfromtxt(infile, names=True, dtype=None)
    print(f"Adding {len(indata.tolist())} GCPs from file")
    for data in indata:
        gcps.append([float(data['X']), float(data['Y']), float(data['Row']), float(data['Col'])])
    return gcps


def _write_transformation_vector(slave_coords, master_coords, outshapefile, prj):
    """
    Writes a shapefile with the GCP transformation vector in georreferenced units

    Parameters
    ----------
    slave_coords
    master_coords
    outshapefile
    prj

    Returns
    -------
    None
    """
    outshapefile = Path(outshapefile)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if outshapefile.exists():
        driver.DeleteDataSource(str(outshapefile))

    data_source = driver.CreateDataSource(str(outshapefile))
    # create the layer
    srs = osr.SpatialReference()
    srs.ImportFromWkt(prj)
    layer = data_source.CreateLayer("Transformation Vectors", srs, ogr.wkbLineString)

    for i, slave_xCoord in enumerate(slave_coords[0][:]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())

        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(slave_xCoord, slave_coords[1][i])
        line.AddPoint(master_coords[0][i], master_coords[1][i])
        # Set the feature geometry using the point
        feature.SetGeometry(line)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()
    # Destroy the data source to free resources
    data_source.Destroy()
    return

def split_blocks(array, nblocks):
    """
    split 2D array into equal blocks in both vertical and horizontal direction
    and calculates mean for each block

    Parameters
    ----------
    input array: 2D numpy array

    nblocks: int
        number of blocks to divide array

    Returns
    -------
    block_means: numpy array
        array with mean values for each block

    """
    # function to split array into equal blocks in both vertical and horizontal direction
    # and calculate mean for each block
    block_means = []
    # split array into equal blocks in axis 0 (i.e. in rows horizontally)
    ar_split1 = np.array_split(array, nblocks)
    for ar in ar_split1:
        # further split array into equal blocks in axis 1 (i.e. in cols vertically)
        ar_split2 = np.array_split(ar, nblocks, axis=1)
        for ar_sub in ar_split2:
            ar_mean = np.nanmean(ar_sub)
            block_means.append(ar_mean)

    return np.array(block_means)
