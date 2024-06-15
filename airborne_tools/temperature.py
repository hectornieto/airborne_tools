from osgeo import gdal
import numpy as np
from scipy import stats as st


def get_component_temperatures(lst, vnir, vnir_soil, vnir_veg, f_res):
    """ Applies a contextual LST-VI method to derive canopy and soil
    temperatures from very high resolution imagery

    Parameters
    ----------
    lst : array_like
        Radiometric surface temperature input of size (nrows, ncols)
    vnir : array_like
        Radiometric spectral band of size (nrows, ncols)
        It can be any spectral band or a spectral index
    vnir_soil : float
        Lower threshold for bare soil pixels
    vnir_veg : float
        Upper threshold for pure vegetation pixels
    f_res : tuple of int
        Resampling factor (rows, cols) at which `t_leaf` and `t_soils`
        will be retrieved

    Returns
    -------
    t_leaf : array_like
        Leaf temperature (nrows/f_res, ncols/f_res)
    t_soil : array_like
        Soil temperature (nrows/f_res, ncols/f_res)
    cor : array_like
        Local correlation between lst and vnir inputs (nrows/f_res, ncols/f_res)

    References
    ----------
    .. [Nieto2019] Nieto, H., Kustas, W.P., Torres-Rúa, A. et al. (2019)
          Evaluation of TSEB turbulent fluxes using different methods for the
          retrieval of soil and canopy component temperatures from UAV thermal
          and multispectral imagery.
          Irrigation Science 37, 389–406.
          https://doi.org/10.1007/s00271-018-0585-9
    """
    # Get the hr size
    nrows, ncols = lst.shape

    # Initialize the outputs
    out_shape = int(np.ceil(nrows / f_res[0])), int(np.ceil(ncols / f_res[1]))
    t_leaf = np.full(out_shape, np.nan)
    t_soil = np.full(out_shape, np.nan)
    cor = np.full(out_shape, np.nan)

    # Loop along all tiles
    row_ini = 0
    i = 0
    print(f"Processing {out_shape[0] * out_shape[1]} tiles:")
    while row_ini < nrows - 1:
        row_end = np.minimum(row_ini + f_res[0], nrows)
        col_ini = 0
        j = 0
        while col_ini < ncols - 1:
            # print((i, j), end=", ")
            col_end = np.minimum(col_ini + f_res[1], ncols)
            lst_subset = lst[row_ini:row_end, col_ini:col_end]
            if ~np.any(np.isfinite(lst_subset)):
                # Jump to the next tile
                j += 1
                col_ini = float(col_end)
                continue

            vnir_subset = vnir[row_ini:row_end, col_ini:col_end]

            soils = vnir_subset <= vnir_soil
            vegs = vnir_subset >= vnir_veg
            reg = st.linregress(np.ravel(vnir_subset), np.ravel(lst_subset))

            cor[i, j] = reg.rvalue
            if np.any(soils):
                t_soil[i, j] = np.nanmean(lst_subset[soils])
            else:
                t_soil[i, j] = reg.intercept + reg.slope * vnir_soil

            if np.any(vegs):
                t_leaf[i, j] = np.nanmean(lst_subset[vegs])
            else:
                t_leaf[i, j] = reg.intercept + reg.slope * vnir_veg

            # Jump to the next tile
            j += 1
            col_ini = int(col_end)

        # Jump to the next tile
        i += 1
        row_ini = int(row_end)


    return t_leaf, t_soil, cor


def component_temperatures(lst_file, vnir_file, vnir_soil, vnir_veg, out_res,
                           output_file):
    """ Applies a contextual LST-VI method to derive canopy and soil
    temperatures from very high resolution imagery

    Parameters
    ----------
    lst_file : str or Path object
        Input Radiometric surface temperature file
    vnir_file : str or Path object
        Input Radiometric spectral file.
        It can be any spectral band or a spectral index
    vnir_soil : float
        Lower threshold for bare soil pixels
    vnir_veg : float
        Upper threshold for pure vegetation pixels
    out_res : float
        Output resolution at which `t_leaf` and `t_soils` will be retrieved
    output_file : str or Path object
        Output component temperatures file. It will have three bands:
        1. Leaf Temperature
        2. Soil Temperature
        3. Local correlation between lst and vnir inputs

    References
    ----------
    .. [Nieto2019] Nieto, H., Kustas, W.P., Torres-Rúa, A. et al. (2019)
          Evaluation of TSEB turbulent fluxes using different methods for the
          retrieval of soil and canopy component temperatures from UAV thermal
          and multispectral imagery.
          Irrigation Science 37, 389–406.
          https://doi.org/10.1007/s00271-018-0585-9
    """

    print("Read input lst file")
    fid = gdal.Open(str(lst_file), gdal.GA_ReadOnly)
    gt = fid.GetGeoTransform()
    proj = fid.GetProjection()
    lst = fid.GetRasterBand(1).ReadAsArray()
    na_value = fid.GetRasterBand(1).GetNoDataValue()
    lst[lst == na_value] = np.nan
    dims = lst.shape
    extent = [gt[0], gt[3] + gt[5] * dims[0],
              gt[0] + gt[1] * dims[1], gt[3]]

    print("Calculate the rescaling factor")
    f_res = (int(np.round(np.abs(out_res / gt[5]))),
             int(np.round(np.abs(out_res / gt[1]))))
    gt_out = [gt[0], out_res, 0, gt[3], 0, -out_res]

    print("Read input vnir file")
    fid = gdal.Open(str(vnir_file), gdal.GA_ReadOnly)
    gt_vnir = fid.GetGeoTransform()
    vnir_shape = fid.RasterYSize, fid.RasterXSize
    # Resample VNIR if dimension do not match
    if gt != gt_vnir or vnir_shape != dims:
        print(f"Resample {vnir_file} with GDAL warp to match {lst_file}")
        fid = gdal.Warp("",
                        fid,
                        format="MEM",
                        dstSRS=proj,
                        xRes=gt[1],
                        yRes=gt[5],
                        outputBounds=extent,
                        resampleAlg="average",
                        srcNodata=fid.GetRasterBand(1).GetNoDataValue(),
                        dstNodata=np.nan)

    vnir = fid.GetRasterBand(1).ReadAsArray()
    na_value = fid.GetRasterBand(1).GetNoDataValue()
    vnir[vnir == na_value] = np.nan
    del fid
    print("Extract the component temperatures")
    t_leaf, t_soil, cor = get_component_temperatures(lst,
                                                     vnir,
                                                     vnir_soil,
                                                     vnir_veg,
                                                     f_res)

    print(f"Saving temperatures to {output_file}")
    driver = gdal.GetDriverByName("GTiff")
    fid = driver.Create(str(output_file), t_leaf.shape[1], t_leaf.shape[0], 3,
                        gdal.GDT_Float32)
    fid.SetProjection(proj)
    fid.SetGeoTransform(gt_out)
    for i, array in enumerate([t_leaf, t_soil, cor]):
        fid.GetRasterBand(i + 1).WriteArray(array)

    del fid
    print("Done")


