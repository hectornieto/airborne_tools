import numpy as np
import cv2
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import json

WGS84_A = 6378137.0
WGS84_B = 6356752.314245

def ecef_from_lla(lat, lon, alt):
    """
    Compute ECEF XYZ from latitude, longitude and altitude.

    All using the WGS84 model.
    Altitude is the distance to the WGS84 ellipsoid.

    """
    a2 = WGS84_A ** 2
    b2 = WGS84_B ** 2
    lat = np.radians(lat)
    lon = np.radians(lon)
    l = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * l + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * l + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * l + alt) * np.sin(lat)
    return np.array([x, y, z])


def coordinates_collinearity(pixel, c_rc, altitude, origin, rotation_matrix,
                             focal_length):
    '''Calculates the projected coordinates of an image pixel by applying the 
    collinearity equation
    
    Parameters
    ----------
    pixel : pixel coordinates (row, col) with the origin at the top left corner
        of the image with x axis pointing right and y axis pointing down
    c_rc: pixel coordinates (row, col) of the principal point
    altitude: Pixel ground elevation, Z (m)
    origin: the coordinates (X0, Y0, Z0) of the center of projection of the camera
    rotation_matrix : Rotation matrix (Omega,Phi,Kappa)
    focal_length : Camera focal length (pixels)

    Returns
    -------
    X, Y : pixel coordinates (X Y) at ground elevation (altitude)'''

    z_z0 = altitude - float(origin[2])
    v = pixel[1] - c_rc[1]
    u = c_rc[0] - pixel[0]
    denom = (rotation_matrix[2, 0] * u + rotation_matrix[2, 1] * v
             - rotation_matrix[2, 2] * focal_length)
    x = origin[0] + z_z0 * (rotation_matrix[0, 0] * u + rotation_matrix[0, 1] * v
                            - rotation_matrix[0, 2] * focal_length) / denom

    y = origin[1] + z_z0 * (rotation_matrix[1, 0] * u + rotation_matrix[1, 1] * v
                            - rotation_matrix[1, 2] * focal_length) / denom

    return x, y


def get_altitude(x, y, dsm, geo_dsm, default_z=0):
    ''' Get the elevation for a given coordinate from a Digital Surface Model

    Parameters
    ----------
    x, y : Cordinates of the point of interest in the map projection system
    dsm : Array with the elevation Fields (m)
    geo_dsm : gdal GeoTransForm model
    default_z : default value in case the X,Y coordinates fall outside the DSM
        extent
    
    Returns
    -------
    altitude: Ground elevation at X Y coodinates
    out: Boolean variable indicating whether the coordinates fell outside the
        DSM boundaries'''

    out = False
    rows_dsm, cols_dsm = dsm.shape
    col_dsm = int((x - geo_dsm[0]) / geo_dsm[1])
    row_dsm = int((y - geo_dsm[3]) / geo_dsm[5])
    if col_dsm >= 0 and col_dsm < cols_dsm and row_dsm >= 0 and row_dsm < rows_dsm:
        altitude = float(dsm[row_dsm, col_dsm])
        if altitude <= 0:
            altitude = default_z
            out = True
    else:
        altitude = default_z
        out = True
    return altitude, out


def get_altitude_dtm_array(x_array, y_array, dsm, geo_dsm, default_z=0):
    ''' Get the elevation for a given coordinate from a Digital Surface Model

    Parameters
    ----------
    X, Y : Cordinates of the point of interest in the map projection system
    dsm : Array with the elevation Fields (m)
    geo_dsm : gdal GeoTransForm model
    default_z : default value in case the X,Y coordinates fall outside the DSM
        extent
    
    Returns
    -------
    altitude: Ground elevation at X Y coodinates
    out: Boolean variable indicating whether the coordinates fell outside the
        DSM boundaries'''

    rows_dsm, cols_dsm = np.shape(dsm)
    altitude = np.zeros(x_array.shape) + default_z
    col_dsm = np.floor((x_array - geo_dsm[0]) / geo_dsm[1]).astype(np.int32)
    row_dsm = np.floor((y_array - geo_dsm[3]) / geo_dsm[5]).astype(np.int32)
    valid = np.logical_and.reduce((col_dsm >= 0,
                                   col_dsm < cols_dsm,
                                   row_dsm >= 0,
                                   row_dsm < rows_dsm))
    altitude[valid] = dsm[row_dsm[valid], col_dsm[valid]]
    return altitude


def observation_geometry(coordinates, origin):
    ''' Estimates the Observation Geometry based on the pixel coordinates 
        and camera coordinates
    
    Parameters
    ---------
    coordinates : Xi,Yi,Zi coordinates of observed pixel
    origin : X0,Y0,Z0 coordinates of the camera (=nadir)
    
    Returns
    -------
    vza : View Zenith Angle (degrees)
    vza : View Azimuth Angle (degrees)'''

    # Get the view zenith angle
    ref_vec = np.array([0, 0, -1])
    coordinates = np.asarray(coordinates)
    obs_vec = np.asarray([coordinates[0] - origin[0],
                          coordinates[1] - origin[1],
                          coordinates[2] - origin[2]])
    cos_theta = obs_vec[2] / (ref_vec[2] *
                              np.sqrt(obs_vec[0] ** 2
                                      + obs_vec[1] ** 2
                                      + obs_vec[2] ** 2))
    vza = np.degrees(np.arccos(cos_theta))
    # Get the view azimuth angle
    vaa = np.degrees(np.arctan2((obs_vec[0], obs_vec[1])))
    return vza, vaa


def navigation_azimuth(lon_0, lat_0, lon_1, lat_1):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''

    x = lon_1 - lon_0
    y = lat_1 - lat_0
    azimuth = y / np.sqrt(x ** 2 + y ** 2)
    azimuth = np.degrees(np.arccos(azimuth))
    return azimuth


def correct_distortion(pixel, c_xy, f_xy, k_array, t_array=[0, 0], skew=0):
    '''Corrects the lens distortion based on Photoscan pinhole camera 
        for a pixel coordinate
    
    Parameters
    ----------
    pixel : Uncorrected pixel coordinate (col,row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down
    c_xy: image princpal point (col,row)
    f_xy:focal lenght (x.y) in pixels
    k_array :  Brown's radial distortion parameters [k1,k2,k3,k4]
    t_array : tangential distortion parameters [P1, P2]
        (default=[0,0], no tangential distortion)
    skew : skewness coefficient (default=0, no skew)
    
    Returns
    -------
    u,v : Corrected pixel coordinates (col, row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down'''

    from math import sqrt
    x = (pixel[0] - c_xy[0]) / f_xy[0]
    y = (pixel[1] - c_xy[1]) / f_xy[1]
    r = sqrt(x ** 2 + y ** 2)
    x_u = x * (1.0 + k_array[0] * r ** 2 + k_array[1] * r ** 4
               + k_array[2] * r ** 6 + k_array[3] * r ** 8) \
          + t_array[1] * (r ** 2 + 2.0 * x ** 2) + 2.0 * t_array[0] * x * y
    y_u = y * (1.0 + k_array[0] * r ** 2 + k_array[1] * r ** 4
               + k_array[2] * r ** 6 + k_array[3] * r ** 8) \
          + t_array[0] * (r ** 2 + 2.0 * y ** 2) + 2.0 * t_array[1] * x * y
    u = c_xy[0] + x_u * f_xy[0] + skew * y_u
    v = c_xy[1] + y_u * f_xy[1]
    return u, v


def undistort(img, c_xy, f_xy, K, T=[0, 0], skew=0):
    '''Corrects the lens distortion based on Photoscan pinhole camera 
        for a pixel coordinate
    
    Parameters
    ----------
    img : Uncorrected image
    c_xy: image princpal point (col,row)
    f_xy:focal lenght (x.y) in pixels
    K :  Brown's radial distortion parameters [k1,k2,k3,k4]
    T : tangential distortion parameters [P1, P2] 
        (default=[0,0], no tangential distortion)
    skew : skewness coefficient (default=0, no skew)
    
    Returns
    -------
    u,v : Corrected pixel coordinates (col, row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down'''

    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    # Calculate undistorted pixel coordinates
    x = (xx - c_xy[0]) / f_xy[0]
    y = (yy - c_xy[1]) / f_xy[1]
    r = np.sqrt(x ** 2 + y ** 2)
    x_u = x * (1.0 + K[0] * r ** 2 + K[1] * r ** 4 + K[2] * r ** 6 + K[3] * r ** 8) \
          + T[1] * (r ** 2 + 2.0 * x ** 2) + 2.0 * T[0] * x * y
    y_u = y * (1.0 + K[0] * r ** 2 + K[1] * r ** 4 + K[2] * r ** 6 + K[3] * r ** 8) \
          + T[0] * (r ** 2 + 2.0 * y ** 2) + 2.0 * T[1] * x * y
    u = c_xy[0] + x_u * f_xy[0] + skew * y_u
    v = c_xy[1] + y_u * f_xy[1]
    # u_remap=img.shape[1]*(u-u_min)/(u_max-u_min)
    # v_remap=img.shape[0]*(v-v_min)/(v_max-v_min)
    # Assign image values to the nearest image value
    u_map = u.astype('float32')
    v_map = v.astype('float32')
    dst = cv2.remap(img, u_map, v_map, cv2.INTER_LINEAR)
    # dst=np.zeros(img.shape)
    # pixels=[(row,col) for row in range(img.shape[0]) for col in range(img.shape[1])]
    # for pixel in pixels:
    # dst[int(v[pixel]),int(u[pixel]),:]=img[pixel[0],pixel[1],:]
    return dst


def pixel_collinearity(coordinates, c_rc, origin, rotation_matrix, focal_length):
    '''Calculates the pixel coordinates for a map projected coordinates by applying the 
    collinearity equation
    
    Parameters
    ----------
    coordinates : map projected coordinates (X,Y,Z) 
    c_rc: pixel coordinates (row, col) of the principal point
    origin: the coordinates (X0,Y0,Z0) of the center of projection of the camera
    rotation_matrix : Rotation matrix (Omega,Phi,Kappa)
    focal_length : Camera focal length (pixels)

    Returns
    -------
    u, v : pixel coordinates (col, row) with the origin at the top left corner
        of the image with x axis pointing right and y axis pointing down '''

    x_x0 = coordinates[0] - origin[0]
    y_y0 = coordinates[1] - origin[1]
    z_z0 = coordinates[2] - origin[2]
    denom = (rotation_matrix[0, 2] * x_x0 + rotation_matrix[1, 2]
             * y_y0 + rotation_matrix[2, 2] * z_z0)
    x = -focal_length * (rotation_matrix[0, 0] * x_x0 + rotation_matrix[1, 0]
                         * y_y0 + rotation_matrix[2, 0] * z_z0) / denom
    y = -focal_length * (rotation_matrix[0, 1] * x_x0 + rotation_matrix[1, 1]
                         * y_y0 + rotation_matrix[2, 1] * z_z0) / denom

    row = c_rc[0] - y
    col = x + c_rc[1]

    return row, col


def get_ypr(rotation):
    '''Calculate the rotation angles from a rotation matrix (Rz(kappa),Ry(pitch),Rx(roll))

    Parameter
    --------
    rotation : 3D rotation matrix
    
    Returns
    -------
    yaw, pitch, roll : photogrametric angles'''
    yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    pitch = np.degrees(np.arcsin(-rotation[2, 0]))
    roll = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))

    return yaw, pitch, roll


def get_opk(rotation):
    '''Calculate the rotation angles from a rotation matrix (Rz(kappa),Ry(pitch),Rx(roll))

    Parameter
    --------
    rotation : 3D rotation matrix

    Returns
    -------
    Omega,phi,kappa : photogrametric angles'''
    omega = np.degrees(np.arctan2(-rotation[1, 2], rotation[2, 2]))
    phi = np.degrees(np.arcsin(rotation[0, 2]))
    kappa = np.degrees(np.arctan2(-rotation[0, 1], rotation[0, 0]))
    return omega, phi, kappa


def rot_x(alpha):
    '''Rotation over x axis
    
    Parameter
    ---------
    alpha : rotation angle over X
    
    Returns
    -------
    r_y: rotation matrix over X'''

    alpha = np.radians(alpha)
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
    return r_x


def rot_y(alpha):
    '''Rotation over Y axis
    
    Parameter
    ---------
    alpha : rotation angle over Y
    
    Returns
    -------
    r_y: rotation matrix over Y'''

    alpha = np.radians(alpha)
    r_y = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                    [0, 1, 0],
                    [-np.sin(alpha), 0, np.cos(alpha)]])
    return r_y


def rot_z(alpha):
    '''Rotation over Z axis
    
    Parameter
    ---------
    alpha : rotation angle over Z
    
    Returns
    -------
    r_z: rotation matrix over Z'''

    alpha = np.radians(alpha)
    r_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
    return r_z


def rotation_ypr(yaw, pitch, roll):
    r_y = rot_z(yaw)
    r_p = rot_y(pitch)
    r_r = rot_x(roll)
    cnb = np.linalg.multi_dot([r_y, r_p, r_r])

    return cnb


def rotation_opk(omega, phi, kappa):
    r_w = rot_x(omega)
    r_p = rot_y(phi)
    r_k = rot_z(kappa)
    ceb = np.linalg.multi_dot([r_w, r_p, r_k])
    return ceb


def rotation_cen(lat, lon, alt, delta=1e-7):

    p1 = ecef_from_lla(lat + delta, lon, alt)
    p2 = ecef_from_lla(lat - delta, lon, alt)
    xnp = p1 - p2
    m = np.linalg.norm(xnp)
    # Unit vector pointing north
    xnp /= m
    znp = np.array([0, 0, -1])
    ynp = np.cross(znp, xnp)
    cen = np.array([xnp, ynp, znp]).T
    return cen


def ypr_2_opk(lat, lon, alt, yaw, pitch, roll, orientation=1):
    # YPR rotation matrix
    cnb = rotation_ypr(yaw, pitch, roll)
    # Convert between image and body coordinates
    # (Swap X/Y, flip Z)
    if orientation == 1:
        cbb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    elif orientation == 3:
        cbb = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

    cen = rotation_cen(lat, lon, alt)

    # OPK rotation matrix
    ceb = np.linalg.multi_dot([cen, cnb, cbb])
    omega, phi, kappa = get_opk(ceb)
    return omega, phi, kappa


def opk_from_opensfm(exif_json):
    with open(exif_json, "r") as fid:
        data = json.load(fid)
    omega = data["opk"]["omega"]
    phi = data["opk"]["phi"]
    kappa = data["opk"]["kappa"]

    return omega, phi, kappa

def create_footprint(lat, lon, alt, yaw, pitch, roll, focal_length, dims, z_ground,
                     orientation=1):
    # YPR rotation matrix
    omega, phi, kappa = ypr_2_opk(lat, lon, alt, yaw, pitch, roll, orientation=orientation)
    ceb = rotation_opk(omega, phi, kappa)
    utm_crs = query_utm_crs_info(datum_name="WGS 84",
                                 area_of_interest=AreaOfInterest(west_lon_degree=lon,
                                                                 south_lat_degree=lat,
                                                                 east_lon_degree=lon,
                                                                 north_lat_degree=lat))
    utm_crs = CRS.from_epsg(utm_crs[0].code)
    proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs, always_xy=True)
    x_0, y_0 = proj.transform(lon, lat)

    c_rc = 0.5 * dims[0], 0.5 * dims[1]
    ul = coordinates_collinearity((0., 0.), c_rc, z_ground, (x_0, y_0, alt), ceb,
                                  focal_length)

    ur = coordinates_collinearity((0., dims[1]), c_rc, z_ground, (x_0, y_0, alt), ceb,
                                  focal_length)

    lr = coordinates_collinearity(dims, c_rc, z_ground, (x_0, y_0, alt), ceb,
                                  focal_length)

    ll = coordinates_collinearity((dims[0], 0.), c_rc, z_ground, (x_0, y_0, alt), ceb,
                                  focal_length)

    return ul, ur, lr, ll, utm_crs


def footprint_2_geojson(ul, ur, lr, ll, crs, out_geojson=None, properties={}):
    feature = {"type": "Feature",
               "geometry": {"type": "Polygon",
                            "coordinates": [[ul, ur, lr, ll]]
                            },
               "properties": properties
               }

    json_dict = {"type": "FeatureCollection",
                 "crs": {"type": "name",
                         "properties": {"name": f"EPSG:{crs.to_epsg()}"}},
                 "features": [feature]}

    if out_geojson:
        out = json.dumps(json_dict,
                         indent=4,
                         separators=(',', ': '))
        with open(out_geojson, "w") as fid:
            fid.write(out)
            fid.flush()

    return json_dict

