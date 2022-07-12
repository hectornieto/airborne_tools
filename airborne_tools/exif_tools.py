from pyexiv2 import Image

def get_raw_metadata(image):
    fid = Image(image)
    exif = fid.read_exif()
    xmp = fid.read_xmp()
    fid.close()
    return exif, xmp


def update_metadata(image, exif=None, xmp=None):
    fid = Image(image)
    if exif:
        fid.modify_exif(exif)
    if xmp:
        fid.modify_xmp(xmp)
    fid.close()


def get_exif_value(value):
    value = value.split(" ")
    values = []
    for v in value:
        v = float(v.split("/")[0]) / float(v.split("/")[1])
        values.append(v)
    if len(values) == 1:
        values = float(values[0])
    return values


def dms_to_dd(gps_coords, gps_coords_ref):
    d, m, s = get_exif_value(gps_coords)
    dd = d + m / 60 + s / 3600
    if gps_coords_ref.upper() in ('S', 'W'):
        return -dd
    elif gps_coords_ref.upper() in ('N', 'E'):
        return dd


def coordinates_from_exif(exif):
    lat = dms_to_dd(exif["Exif.GPSInfo.GPSLatitude"],
                    exif["Exif.GPSInfo.GPSLatitudeRef"])
    lon = dms_to_dd(exif["Exif.GPSInfo.GPSLongitude"],
                    exif["Exif.GPSInfo.GPSLongitudeRef"])
    alt = get_exif_value(exif["Exif.GPSInfo.GPSAltitude"])

    return lat, lon, alt

