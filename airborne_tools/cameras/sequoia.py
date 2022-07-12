from airborne_tools import exif_tools as et

def get_coordinates(image_file):
    exif, xmp = et.get_raw_metadata(image_file)
    lat, lon, alt = et.coordinates_from_exif(exif)
    roll = float(xmp["Xmp.Camera.Roll"])
    pitch = float(xmp["Xmp.Camera.Pitch"])
    yaw = float(xmp["Xmp.Camera.Yaw"])
    return lat, lon, alt, roll, pitch, yaw
