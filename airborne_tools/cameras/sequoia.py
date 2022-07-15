from pathlib import Path
import json
from airborne_tools import exif_tools as et


def filename_to_params(in_filename):
    filename = in_filename.stem
    pattern = filename.rsplit("_", 1)[0]
    path = in_filename.parent
    return path, pattern


def get_bands_from_filename(in_filename):
    path, pattern = filename_to_params(in_filename)
    bands = path.glob(f"{pattern}_*.TIF")
    return bands


def find_incoherent_filenames(input_dir, ref_band="RED"):
    imput_dir = Path(input_dir)
    scenes = imput_dir.glob(f"IMG_*_*_*_{ref_band}.TIF")
    wrong_scenes = []
    for scene in scenes:
        path, pattern = filename_to_params(scene)
        wrong_scenes.append(pattern)
        bands = list(get_bands_from_filename(scene))
        if len(bands) < 4:
            print(f"Missing band for image {pattern}")

    return wrong_scenes


def get_coordinates(image_file):
    exif, xmp = et.get_raw_metadata(str(image_file))
    lat, lon, alt = et.coordinates_from_exif(exif)
    yaw = float(xmp["Xmp.Camera.Yaw"])
    pitch = float(xmp["Xmp.Camera.Pitch"])
    roll = float(xmp["Xmp.Camera.Roll"])
    return lat, lon, alt, yaw, pitch, roll


def create_shots(input_folder, band="RED", out_geojson=None):
    input_folder = Path(input_folder)
    scenes = input_folder.glob(f"*_{band}.TIF")
    json_dict = {"type": "FeatureCollection",
                 "crs": {"type": "name",
                         "properties": {"name": "EPSG:4326"}},
                 "features": []}
    for scene in scenes:
        filename = scene.name
        print(filename)
        lat, lon, alt, yaw, pitch, roll = get_coordinates(scene)
        feature = {"type": "Feature",
                   "geometry": {"type": "Point",
                                "coordinates": [lon, lat]},
                   "properties": {"filename": filename,
                                  "path": str(scene),
                                  "latitude": lat,
                                  "longitude": lon,
                                  "altitude": alt,
                                  "roll": roll,
                                  "pitch": pitch,
                                  "yaw": yaw}}
        json_dict["features"].append(feature)

    if out_geojson:
        out = json.dumps(json_dict,
                         indent=4,
                         separators=(',', ': '))

        with open(out_geojson, "w") as fid:
            fid.write(out)
            fid.flush()

    return json_dict


