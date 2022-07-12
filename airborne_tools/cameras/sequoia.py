from pathlib import Path
import json
from airborne_tools import exif_tools as et

def get_coordinates(image_file):
    exif, xmp = et.get_raw_metadata(str(image_file))
    lat, lon, alt = et.coordinates_from_exif(exif)
    roll = float(xmp["Xmp.Camera.Roll"])
    pitch = float(xmp["Xmp.Camera.Pitch"])
    yaw = float(xmp["Xmp.Camera.Yaw"])
    return lat, lon, alt, roll, pitch, yaw


def create_tracks(input_folder, band="RED", out_geojson=None):
    input_folder = Path(input_folder)
    scenes = input_folder.glob(f"*_{band}.TIF")
    json_dict = {"type": "FeatureCollection",
                 "crs": {"type": "name",
                         "properties": {"name": "EPSG:4326"}},
                 "features": []}
    for scene in scenes:
        filename = scene.name
        print(filename)
        lat, lon, alt, roll, pitch, yaw = get_coordinates(scene)
        feature = {"type": "Feature",
                   "geometry": {"type": "Point",
                                "coordinates": [lon, lat]},
                   "properties": {"filename": filename,
                                  "path": scene,
                                  "latitude": lat,
                                  "longitude": lon,
                                  "altitude": alt,
                                  "roll": roll,
                                  "pitch": pitch,
                                  "yaw": yaw}}
        json_dict["features"].append(feature)

    if out_geojson:
        with open(out_geojson, "w") as fid:
            json.dump(fid, json_dict)

    return json_dict

input_folder = Path("/media/hector/TOSHIBA_2TB/vuelo/images")
out_gejson = input_folder.parent / "track.geojson"
create_tracks(input_folder, band="RED", out_geojson=out_gejson)
