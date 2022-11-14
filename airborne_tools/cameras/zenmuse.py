import os
from pathlib import Path
import subprocess as sp
from airborne_tools  import exif_tools as et
import numpy as np
import scipy as sci
from PIL import Image

DJI_THERMAL_SDK_BIN = Path.home() / "dji_thermal_sdk"/ "utility" / "bin" / "linux" / \
                      "release_x64" / "dji_irp"

os.environ["LD_LIBRARY_PATH"] = str(DJI_THERMAL_SDK_BIN.parent)

IMAGE_SIZE = 512, 640

def z10t_to_tif(input_image, output_image, export_metadata=True):

    output_folder = output_image.parent
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    filename = output_image.stem

    bin_image = output_folder / filename
    args = [str(DJI_THERMAL_SDK_BIN), "-a", "measure", "-s", str(input_image),  "-o", str(bin_image)]

    proc = sp.Popen(args)
    try:
        outs, errs = proc.communicate(timeout=15)
        # print(outs)
        print(errs)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print(outs)
        print(errs)

    raw_to_tif(bin_image, output_image)
    bin_image.unlink()

    if export_metadata:
        exif, xmp = et.get_raw_metadata(str(input_image))
        et.update_metadata(str(output_image), exif=exif, xmp=xmp)



def raw_to_tif(raw_image, tif_output):

    dtype = np.int16
    array = np.fromfile(raw_image, dtype)

    array = 100 * (273.15 + array.reshape(IMAGE_SIZE).astype(np.float32) / 10)
    im = Image.fromarray(array.astype(np.uint16))
    im.save(tif_output)
    return array


