from pathlib import Path
from airborne_tools import temperature as temp

workdir = Path()
indir = workdir / "test"
outdir = workdir / "test"

lst_file = indir / "SLM_001_002_20140809_1041_TIR.tif"
ndvi_file = indir / "SLM_001_002_20140809_1041_NDVI.tif"

out_file = outdir / "SLM_001_002_20140809_1041_TC-TS.tif"
out_res = 3.6
ndvi_soil = 0.3
ndvi_veg = 0.6

temp. component_temperatures(lst_file, ndvi_file, ndvi_soil, ndvi_veg, out_res,
                             out_file)