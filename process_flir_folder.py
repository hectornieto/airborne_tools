# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:19:18 2017

@author: hnieto
"""

import os.path as pth
from glob import glob
import os
import numpy as np
from airborne_tools import flir

#==============================================================================
# # INPUT PARAMETERS
#==============================================================================
sync_imu = True # set True to synchronize GPS_IMU data with the images
hour_lag = 0 # set a non zero value if you suspect that the images are not well synchronized

# Set working folders
workdir = pth.join('H:\\\\', '2018', 'Mollerussa', 'Vuelo20180710', 'images')
# Location of the GPS IMU data
imu_file = pth.join(workdir, 'GPS', 'MOLLE10072018_IG-500N_007000277.txt')
# Folder where the raw FPF images are stored
fpf_dir = pth.join(workdir, 'FLIR', 'L0')
#==============================================================================
# # END INPUT PARAMETERS
#==============================================================================

outdir = pth.join(workdir, 'FLIR', 'L1B')
if not pth.isdir(outdir):
    os.makedirs(outdir)

outdir_8b = pth.join(workdir, 'FLIR', 'TIFF_8bit')
if not pth.isdir(outdir_8b):
    os.makedirs(outdir_8b)   

image_list = glob(pth.join(fpf_dir, '*.FPF'))

if sync_imu:
    print('Reading IMU data')
    imu_data = np.genfromtxt(imu_file, dtype=None, names=True, delimiter='\t')

fid = open(pth.join(workdir, 'imu_log_flir.txt'), 'w')
header = 'camera\tlat\tlon\talt\tyaw\tpitch\troll\n'
fid.write(header)

for image in image_list:
    header_dict, image = flir.read_fpf(image)
    year_str = str(header_dict['year']).zfill(4)
    month_str = str(header_dict['month']).zfill(2)
    day_str = str(header_dict['day']).zfill(2)
    hour_str = str(header_dict['hour']-hour_lag).zfill(2)
    minute_str = str(header_dict['minute']).zfill(2)
    second_str = str(header_dict['second']).zfill(2)
    millisecond_str = str(header_dict['millisecond']).zfill(3)
    output_name = '%s%s%sT%s%s%s%s.tif'%(year_str,
                                         month_str,
                                         day_str,
                                         hour_str,
                                         minute_str,
                                         second_str,
                                         millisecond_str)
    
    flir.saveImg(image, pth.join(outdir, output_name))
	
    if sync_imu:
	    imu = flir.sync_flir_IMU(header_dict, imu_data, hour_lag=hour_lag)#lat, lon, alt, yaw, pitch, roll
	    out_string = output_name
	    for item in imu:
			out_string += '\t%s'%item
	    fid.write(out_string + '\n')
    
    image = flir.Convert_16_to_8_bit(image)
    output_name = '%s%s%sT%s%s%s%s_8bit.tif'%(year_str,
                                              month_str,
                                              day_str,
                                              hour_str,
                                              minute_str,
                                              second_str,
                                              millisecond_str)
    
    flir.saveImg(image, 
                 pth.join(outdir_8b, output_name), 
                 scale=1, 
                 data_type=flir.gdal.GDT_Byte)
    
fid.flush()
fid.close()
