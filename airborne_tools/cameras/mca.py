import os.path as pth
import struct
import numpy as np
import gdal
from PIL import Image
from PIL.TiffTags import TAGS
from time import strptime

DATETIME_TAG = 306
EXPOSURE_TAG = 316
GPS_TAG = 34853
IMAGEWIDTH_TAG = 256
IMAGELENGTH_TAG = 257
  
def read_raw10_MCA(input_file):
    
##10 Bit Raw File Format
##The RAW file format contains both Header and trailer information. For values
##greater than 255, two bytes are used in little endian (Intel) configuration for
##header, trailer and pixel values.
##Byte 0-3 Size of raw image in bytes – 32 bit value
##Byte 4 Bits per pixel – 10 for this format
##Byte 5 Format tag – 16 for RAW files
##Bytes 6-7 Pixel Columns – 16 bit value. This is pixels not bytes
##Bytes 8-9 Pixel Rows – 16 bit values
##Bytes 10-(image size + 10) PIXEL DATA – 16 bit values
##Bytes (image size + 10)-(EOF - 28) GPS data. $GGA and $RMC strings
##Last 28 Bytes – ASCII exposure string
##formatted: "EXPOSURE:%08ld uSeconds\n"
    
    if not pth.isfile(input_file):
        print('File does not exists')
        return

    #Read file
    with open(input_file, 'rb') as fid:
        fileContent =fid.read()
    # Read header
    header=fileContent[:10]
    header_dict={}
    #Image size
    header_dict['image_size']=struct.unpack('L',header[:4])[0]
    #Bit depth
    header_dict['bits']=struct.unpack('b',header[4:5])[0]
    #Format tag
    header_dict['tag']=struct.unpack('b',header[5:6])[0]
    #Columns
    header_dict['n_cols']=struct.unpack('h',header[6:8])[0]   
    #Rows
    header_dict['n_rows']=struct.unpack('h',header[8:10])[0]
    
    total_elements=header_dict['n_cols']*header_dict['n_rows']
    image_binary=fileContent[10:2*total_elements+10]
    
    fmt='%sh'%total_elements
    image=np.asarray(struct.unpack(fmt,image_binary))
    image=image.reshape((header_dict['n_rows'],header_dict['n_cols']))
    GPS=fileContent[2*total_elements+10:-28]
    header_dict['GPS']=struct.unpack('%ss'%len(GPS),GPS)[0]
    exposure=fileContent[-28:]
    header_dict['exposure']=struct.unpack('%ss'%len(exposure),exposure)[0]
               
    return header_dict,image

def Convert_16_to_8_bit(image):

    #Converts the 16 bit image array into a 8 bit for creating an RGB image for Photoscan
    from scipy.stats import scoreatpercentile
    import numpy
    
    #  2% Linear stretching
    im_min=scoreatpercentile(image.flatten(),2)
    im_max=scoreatpercentile(image.flatten(),98)

    shape=numpy.shape(image)

    if im_max==im_min:
        im256=numpy.zeros(shape)
    else:
        im256=255*((image-im_min)/(im_max-im_min))
        im256[image<im_min]=0
        im256[image>im_max]=255

    return im256 

def sync_macaw_IMU(header_dict, imu_data, hour_lag=0):
    
    index=np.logical_and.reduce((imu_data['year']==header_dict['year'],
                                 imu_data['month']==header_dict['month'],
                                 imu_data['day']==header_dict['day'],
                                 imu_data['hour']==header_dict['hour']-hour_lag,
                                 imu_data['min']==header_dict['minute'],
                                 imu_data['sec']==header_dict['second']))
    
    lats = imu_data['latitude'][index]
    lons = imu_data['longitude'][index]
    alts = imu_data['altitude'][index]
    yaws = imu_data['yaw'][index]
    pitchs = imu_data['pitch'][index]
    rolls = imu_data['roll'][index]
		
    # Find the closest data in time
    nano_seconds = imu_data['nano'][index]*1e-9

    dif_nano = np.abs(nano_seconds-header_dict['millisecond']*1e-3)
    min_diff = np.min(dif_nano)
    index = dif_nano==min_diff
    
    lat = lats[index][0]
    lon = lons[index][0]
    alt = alts[index][0]
    yaw = yaws[index][0]
    pitch = pitchs[index][0]
    roll = rolls[index][0]
    return lat, lon, alt, yaw, pitch, roll
	
def saveImg(data, outPath, scale=100, data_type=gdal.GDT_UInt16):
    
    # Start the gdal driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outPath, data.shape[1], data.shape[0], 1, data_type)
    ds.GetRasterBand(1).WriteArray(data*scale)
    print('Saved ' +outPath )
    ds = None

    return True