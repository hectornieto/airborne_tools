import numpy as np
import gdal
from PIL import Image
from time import strptime

DATETIME_TAG = 306
EXPOSURE_TAG = 316
GPS_TAG = 34853
IMAGEWIDTH_TAG = 256
IMAGELENGTH_TAG = 257

def read_macaw_header(image):
    header_dict=dict()
    #read file
    img = Image.open(image)
    header_dict['exposure'] = img.tag[EXPOSURE_TAG]
    header_dict['GPS'] = img.tag[GPS_TAG][0]
    header_dict['date_time'] = img.tag[DATETIME_TAG]
    header_dict['n_cols'] = img.tag[IMAGEWIDTH_TAG][0]
    header_dict['n_rows'] = img.tag[IMAGELENGTH_TAG][0]
    
    datetime = strptime(header_dict['date_time'],
                        '%Y:%m:%d %H:%M:%S')
    
    header_dict['year'] = datetime.tm_year
    header_dict['month'] = datetime.tm_mon
    header_dict['day'] = datetime.tm_mday
    header_dict['hour'] = datetime.tm_hour
    header_dict['minute'] = datetime.tm_min
    header_dict['second'] = datetime.tm_sec
    header_dict['millisecond'] = 0
    
    return header_dict
    
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