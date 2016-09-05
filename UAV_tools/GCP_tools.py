# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:13:51 2016

@author: hector
"""
from __future__ import print_function
import gdal
import numpy as np
import cv2

def scale_grayscale_image(image, noData=None):
    
    if noData==None:
        index=np.ones(image.shape,dtype=np.bool)
    else:
        index=image!=noData
    if np.sum(index)>30:
        image[index]= 255*((image[index]-np.amin(image[index]))/(np.amax(image[index])-np.amin(image[index])))
        image=image.astype(np.uint8)
        image=cv2.equalizeHist(image.astype(np.uint8))
    else:
        image*=0
    return image

def get_map_coordinates(row,col,geoTransform):
    X=geoTransform[0]+geoTransform[1]*col+geoTransform[2]*row
    Y=geoTransform[3]+geoTransform[4]*col+geoTransform[5]*row
    return X,Y

def get_pixel_coordinates(X,Y,geoTransform):
    row=(Y - geoTransform[3]) / geoTransform[5]
    col=(X - geoTransform[0]) / geoTransform[1]
    return row,col

def compute_PCA(imageFile,pcaComponents=None,outfile=None,bandsPCA=None,normalize=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    #from sknn.platform import gpu32
    
    driver = gdal.GetDriverByName('GTiff')
    
    fid=gdal.Open(imageFile,gdal.GA_ReadOnly)
    if bandsPCA:    
        bands=list(bandsPCA)
        nbands=len(bands)
    else:
        nbands=fid.RasterCount
        bands=range(nbands)
    if not pcaComponents:
        pcaComponents=nbands
    cols=fid.RasterXSize
    rows=fid.RasterYSize
    hyper_geo=fid.GetGeoTransform()
    hyper_prj=fid.GetProjection()
    input_array=np.zeros((rows*cols,nbands))
    for i,band in enumerate(bands):
        print('Reading hyperspectral band %s'%band)
        array=fid.GetRasterBand(band+1).ReadAsArray()
        mask=array>0
        if normalize:
            scalerInput=StandardScaler()
            scalerInput.fit(array[array>0].reshape(-1,1))
            input_array[:,i]=scalerInput.transform(array.reshape(-1,1)).reshape(-1)
            input_array[:,i]*=mask.reshape(-1)
        else:
            input_array[:,i]=array.reshape(-1)
        del array
    del fid
    pca = PCA(n_components=pcaComponents)
    input_array=np.ma.masked_array(input_array,mask=input_array==0)
    pca.fit(input_array)
    print('Explained variance per component,'+str(pca.explained_variance_ratio_))
    print('Explained variance total,'+str(np.sum(np.asarray(pca.explained_variance_ratio_))))
    
    output_array=pca.transform(input_array)*mask.reshape(-1,1)
    output_array=output_array.reshape((rows,cols,pcaComponents))
    if outfile:
        ds = driver.Create(outfile, cols, rows ,pcaComponents, gdal.GDT_Float32)
        ds.SetGeoTransform(hyper_geo)
        ds.SetProjection(hyper_prj)
        for band in range(pcaComponents):
            ds.GetRasterBand(band+1).WriteArray(output_array[:,:,band])
            ds.FlushCache()
        del ds
    
    return output_array,pca.explained_variance_ratio_

def warp_image_with_GCPs(input_file,GCP_list,output_extent,subsetBands=None,transform=0):
    from os.path import exists,dirname
    from os import remove
    import subprocess
    
    GDALGCPs=create_gdal_GCPs(GCP_list)  
    
    infid=gdal.Open(input_file,gdal.GA_ReadOnly)
    prj=infid.GetProjection()
    geo=infid.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    tempfile=dirname(input_file)+'/temp.tif'
    if exists(tempfile): 
        remove(tempfile)
    infid=gdal.Open(input_file,gdal.GA_ReadOnly)
    
    nbands=infid.RasterCount
    if subsetBands==None:
        subsetBands=range(1,nbands+1)
    nbands=len(subsetBands)
    ds = driver.Create(tempfile, infid.RasterXSize, infid.RasterYSize,nbands, gdal.GDT_UInt16)
    for i,band in enumerate(subsetBands):
        print('Saving Band ' +str(band))
        array=infid.GetRasterBand(band).ReadAsArray()
        band=ds.GetRasterBand(i+1)
        band.WriteArray(array)
        band.SetNoDataValue(0)
        del band
        del array
    del infid
    
    ds.SetGCPs(GDALGCPs,prj)
    del ds
    # Run GDAL Warp
    outfile=input_file[:-4]+'_Georref.bsq'
    if transform==0:
        gdal_command='gdalwarp -tps -overwrite -r bilinear -of ENVI -srcnodata 0 -dstnodata 0 -multi -tr %s %s '%(geo[1],geo[5]) +' -te %s %s %s %s '%output_extent+' "'+ tempfile + '" "'  + outfile +'"'
    else:
        gdal_command='gdalwarp -order ' +str(transform)+' -overwrite -r bilinear -of ENVI -srcnodata 0 -dstnodata 0 -multi -tr %s %s '%(geo[1],geo[5]) +' -te %s %s %s %s '%output_extent+' "'+ tempfile + '" "'  + outfile +'"'
    
    proc=subprocess.Popen(gdal_command,shell=True,stdout=subprocess.PIPE,
         stdin=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    while True:
        out = proc.stdout.read(1)
        if out == '' and proc.poll() != None:
        	break
        if out != '':
            print(out,end='')
    proc.communicate()
    remove(tempfile)
    return True

def find_GCPs(masterImage,slaveImage,masterGeo,UL_offset=(0,0),match_factor=0.75,useSIFT=False):

    GCP_list=[]
    # Create the feature detector/descriptor and matching objects
    if useSIFT==True:
        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 2
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
    else:
        detector=cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level =2) #2
        search_params = dict(checks=50)   # or pass empty dictionary
    
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    # Finde features and their descriptors
    kp_master, des_master = detector.detectAndCompute(masterImage,None)
    kp_slave, des_slave = detector.detectAndCompute(slaveImage,None)
    if len(kp_master) < 2 or len(kp_slave) < 2:
        return GCP_list
    # Get the 2 best matches per feature
    matches = matcher.knnMatch(des_master,des_slave,k=2)
    GCP_id=0
    GCP_list=[]
    for i,m_n in enumerate(matches):
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        dist_ratio=m.distance/n.distance
        if  dist_ratio < match_factor:
            master_pt=np.float32( kp_master[m.queryIdx].pt)
            Xmaster,Ymaster = get_map_coordinates(float(master_pt[1]),float(master_pt[0]),masterGeo)
            slave_pt=np.float32( kp_slave[m.trainIdx].pt)
            GCP_list.append((Xmaster,Ymaster,UL_offset[0]+float(slave_pt[1]),UL_offset[1]+float(slave_pt[0]),dist_ratio))
            GCP_id+=1
    return GCP_list
    
def filter_GCP_by_map_proximity(GCPs,slaveGeo,distThres):

    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)
    Xslave,Yslave = get_map_coordinates(GCPs[:,2],GCPs[:,3],slaveGeo)
    dist=np.sqrt((Xslave-GCPs[:,0])**2+(Yslave-GCPs[:,1])**2)
    GCPs=GCPs[dist<=distThres]
    return GCPs
    
def filter_GCP_by_GCP_proximity(GCPs,pixelThres):

    # Filter GCPs based on image proximity
    GCPs_good=[]
    for i,gcp_test in enumerate(GCPs):
        good =True
        if i==len(GCPs)-2: continue
        for j in range(i+1,len(GCPs)):
            #print(i,j)
            dist=np.sqrt((gcp_test[2]-GCPs[j][2])**2+(gcp_test[3]-GCPs[j][3])**2)
            if dist<pixelThres:# GCPs closer to each other are discarded to avoid overfitting
                good=False
                break
        if good==True:
            GCPs_good.append(gcp_test)

    return GCPs_good

def filter_GCP_by_intersection(GCPs,slave_geo):

    # Filter GCPs based on image proximity
    GCPs_good=[]
    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)

    X_slave,Y_slave=get_map_coordinates(GCPs[:,2],GCPs[:,3],slave_geo)
    obs_vec=(GCPs[:,0]-X_slave,GCPs[:,1]-Y_slave)
    azimuth=calc_azimuth(obs_vec)
    cos_azimuth=np.cos(np.radians(azimuth))
    sin_azimuth=np.sin(np.radians(azimuth))
    mean_azimuth=np.degrees(np.arctan2(np.mean(sin_azimuth),np.mean(cos_azimuth)))
    diff=np.abs(calc_azimuth_difference(azimuth,mean_azimuth))
    indices=diff.argsort()[::-1]
    for i,index in enumerate(indices):
        good =True
        if i==len(indices)-2: continue
        for j in range(i+1,len(indices)):
            #print(i,j)
            intercept=calc_vector_intersection((X_slave[index],Y_slave[index]),(GCPs[index,0],GCPs[index,1]),
                                             (X_slave[indices[j]],Y_slave[indices[j]]),(GCPs[indices[j],0],GCPs[indices[j],1]))
            if intercept:# GCPs closer to each other are discarded to avoid overfitting
                good=False
                break
        if good==True:
            GCPs_good.append(GCPs[index])

    return GCPs_good

def filter_GCP_by_azimuth(GCPs,slave_geo,angleThres):

    # Filter GCPs based on image proximity
    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)
    X_slave,Y_slave=get_map_coordinates(GCPs[:,2],GCPs[:,3],slave_geo)
    obs_vec=(GCPs[:,0]-X_slave,GCPs[:,1]-Y_slave)
    azimuth=calc_azimuth(obs_vec)
    cos_azimuth=np.cos(np.radians(azimuth))
    sin_azimuth=np.sin(np.radians(azimuth))
    mean_azimuth=np.degrees(np.arctan2(np.mean(sin_azimuth),np.mean(cos_azimuth)))
    diff=np.abs(calc_azimuth_difference(azimuth,mean_azimuth))
    GCPs_good=GCPs[diff<=angleThres]
    return GCPs_good

def calc_vector_intersection(start1,end1,start2,end2):

    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
    return ccw(start1,start2,end2) != ccw(end1,start2,end2) and ccw(start1,end1,start2) != ccw(start1,end1,end2)


def filter_GCP_by_warp_error(GCPs,errorThres):
    def fit_polynomial_warp(GCPs):
    
        GCPs=np.asarray(GCPs)
        if GCPs.shape[0]<15:
            return None,None
        rows=GCPs[:,2]
        cols=GCPs[:,3]
        rows2=rows**2
        cols2=cols**2
        rowscols=rows*cols
        rows2cols=rows**2*cols
        rowscols2=rows*cols**2
        rows3=rows**3
        cols3=cols**3
        
        X=np.matrix([np.ones(rows.shape),rows,cols,rowscols,rows2,cols2,rows2cols,rowscols2,rows3,cols3]).T
        mapX=GCPs[:,0].reshape(-1,1)
        mapY=GCPs[:,1].reshape(-1,1)
        thetaX=(X.T*X).I*X.T*mapX    
        thetaY=(X.T*X).I*X.T*mapY    
        return np.asarray(thetaX).reshape(-1),np.asarray(thetaY).reshape(-1)

    def calc_warp_erors(GCPs,thetaX,thetaY):
        def polynomial_warp(rows,cols,thetaX,thetaY):
        
            X=thetaX[0]+thetaX[1]*rows+thetaX[2]*cols+thetaX[3]*rows*cols+thetaX[4]*rows**2+thetaX[5]*cols**2+thetaX[6]*rows**2*cols+thetaX[7]*rows*cols**2+thetaX[8]*rows**3+thetaX[9]*cols**3
            Y=thetaY[0]+thetaY[1]*rows+thetaY[2]*cols+thetaY[3]*rows*cols+thetaY[4]*rows**2+thetaY[5]*cols**2+thetaY[6]*rows**2*cols+thetaY[7]*rows*cols**2+thetaY[8]*rows**3+thetaY[9]*cols**3
            return X,Y
    
        GCPs=np.asarray(GCPs)
        if len(GCPs.shape)==1:
            GCPs=GCPs.reshape(1,-1)
    
        rows=GCPs[:,2]
        cols=GCPs[:,3]
        X_model,Y_model=polynomial_warp(rows,cols,thetaX,thetaY)
        error=np.sqrt((X_model-GCPs[:,0])**2+(Y_model-GCPs[:,1])**2)
        return error

    GCPs=np.asarray(GCPs)
    if GCPs.shape[0]<15:
        return GCPs
    thetaX,thetaY=fit_polynomial_warp(GCPs)
    error=calc_warp_erors(GCPs,thetaX,thetaY)
    while np.max(error)>errorThres and len(error)>30:
        index=error.argsort()[::-1]
        GCPs=GCPs[index[1:]]
        thetaX,thetaY=fit_polynomial_warp(GCPs)
        error=calc_warp_erors(GCPs,thetaX,thetaY)

    return GCPs
    
def calc_azimuth(obs_vec):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''

    # Get the view azimuth angle
    azimuth = np.degrees(np.arctan2(obs_vec[0],obs_vec[1]))
    return azimuth

def calc_azimuth_difference(azimut1,azimut2):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''

    # Get the view azimuth angle
    azimuth = np.degrees(np.arctan2(np.sin(np.radians(azimut1-azimut2)),np.cos(np.radians(azimut1-azimut2))))
    return azimuth

def create_gdal_GCPs(GCPs):

    gdalGCP=[]
    for i,gcp in enumerate(GCPs):
        gdalGCP.append(gdal.GCP(gcp[0],gcp[1],0,gcp[3],gcp[2],'',str(i)))
    return gdalGCP

def GCPs_to_ASCII(GCPs,outfile):

    header='ID \t X \t Y \t Row \t Col'
    fid=open(outfile,'w')
    fid.write(header+'\n')
    for i,gcp in enumerate(GCPs):
        fid.write('%s \t %s \t %s \t %s \t %s \n'%(i,gcp[0],gcp[1],gcp[2],gcp[3]))
    fid.flush()
    fid.close()
    return True

def ASCII_to_GCP(infile,GCP=[]):

    indata=np.genfromtxt(infile,names=True,dtype=None)
    for data in indata:
        GCP.append([float(data['X']),float(data['Y']),float(data['Row']),float(data['Col'])])
    return GCP

def write_transformation_vector(slaveCoords,masterCoords,outshapefile,prj):
    from osgeo import ogr
    import osr
    import os

    DriverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(DriverName)
    if os.path.exists(outshapefile):
        driver.DeleteDataSource(outshapefile)
    
    data_source = driver.CreateDataSource(outshapefile)
    # create the layer
    srs=osr.SpatialReference()
    srs.ImportFromWkt(prj)
    layer = data_source.CreateLayer("Transformation Vectors", srs, ogr.wkbLineString)

    for i,slaveXCoord in enumerate(slaveCoords[0][:]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
 
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(slaveXCoord,slaveCoords[1][i])
        line.AddPoint(masterCoords[0][i],masterCoords[1][i])
          # Set the feature geometry using the point
        feature.SetGeometry(line)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()
    # Destroy the data source to free resources
    data_source.Destroy()
    return True
