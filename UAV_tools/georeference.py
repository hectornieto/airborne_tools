def CalcCoordinatesCollinearity(pixel,c_xy,altitude,origin,rotation_matrix, 
                                focal_length):
    '''Calculates the projected coordinates of an image pixel by applying the 
    collinearity equation
    
    Parameters
    ----------
    pixel : pixel coordinates (col,row) with the origin at the top left corner
        of the image with x axis pointing right and y axis pointing down
    c_xy: pixel coordinates (row,col) of the principal point
    altitude: Pixel ground elevation, Z (m)
    origin: the coordinates (X0,Y0,Z0) of the center of projection of the camera
    rotation_matrix : Rotation matrix (Omega,Phi,Kappa)
    focal_length : Camera focal length (pixels)

    Returns
    -------
    X, Y : pixel coordinates (X Y) at ground elevation (altitude)'''
    
    Z_Z0=altitude-float(origin[2])
    y=c_xy[1]-pixel[1]
    x=pixel[0]-c_xy[0]
    denom=(rotation_matrix[2,0]*x+rotation_matrix[2,1]*y-rotation_matrix[2,2]*focal_length)
    X_X0=Z_Z0*(rotation_matrix[0,0]*x+rotation_matrix[0,1]*y-rotation_matrix[0,2]*focal_length)/denom
    Y_Y0=Z_Z0*(rotation_matrix[1,0]*x+rotation_matrix[1,1]*y-rotation_matrix[1,2]*focal_length)/denom
    X=X_X0+origin[0]
    Y=Y_Y0+origin[1]
    return X,Y  
 
def CalcCoordinatesDirectGeorreferencing(pixel,origin,Z0,rotation):
    '''Calculates the projected coordinates of an image pixel by applying a 
    direct georreferencing algebra
    
    Parameters
    ----------
    pixel : pixel coordinates (col,row) with the origin at the top left corner
        of the image with x axis pointing right and y axis pointing down
    c_xy: pixel coordinates (row,col) of the principal point
    altitude: Pixel ground elevation, Z in map projectio system
    origin: the coordinates (X0,Y0,Z0) in map projection system of the center
        of projection of the camera
    rotation_matrix : Rotation matrix (Omega,Phi,Kappa)
    focal_length : Camera focal length (pixels)

    Returns
    -------
    X, Y, Z : pixel coordinates (X Y Z) in map projection system 
        at ground elevation (altitude)'''
    import numpy as np
    #Se rota el vector imagen para el calculo del factor de escala
    img_rot=rotation*pixel.T
    #factor de escala=((h_mdt-h_avion)/zi_rot)
    ha=Z0-float(origin[2])
    landa=ha/float(img_rot[2])
              
    #Ecuacion de colinealidad
    #(Xg,Yg,Zg)=(Xs,Ys,Zs)+landa*dRx(xi,yi,-focal)
    geo_arr=np.matrix(origin).T+(landa*rotation*pixel.T)
    return np.array(geo_arr).reshape(-1)
    
def GetAltitudeDTM(X,Y,dsm,geo_dsm, default_Z=0):
    ''' Get the elevation for a given coordinate from a Digital Surface Model

    Parameters
    ----------
    X, Y : Cordinates of the point of interest in the map projection system
    dsm : Array with the elevation Fields (m)
    geo_dsm : gdal GeoTransForm model
    default_Z : default value in case the X,Y coordinates fall outside the DSM
        extent
    
    Returns
    -------
    altitude: Ground elevation at X Y coodinates
    out: Boolean variable indicating whether the coordinates fell outside the
        DSM boundaries'''    
    
    from numpy import shape
    out=False
    rows_dsm,cols_dsm=shape(dsm)
    col_dsm=int((X-geo_dsm[0])/geo_dsm[1])
    row_dsm=int((Y-geo_dsm[3])/geo_dsm[5])
    if col_dsm >=0 and col_dsm < cols_dsm and row_dsm>=0 and row_dsm < rows_dsm:
        altitude=float(dsm[row_dsm,col_dsm])
        if altitude<=0:
            altitude=default_Z
            out=True
    else:
        altitude=default_Z
        out=True
    return altitude,out      

def GetAltitudeDTM_Numpy(X_array,Y_array,dsm,geo_dsm, default_Z=0):
    ''' Get the elevation for a given coordinate from a Digital Surface Model

    Parameters
    ----------
    X, Y : Cordinates of the point of interest in the map projection system
    dsm : Array with the elevation Fields (m)
    geo_dsm : gdal GeoTransForm model
    default_Z : default value in case the X,Y coordinates fall outside the DSM
        extent
    
    Returns
    -------
    altitude: Ground elevation at X Y coodinates
    out: Boolean variable indicating whether the coordinates fell outside the
        DSM boundaries'''    
    
    import numpy as np
    rows_dsm,cols_dsm=np.shape(dsm)
    altitude=np.zeros(X_array.shape)+default_Z
    col_dsm=np.floor((X_array-geo_dsm[0])/geo_dsm[1]).astype(np.int32)
    row_dsm=np.floor((Y_array-geo_dsm[3])/geo_dsm[5]).astype(np.int32)
    valid=np.where(np.logical_and.reduce((col_dsm >=0, col_dsm < cols_dsm, row_dsm>=0, row_dsm < rows_dsm)))[0].astype(np.int32)
    altitude[valid]=dsm[row_dsm[valid],col_dsm[valid]]
    return altitude  
        
def CalcObsGeometry(coordinates,origin):
    ''' Estimates the Observation Geometry based on the pixel coordinates 
        and camera coordinates
    
    Parameters
    ---------
    coordinates : Xi,Yi,Zi coordinates of observed pixel
    origin : X0,Y0,Z0 coordinates of the camera (=nadir)
    
    Returns
    -------
    vza : View Zenith Angle (degrees)
    vza : View Azimuth Angle (degrees)'''
    
    import numpy as np
    # Get the view zenith angle
    ref_vec= np.array([0,0,-1])
    coordinates=np.asarray(coordinates)
    obs_vec=np.asarray([coordinates[0]-origin[0],coordinates[1]-origin[1],coordinates[2]-origin[2]])
    cos_theta=obs_vec[2]/(ref_vec[2]*np.sqrt(obs_vec[0]**2+obs_vec[1]**2+obs_vec[2]**2))
    vza=np.degrees(np.arccos(cos_theta))
    # Get the view azimuth angle
    vaa=np.degrees(np.arctan2((obs_vec[0],obs_vec[1])))
    return vza,vaa

def CalcAzimuth(lon_0,lat_0,lon_1,lat_1):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''
    
    from math import sqrt,acos,degrees
    X=lon_1-lon_0
    Y=lat_1-lat_0
    azimuth=Y/sqrt(X**2+Y**2)
    azimuth=degrees(acos(azimuth))
    return azimuth

def CorrectDistortion(pixel, c_xy,f_xy, K, T=[0,0], skew=0):
    '''Corrects the lens distortion based on Photoscan pinhole camera 
        for a pixel coordinate
    
    Parameters
    ----------
    pixel : Uncorrected pixel coordinate (col,row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down
    c_xy: image princpal point (col,row)
    f_xy:focal lenght (x.y) in pixels
    K :  Brown's radial distortion parameters [k1,k2,k3,k4]
    T : tangential distortion parameters [P1, P2] 
        (default=[0,0], no tangential distortion)
    skew : skewness coefficient (default=0, no skew)
    
    Returns
    -------
    u,v : Corrected pixel coordinates (col, row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down'''

    from math import sqrt
    x=(pixel[0]-c_xy[0])/f_xy[0]
    y=(pixel[1]-c_xy[1])/f_xy[1]
    r=sqrt(x**2+y**2)
    x_u=x*(1.0 + K[0]*r**2 + K[1]*r**4 + K[2]*r**6 + K[3]*r**8)\
        +T[1]*(r**2+2.0*x**2)+2.0*T[0]*x*y
    y_u=y*(1.0 + K[0]*r**2 + K[1]*r**4 + K[2]*r**6 + K[3]*r**8)\
        +T[0]*(r**2+2.0*y**2)+2.0*T[1]*x*y
    u=c_xy[0]+x_u*f_xy[0]+skew*y_u
    v=c_xy[1]+y_u*f_xy[1]
    return u,v

def Undistort(img, c_xy,f_xy, K, T=[0,0], skew=0):
    '''Corrects the lens distortion based on Photoscan pinhole camera 
        for a pixel coordinate
    
    Parameters
    ----------
    img : Uncorrected image
    c_xy: image princpal point (col,row)
    f_xy:focal lenght (x.y) in pixels
    K :  Brown's radial distortion parameters [k1,k2,k3,k4]
    T : tangential distortion parameters [P1, P2] 
        (default=[0,0], no tangential distortion)
    skew : skewness coefficient (default=0, no skew)
    
    Returns
    -------
    u,v : Corrected pixel coordinates (col, row) with the origin at the top 
        left corner of the image with x axis pointing right and y axis pointing down'''

    import numpy as np
    import cv2
    
    xx,yy=np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    #Calculate undistorted pixel coordinates
    x=(xx-c_xy[0])/f_xy[0]
    y=(yy-c_xy[1])/f_xy[1]
    r=np.sqrt(x**2+y**2)
    x_u=x*(1.0 + K[0]*r**2 + K[1]*r**4 + K[2]*r**6 + K[3]*r**8)\
        +T[1]*(r**2+2.0*x**2)+2.0*T[0]*x*y
    y_u=y*(1.0 + K[0]*r**2 + K[1]*r**4 + K[2]*r**6 + K[3]*r**8)\
        +T[0]*(r**2+2.0*y**2)+2.0*T[1]*x*y
    u=c_xy[0]+x_u*f_xy[0]+skew*y_u
    v=c_xy[1]+y_u*f_xy[1]
    #u_remap=img.shape[1]*(u-u_min)/(u_max-u_min)
    #v_remap=img.shape[0]*(v-v_min)/(v_max-v_min)
    # Assign image values to the nearest image value
    u_map=u.astype('float32')
    v_map=v.astype('float32')
    dst=cv2.remap(img,u_map,v_map,cv2.INTER_LINEAR)
    #dst=np.zeros(img.shape)
    #pixels=[(row,col) for row in range(img.shape[0]) for col in range(img.shape[1])]
    #for pixel in pixels:
        #dst[int(v[pixel]),int(u[pixel]),:]=img[pixel[0],pixel[1],:]
    return dst
    
def CorrectMeridanConvergenceUTM(lat,lon,lon_cm):
    ''' Apply a heading offset correction for meridian UTM convergence
    
    Parameters
    ----------
    lat, lon : Point latitude and longitude coordinates (degrees)
    lon_cm : Central Meridian longitude (degrees)
    
    Returns
    -------
    offset : heading offset to apply to the geographic heading (degrees)'''
    
    from math import radians, degrees, tan, sin, atan
    offset=degrees(atan(sin(radians(lat))*tan(radians(lon-lon_cm))))
    return offset

def CalcPixelCollinearity(coordinates,c_xy,origin,rotation_matrix, focal_length):
    '''Calculates the pixel coordinates for a map projected coordinates by applying the 
    collinearity equation
    
    Parameters
    ----------
    coordinates : map projected coordinates (X,Y,Z) 
    c_xy: pixel coordinates (row,col) of the principal point
    origin: the coordinates (X0,Y0,Z0) of the center of projection of the camera
    rotation_matrix : Rotation matrix (Omega,Phi,Kappa)
    focal_length : Camera focal length (pixels)

    Returns
    -------
    u, v : pixel coordinates (col, row) with the origin at the top left corner
        of the image with x axis pointing right and y axis pointing down '''
    
    X_X0=coordinates[0]-origin[0]
    Y_Y0=coordinates[1]-origin[1]
    Z_Z0=coordinates[2]-origin[2]
    denom=(rotation_matrix[0,2]*X_X0+rotation_matrix[1,2]\
        *Y_Y0+rotation_matrix[2,2]*Z_Z0)
    x=-focal_length*(rotation_matrix[0,0]*X_X0+rotation_matrix[1,0]\
        *Y_Y0+rotation_matrix[2,0]*Z_Z0)/denom
    y=-focal_length*(rotation_matrix[0,1]*X_X0+rotation_matrix[1,1]\
        *Y_Y0+rotation_matrix[2,1]*Z_Z0)/denom
    u=x+c_xy[0]
    v=c_xy[1]-y
    return u,v

def CalcRotationIMU(angle_x,angle_y,angle_z):
    '''Calculates the 3D rotation matrix, first rotation over X, second over Y 
        and third over Z
    
    Parameters
    ----------
    angle_x,angle_y,angle_z : rotation angles (degrees)
    
    Returns
    -------
    rotation: 3D rotation matrix'''
    
    Rz=CalcRz(angle_z)
    Ry=CalcRy(angle_y)
    Rx=CalcRx(angle_x)
    rotation=Rz*Ry*Rx
    return rotation

def CalcRollPitchYaw(rotation):
    '''Calculate the rotation angles from a rotation matrix (Rz(kappa),Ry(pitch),Rx(roll))

    Parameter
    --------
    rotation : 3D rotation matrix
    
    Returns
    -------
    Omega,phi,kappa : photogrametric angles'''
    
    import numpy as np
    alpha_y=np.degrees(np.arcsin(-rotation[2,0]))
    alpha_x=np.degrees(np.arctan2(rotation[2,1],rotation[2,2]))
    alpha_z=np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))
    return alpha_x,alpha_y,alpha_z
    
def CalcRx(alpha):
    '''Rotation over x axis
    
    Parameter
    ---------
    alpha : rotation angle over X
    
    Returns
    -------
    Rx: rotation matrix over X'''
    
    import numpy as np
    alpha=np.radians(alpha)
    Rx=np.matrix([[1,0,0],
                  [0,np.cos(alpha),-np.sin(alpha)],
                   [0,np.sin(alpha),np.cos(alpha)]])
    return Rx

def CalcRy(alpha):
    '''Rotation over Y axis
    
    Parameter
    ---------
    alpha : rotation angle over Y
    
    Returns
    -------
    Ry: rotation matrix over Y'''    
    
    import numpy as np
    alpha=np.radians(alpha)
    Ry=np.matrix([[np.cos(alpha),0,np.sin(alpha)],
                   [0,1,0],
                    [-np.sin(alpha),0,np.cos(alpha)]])
    return Ry
   
def CalcRz(alpha):
    '''Rotation over Z axis
    
    Parameter
    ---------
    alpha : rotation angle over Z
    
    Returns
    -------
    Rz: rotation matrix over Z'''     
    
    import numpy as  np
    alpha=np.radians(alpha)    
    Rz=np.matrix([[np.cos(alpha),-np.sin(alpha),0],
                   [np.sin(alpha),np.cos(alpha),0],
                    [0,0,1]])
    return Rz
    
def CalcOmegaPhiKappa(rotation):
    '''Calculate the rotation angles from a rotation matrix (Rz(kappa),Ry(pitch),Rx(roll))

    Parameter
    --------
    rotation : 3D rotation matrix
    
    Returns
    -------
    Omega,phi,kappa : photogrametric angles'''
    
    from math import degrees, asin, atan
    phi=degrees(asin(rotation[0][2]))
    omega=degrees(atan(-rotation[1][2]/rotation[2][2]))
    kappa=degrees(atan(-rotation[0][1]/rotation[0][0]))
    return omega,phi,kappa

def CalcRotationMatrixIMU_Legat(yaw,pitch,roll,lat_0,lon_0,lat_origin,lon_origin,
            lon_cm,R_n_l= [[1,0,0],[0,1,0],[0,0,1]],R_e_n=[[1,0,0],[0,1,0],[0,0,1]],
            R_c_b=[[1,0,0],[0,1,0],[0,0,1]]):
    from numpy import dot,array,radians,degrees,sin,cos
    
    R_lprime_p=array([[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,-1.0]])

    a=6378137.0
    b=a*(1.0-1.0/298.257223563)
    e2=(a**2-b**2)/b**2
    gamma_PC=radians(lon_0-lon_cm)*sin(radians(lat_0))+\
        radians((lon_0-lon_cm)/3)*sin(radians(lat_0))*cos(radians(lat_0))**2+\
        3*e2*cos(radians(lat_0))**2
    gamma_PC=degrees(gamma_PC)  
    R_l_lprime=CalcRz(-gamma_PC)
    
    R_n_l=array(R_n_l)
    R_e_n=array(R_e_n)
    R_l_e=CalcRotationMatrixECEF(lat_0,lon_0).T
    R_b_l=CalcRotation(roll,pitch,yaw)
    R_c_b=array(R_c_b)
    
    R_c_p=dot(R_b_l,R_c_b)
    R_c_p=dot(R_l_e,R_c_p)
    R_c_p=dot(R_e_n,R_c_p)
    R_c_p=dot(R_n_l,R_c_p)
    R_c_p=dot(R_l_lprime,R_c_p)
    R_c_p=dot(R_lprime_p,R_c_p)
    
    return R_c_p 

def CalcRotationMatrixIMU_Skaloud(yaw,pitch,roll,lat_0,lon_0,lat_origin,lon_origin,
            lon_cm,R_e_n=[[1,0,0],[0,1,0],[0,0,1]]):
    from numpy import dot,array,radians,degrees,sin,cos
   
    R_lbar_n=CalcRotationMatrixECEF(lat_0,lon_0).T
    R_n_e=array(R_e_n).T
    R_e_l=CalcRotationMatrixECEF(lat_origin,lon_origin)
    R_l_b=CalcRotation(roll,pitch,yaw)
    
    a=6378137.0
    b=a*(1.0-1.0/298.257223563)
    e2=(a**2-b**2)/b**2
    gamma_PC=radians(lon_0-lon_cm)*sin(radians(lat_0))+\
        radians((lon_0-lon_cm)/3)*sin(radians(lat_0))*cos(radians(lat_0))**2*\
        (1.0+3*e2*cos(radians(lat_0))**2)
    gamma_PC=degrees(gamma_PC)  
    R_l_lprime=CalcRz(-gamma_PC)
    
    R_ENU_NED=array([[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,-1.0]])
    
    R_b_b=dot(R_n_e,R_lbar_n)
    R_b_b=dot(R_e_l,R_b_b)
    R_b_b=dot(R_l_b,R_b_b)
    
    R_b_b=dot(R_b_b,R_l_lprime)
    R_b_b=dot(R_b_b,R_ENU_NED)
    
    return R_b_b 

def CalcRotationMatrixIMU_Baumker(yaw,pitch,roll,lat_0,lon_0,lat_origin,lon_origin,
                                  lon_cm,R_c_b=[[1,0,0],[0,1,0],[0,0,1]]):
    from numpy import dot,array
    
    R_lprime_p=array([[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,-1.0]])
    R_l0_lprime=CalcRotationCurvature(lat_0,lon_0,lon_origin,lat_origin,lon_cm)
    R_e_l0=CalcRotationMatrixECEF(lat_origin,lon_origin).T
    R_l_e=CalcRotationMatrixECEF(lat_0,lon_0)
    R_b_l=CalcRotation(roll,pitch,yaw)
    R_c_b=array(R_c_b)
    

    #C_b_ni=CalcRotationMatrixNavigation(yaw,pitch,roll)
    R_c_p=dot(R_b_l,R_c_b)
    R_c_p=dot(R_l_e,R_c_p)
    R_c_p=dot(R_e_l0,R_c_p)
    R_c_p=dot(R_l0_lprime,R_c_p)
    R_c_p=dot(R_lprime_p,R_c_p)
    
    return R_c_p 
    
def CalcRotationMatrixECEF(lat,lon):
    
    from  numpy import array
    from math import cos, sin , radians
    
    lon=radians(lon)
    lat=radians(lat)
    C_e_n=array([[-sin(lat)*cos(lon),-sin(lat)*sin(lon),cos(lat)],\
                [-sin(lon),cos(lon),0],\
                [cos(lat)*cos(lon),cos(lat)*sin(lon),sin(lat)]])
    return C_e_n

def CalcRotationCurvature(lon,lat,lon_origin,lat_origin,lon_cm):

    from numpy import  array
    from math import radians, cos, sin
    
    lon=radians(lon)
    lat=radians(lat)
    lon_origin=radians(lon_origin)
    lat_origin=radians(lat_origin)
    lon_cm=radians(lon_cm)
    e_n=(lon-lon_origin)*cos(lat)
    e_e=(lat-lat_origin)
    e_v=(lon-lon_cm)*sin(lat)
    C_n0_n=array([[1.0,e_v,-e_e],
                  [-e_v,1.0,e_n],
                  [e_e,-e_n,1.0]])

    return C_n0_n

