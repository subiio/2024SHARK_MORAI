#!/usr/bin/env python3


import message_filters
import os
# import ros_numpy
import math
import numpy as np


# CameraIntrintrinsicParam = [1050.9034423828125 * 640 / 1280, 6.05, 320, 0.0, 1050.9034423828125 * 640 / 1280, 240, 0.0,0.0,1.0]
# CameraIntrintrinsicParam = [1275.9034423828125 * 640 / 1280 + 150, 6.054852008819580078, 640, 0.0, 1275.9034423828125 * 640 / 1280 +150, 360, 0.0,0.0,1.0]
# Camera_Intrinsic = np.array(CameraIntrintrinsicParam).reshape(3,3)

# CamearLidarTranslation = [0.0000001,0.05,-0.05]  #  카메라와 라이다와의 상대적 위치관계
CamearLidarTranslation = [2.18,0.00,-0.67]  
CameraLidarRotation = [0,0,0,0,0,0,0,0,0]

pan = 0.0
tilt = 0.0

# x_i = 0.0
# x_e = 0.0
# 
# y_i = 0.0
# y_e = 0.0

params_cam = {
    "WIDTH": 1080, # image width
    "HEIGHT": 720, # image height
    "FOV":90, # Field of view
    # "X": 0.4, # meter
    # "Y": -0.0 ,
    # "Z": -0.35,
    # "YAW": -0.015, # deg
    # "PITCH": 0.03,
    # "ROLL": 0.0
}



def ExtrinsicMat(pan,tilt,CamearLidarTranslation):
# def ExtrinsicMat(Rotation,CamearLidarTranslation):

    cosP = math.cos(pan*math.pi/180)
    sinP = math.sin(pan*math.pi/180)
    cosT = math.cos(tilt*math.pi/180)
    sinT = math.sin(tilt*math.pi/180)
    
    RotMat = np.array([[sinP, -cosP,0],[sinT*cosP, sinT*sinP,-cosT],[cosT*cosP,cosT*sinP,sinT]]) #u,v,a축
    # RotMat = eulerAnglesToRotationMatrix(Rotation)
    
    CamLidTr = np.array(CamearLidarTranslation).reshape(3,1)    
    Rot_Tr = np.hstack((RotMat, CamLidTr))
    np_zero = np.array([0,0,0,1])
    Extrinsic_param = np.vstack((Rot_Tr, np_zero))    
    
    return Extrinsic_param

# def Cam_Intrinsic(Camera_Intrinsic):
    # np_zero = np.array([[0,0,0]]).T
    # Intrinsic_param = np.hstack((Camera_Intrinsic, np_zero))
    # return Camera_Intrinsic

def getCameraMat(params_cam):
    focalLength = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
    principalX = params_cam["WIDTH"]/2 #주점 >>> 일반적으로 이미지의 중심
    principalY = params_cam["HEIGHT"]/2 # 주점 >>> 일반적으로 이미지의 중심
    CameraMat = np.array([focalLength,0.,principalX,0,focalLength,principalY,0,0,1]).reshape(3,3)
    return CameraMat