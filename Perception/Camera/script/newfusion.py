#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import math
import time
import multiprocessing

# External modules
import cv2
import numpy as np
from numpy.linalg import inv

# ROS module
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage

params_lidar = {
    "X": 0.0, # meterx``
    "Y": 0.0,
    "Z": 0.0,
    "YAW": 0.0, # deg
    "PITCH": 0.0,
    "ROLL": 0.0
}

params_cam = {
    "WIDTH":  640, # image width
    "HEIGHT": 480, # image height
    "FOV": 90, # Field of view
    "X": 2.18, # meter
    "Y": 0,
    "Z": -0.67,
    "YAW": 0.00, # deg
    "PITCH": 0.00,
    "ROLL": 0.0
}

real_K_MAT = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

CV_BRIDGE = CvBridge()

def getRotMat(RPY):
    cosR = math.cos(RPY[0])
    cosP = math.cos(RPY[1])
    cosY = math.cos(RPY[2])
    sinR = math.sin(RPY[0])
    sinP = math.sin(RPY[1])
    sinY = math.sin(RPY[2])
    
    rotRoll = np.array([1,0,0, 0,cosR,-sinR, 0,sinR,cosR]).reshape(3,3)
    rotPitch = np.array([cosP,0,sinP, 0,1,0, -sinP,0,cosP]).reshape(3,3)
    rotYaw = np.array([cosY,-sinY,0, sinY,cosY,0, 0,0,1]).reshape(3,3)
    
    rotMat = rotYaw.dot(rotPitch.dot(rotRoll))
    return rotMat

def getTransformMat(params_lidar, params_cam):
    lidarPosition = np.array([params_lidar.get(i) for i in (["X","Y","Z"])])
    camPosition = np.array([params_cam.get(i) for i in (["X","Y","Z"])])

    lidarRPY = np.array([params_lidar.get(i) for i in (["ROLL","PITCH","YAW"])])
    camRPY = np.array([params_cam.get(i) for i in (["ROLL","PITCH","YAW"])])
    camRPY = camRPY + np.array([-90*math.pi/180,0,-90*math.pi/180])

    camRot = getRotMat(camRPY)
    camTransl = np.array([camPosition])
    Tr_cam_to_vehicle = np.concatenate((camRot,camTransl.T),axis = 1)
    Tr_cam_to_vehicle = np.insert(Tr_cam_to_vehicle, 3, values=[0,0,0,1],axis = 0)

    lidarRot = getRotMat(lidarRPY)
    lidarTransl = np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.concatenate((lidarRot,lidarTransl.T),axis = 1)
    Tr_lidar_to_vehicle = np.insert(Tr_lidar_to_vehicle, 3, values=[0,0,0,1],axis = 0)

    invTr = inv(Tr_cam_to_vehicle)
    Tr_lidar_to_cam = invTr.dot(Tr_lidar_to_vehicle).round(6)
    return Tr_lidar_to_cam

def getCameraMat(params_cam):
    focalLength = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
    principalX = params_cam["WIDTH"]/2
    principalY = params_cam["HEIGHT"]/2
    CameraMat = np.array([focalLength,0.,principalX,0,focalLength,principalY,0,0,1]).reshape(3,3)
    return CameraMat

def getRealCameraMat():
    realCameraMat = np.array(real_K_MAT).reshape(3,3)
    return realCameraMat

def transformLiDARToCamera(TransformMat, pc_lidar):
    cam_temp = TransformMat.dot(pc_lidar)
    cam_temp = np.delete(cam_temp, 3, axis=0)
    return cam_temp

def transformCameraToImage(width, height, CameraMat, pc_camera, pc_lidar):
    cam_temp = pc_camera
    lid_temp = pc_lidar
    img_temp = CameraMat.dot(pc_camera)
    cam_temp = np.delete(cam_temp,np.where(img_temp[2,:]<0),axis=1)
    lid_temp = np.delete(lid_temp,np.where(img_temp[2,:]<0),axis=1)
    img_temp = np.delete(img_temp,np.where(img_temp[2,:]<0),axis=1)
    img_temp /= img_temp[2,:]
    cam_temp = np.delete(cam_temp,np.where(img_temp[0,:]>width),axis=1)
    lid_temp = np.delete(lid_temp,np.where(img_temp[0,:]>width),axis=1)
    img_temp = np.delete(img_temp,np.where(img_temp[0,:]>width),axis=1)
    cam_temp = np.delete(cam_temp,np.where(img_temp[1,:]>height),axis=1)
    lid_temp = np.delete(lid_temp,np.where(img_temp[1,:]>height),axis=1)
    img_temp = np.delete(img_temp,np.where(img_temp[1,:]>height),axis=1)
    cam_temp = np.delete(cam_temp,np.where(img_temp[0,:]<0),axis=1)
    lid_temp = np.delete(lid_temp,np.where(img_temp[0,:]<0),axis=1)
    img_temp = np.delete(img_temp,np.where(img_temp[0,:]<0),axis=1)
    cam_temp = np.delete(cam_temp,np.where(img_temp[1,:]<0),axis=1)
    lid_temp = np.delete(lid_temp,np.where(img_temp[1,:]<0),axis=1)
    img_temp = np.delete(img_temp,np.where(img_temp[1,:]<0),axis=1)
    return img_temp, cam_temp, lid_temp

def draw_pts_img(img, xi, yi):
    point_np = img
    for ctr in zip(xi, yi):
        cv2.circle(point_np, ctr, 2, (0,255,0),-1)
    return point_np

# all topics are processed in this callback function
def callback(velodyne, image):
    rospy.loginfo('Fusion Processing')

    width = params_cam["WIDTH"]
    height = params_cam["HEIGHT"]
    TransformMat = getTransformMat(params_cam, params_lidar)
    CameraMat = getCameraMat(params_cam)
    RealCameraMat = getRealCameraMat()

    # image callback function
    try:
        img = CV_BRIDGE.compressed_imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))

    # lidar callback function
    point_list = []
    for point in pc2.read_points(velodyne, skip_nans=True):
        point_list.append((point[0], point[1], point[2], 1))
    pc_np = np.array(point_list, np.float32)

    filtered_xyz_p = pc_np[:, 0:3]
    filtered_xyz_p = filtered_xyz_p.T
    xyz_p = pc_np[:, 0:3]
    xyz_p = np.insert(xyz_p,3,1,axis=1).T   # Transpose
    
    # filtering point cloud in front of camera
    filtered_xyz_p = np.delete(filtered_xyz_p,np.where(xyz_p[0,:]<0),axis=1)
    xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]<0),axis=1)
    filtered_xyz_p = np.delete(filtered_xyz_p,np.where(xyz_p[0,:]>10),axis=1)
    xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]>10),axis=1)
    filtered_xyz_p = np.delete(filtered_xyz_p,np.where(xyz_p[2,:]<-0.7),axis=1)
    xyz_p = np.delete(xyz_p,np.where(xyz_p[2,:]<-0.7),axis=1) #Ground Filter

    xyz_c = transformLiDARToCamera(TransformMat, xyz_p)
    xy_i, filtered_xyz_c, filtered_xyz_p = transformCameraToImage(width, height, RealCameraMat, xyz_c, filtered_xyz_p)

    xy_i = xy_i.astype(np.int32)
    projectionImage = draw_pts_img(img, xy_i[0,:], xy_i[1,:])

    try:
        cv2.imshow("LidartoCameraProjection", projectionImage)
        cv2.waitKey(1)
    except:
        pass

# practical main function
def listener(image_color, velodyne_points):
    rospy.init_node('fusion_camera_lidar', anonymous=True)
    rospy.loginfo('Current PID: [{}]'.format(os.getpid()))
    rospy.loginfo('PointCloud2 topic: {}'.format(velodyne_points))
    rospy.loginfo('Image topic: {}'.format(image_color))

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, CompressedImage)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    # Synchronize the topic by time: velodyne, image
    ats = message_filters.ApproximateTimeSynchronizer(
        [velodyne_sub, image_sub], queue_size=10, slop=0.1)
    ats.registerCallback(callback)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')

if __name__ == '__main__':
    velodyne_points = '/velodyne_points'
    image_color = '/image_jpeg/compressed'

    listener(image_color, velodyne_points)
