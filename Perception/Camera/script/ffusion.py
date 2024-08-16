#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import threading
import rospy
import message_filters
import cv2
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
import time

# 초기화
CV_BRIDGE = CvBridge()
params_cam = {
    "WIDTH": 640,
    "HEIGHT": 480,
    "FOV": 78
}
real_K_MAT = [640, 0, 320, 0, 480, 240, 0, 0, 1]
g_cmd = {
    'rvec': [0.0, 0.0, 0.0],
    't_mat': [2.18, 0, -0.67],
    'focal_length': [636.4573730956954, 667.7077677609984]
}
params_lidar = {
    "X": 0.0,
    "Y": 0.0,
    "Z": 0.0,
    "YAW": 0.0,
    "PITCH": 0.0,
    "ROLL": 0.0
}

def getCameraMat(params_cam):
    focalLength = params_cam["WIDTH"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
    principalX = params_cam["WIDTH"] / 2
    principalY = params_cam["HEIGHT"] / 2
    CameraMat = np.array([focalLength, 0., principalX, 0, focalLength, principalY, 0, 0, 1]).reshape(3, 3)
    return CameraMat

def camermatChanger():
    real_K_MAT[0] = g_cmd["focal_length"][0]
    real_K_MAT[4] = g_cmd["focal_length"][1]
    realCameraMat = np.array(real_K_MAT).reshape(3, 3)
    return realCameraMat

def getRotMat(RPY):
    cosR = math.cos(RPY[0])
    cosP = math.cos(RPY[1])
    cosY = math.cos(RPY[2])
    sinR = math.sin(RPY[0])
    sinP = math.sin(RPY[1])
    sinY = math.sin(RPY[2])
    
    rotRoll = np.array([1, 0, 0, 0, cosR, -sinR, 0, sinR, cosR]).reshape(3, 3)
    rotPitch = np.array([cosP, 0, sinP, 0, 1, 0, -sinP, 0, cosP]).reshape(3, 3)
    rotYaw = np.array([cosY, -sinY, 0, sinY, cosY, 0, 0, 0, 1]).reshape(3, 3)
    
    rotMat = rotYaw.dot(rotPitch.dot(rotRoll))
    return rotMat

def matrixChanger(params_cam, params_lidar):
    lidarPosition = np.array([params_lidar.get(i) for i in ["X", "Y", "Z"]])
    camPosition = np.array([g_cmd["t_mat"][i] for i in range(3)])
    lidarRPY = np.array([params_lidar.get(i) for i in ["ROLL", "PITCH", "YAW"]])
    camRPY = np.array([g_cmd["rvec"][i] * math.pi / 180 for i in range(3)])
    camRPY = camRPY + np.array([-90 * math.pi / 180, 0, -90 * math.pi / 180])

    camRot = getRotMat(camRPY)
    camTransl = np.array([camPosition])
    Tr_cam_to_vehicle = np.concatenate((camRot, camTransl.T), axis=1)
    Tr_cam_to_vehicle = np.insert(Tr_cam_to_vehicle, 3, values=[0, 0, 0, 1], axis=0)

    lidarRot = getRotMat(lidarRPY)
    lidarTransl = np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.concatenate((lidarRot, lidarTransl.T), axis=1)
    Tr_lidar_to_vehicle = np.insert(Tr_lidar_to_vehicle, 3, values=[0, 0, 0, 1], axis=0)

    invTr = inv(Tr_cam_to_vehicle)
    Tr_lidar_to_cam = invTr.dot(Tr_lidar_to_vehicle).round(6)

    return Tr_lidar_to_cam

def transformLiDARToCamera(TransformMat, pc_lidar):
    cam_temp = TransformMat.dot(pc_lidar)
    cam_temp = np.delete(cam_temp, 3, axis=0)
    return cam_temp

def transformCameraToImage(width, height, CameraMat, pc_camera, pc_lidar):
    img_temp = CameraMat.dot(pc_camera)
    img_temp /= img_temp[2, :]
    return img_temp, pc_camera, pc_lidar

def draw_pts_img(img, points, distances):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normalized_distances = distances / np.max(distances)

    for ctr, distance in zip(points, normalized_distances):
        color = (int(255 * (1 - distance)), 0, int(255 * distance))
        cv2.circle(img, ctr, 2, color, -1)

    return img

def calc_distance_position2(points):
    mat_points = np.array(points).T
    position = []
    index = np.argmin(mat_points[0])
    position.append(min(mat_points[0]))
    position.append(mat_points[1][index])
    position.append(mat_points[2][index])
    tmp_position = np.array(position)
    dist = math.sqrt(tmp_position.dot(tmp_position))
    return dist, position

def handle_key_input(key):
    global g_cmd, g_cmd_toggle, fl_x, fl_y
    if key == 49: g_cmd_toggle = ['rvec', 0]  # 1
    if key == 50: g_cmd_toggle = ['rvec', 1]  # 2
    if key == 51: g_cmd_toggle = ['rvec', 2]  # 3

    if key == 52: g_cmd_toggle = ['t_mat', 0]  # 4
    if key == 53: g_cmd_toggle = ['t_mat', 1]  # 5
    if key == 54: g_cmd_toggle = ['t_mat', 2]  # 6
    
    if key == 113: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] += 0.005  # q
    if key == 119: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] += 0.05  # w
    if key == 101: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] += 0.5  # e

    if key == 97: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] -= 0.005  # a
    if key == 115: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] -= 0.05  # s
    if key == 100: g_cmd[g_cmd_toggle[0]][g_cmd_toggle[1]] -= 0.5  # d

    if key == 114: fl_x += 0.01
    if key == 102: fl_x -= 0.01
  
    if key == 116: fl_y += 0.01
    if key == 103: fl_y -= 0.01

    g_cmd["focal_length"] = [636.4573730956954 * fl_x, 667.7077677609984 * fl_y]

def key_listener():
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            key = key % 256
            handle_key_input(key)

def callback(ouster, image, dynamicObejct, pcd_pub=None):
    print("-----------sync is working!")
    start_time = time.time()
    global g_cmd

    width = params_cam["WIDTH"]
    height = params_cam["HEIGHT"]

    TransformMat = matrixChanger(params_cam, params_lidar)
    CameraMat = getCameraMat(params_cam)
    RealCameraMat = camermatChanger()
    print(img.size())

    try:
        print("cv_br is working")
        img = CV_BRIDGE.compressed_imgmsg_to_cv2(image, "bgr8")
        if img is None:
            rospy.logerr("Failed to decode CompressedImage")
            return
        mask = np.zeros_like(img)
        rospy.loginfo("Image converted to CV2 format")
        
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))
        return

    point_list = []
    distance_list = []
    for point in pc2.read_points(ouster, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        angle = math.atan2(point[0], point[1]) * 180 / math.pi
        if 0 < angle < 180:
            point_list.append((point[0], point[1], point[2], 1))
            distance_list.append(math.sqrt(point[0]**2 + point[1]**2 + point[2]**2))

    pc_np = np.array(point_list, np.float32)
    distances = np.array(distance_list, np.float32)

    box_list = []
    for detection in dynamicObejct.detections:
        center_x = detection.bbox.center.x
        center_y = detection.bbox.center.y
        left_x = int(center_x - (detection.bbox.size_x / 2.0))
        left_y = int(center_y - (detection.bbox.size_y / 2.0))
        right_x = int(center_x + (detection.bbox.size_x / 2.0))
        right_y = int(center_y + (detection.bbox.size_y / 2.0))
        bbox_dict = {
            'class': "cone",
            'center_point': [center_x, center_y],
            'left_point': [left_x, left_y],
            'right_point': [right_x, right_y],
            'id': detection.results[0].score
        }
        box_list.append(bbox_dict)

    filtered_xyz_p = pc_np[:, 0:3].T
    xyz_p = np.insert(pc_np[:, 0:3], 3, 1, axis=1).T

    xyz_c = transformLiDARToCamera(TransformMat, xyz_p)
    xy_i, filtered_xyz_c, filtered_xyz_p = transformCameraToImage(width, height, RealCameraMat, xyz_c, filtered_xyz_p)

    mat_xyz_p = filtered_xyz_p.T
    mat_xyz_c = filtered_xyz_c.T
    mat_xy_i = xy_i.T
    xy_i = xy_i.astype(np.int32)

    img = draw_pts_img(img, zip(xy_i[0, :], xy_i[1, :]), distances)

    pd_list = MarkerArray()
    for i, box in enumerate(box_list):
        inner_3d_point = []

        scale_x = int(box['right_point'][0]) - int(box['left_point'][0])
        scale_y = int(box['right_point'][1]) - int(box['left_point'][1])
        cv2.rectangle(img, (int(box['left_point'][0]), int(box['left_point'][1])), (int(box['right_point'][0]), int(box['right_point'][1])), (0, 0, 255), 2)

        for k, xy in enumerate(mat_xy_i):
            if (box['left_point'][0] + 0.45 * scale_x) < xy[0] < (box['right_point'][0] - 0.45 * scale_x) and (box['left_point'][1] + 0.1 * scale_y) < xy[1] < (box['right_point'][1] - 0.5 * scale_y):
                inner_3d_point.append(mat_xyz_p[k].tolist())

        if inner_3d_point:
            dist, position = calc_distance_position2(inner_3d_point)
            tmp_pd = Marker()

            tmp_pd.lifetime = rospy.Duration(0.1)
            tmp_pd.header.frame_id = "base_link"
            tmp_pd.header.stamp = rospy.Time.now()
            tmp_pd.pose.position.x = position[0]
            tmp_pd.pose.position.y = position[1]
            tmp_pd.pose.position.z = position[2]

            tmp_pd.pose.orientation.x = box['center_point'][0]
            tmp_pd.pose.orientation.y = box['center_point'][1]
            tmp_pd.pose.orientation.z = scale_x
            tmp_pd.pose.orientation.w = scale_y

            tmp_pd.text = box['class']

            tmp_pd.scale.x = 0.2
            tmp_pd.scale.y = 0.3
            tmp_pd.scale.z = 1.0
            tmp_pd.color.r = 1.0
            tmp_pd.color.g = 0.0
            tmp_pd.color.b = 0.0
            tmp_pd.color.a = 1.0
            tmp_pd.type = Marker.CUBE
            tmp_pd.action = Marker.ADD

            pd_list.markers.append(tmp_pd)

    pcd_pub.publish(pd_list)

    print("--------------------------------------------------------------------------\n\n")
    print(f"rvec = {g_cmd['rvec']} \n")
    print(f"t_mat = {g_cmd['t_mat']} \n")
    print(f"focal_length = {g_cmd['focal_length']} \n")
    print("--------------------------------------------------------------------------\n\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    image_msg = CV_BRIDGE.cv2_to_imgmsg(img, "bgr8")

    # Image 메시지를 다시 CompressedImage 메시지로 변환
    compressed_image_msg = CompressedImage()
    compressed_image_msg.format = "jpeg"
    compressed_image_msg.data = np.array(cv2.imencode('.jpg', image_msg)[1]).tostring()

    # 변환된 CompressedImage 메시지를 새 토픽으로 발행
    marker_img_pub.publish(compressed_image_msg)

    cv2.namedWindow("LidartoCameraProjection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LidartoCameraProjection", 1280, 960)
    cv2.imshow("LidartoCameraProjection", img)
    cv2.waitKey(1)

def static_bbox_callback(msg):
    # static_bbox 메시지의 시간 정보를 출력합니다.
    rospy.loginfo(f"Static BBox Time: {msg.header.stamp}")

def image_callback(msg):
    # image 메시지의 시간 정보를 출력합니다.
    rospy.loginfo(f"Image Time: {msg.header.stamp}")

def velodyne_callback(msg):
    # velodyne 메시지의 시간 정보를 출력합니다.
    rospy.loginfo(f"Velodyne Time: {msg.header.stamp}")


def listener(image_color, ouster, static_bbox):
    rospy.init_node('fusion_camera_lidar', anonymous=True)

    staticObejct = message_filters.Subscriber(static_bbox, Detection2DArray)
    image_sub = message_filters.Subscriber(image_color, CompressedImage)
    velodyne_sub = message_filters.Subscriber(ouster, PointCloud2)
    pcd_pub = rospy.Publisher('/static_obstacles', MarkerArray, queue_size=10)
    marker_img_pub = rospy.Publisher('/marker_img', CompressedImage, queue_size=10)

    ats = message_filters.ApproximateTimeSynchronizer(
        [velodyne_sub, image_sub, staticObejct], queue_size=50, slop=0.1)
    ats.registerCallback(callback, pcd_pub)

    rospy.Subscriber(static_bbox, Detection2DArray, static_bbox_callback)
    rospy.Subscriber(image_color, CompressedImage, image_callback)
    rospy.Subscriber(ouster, PointCloud2, velodyne_callback)


    key_thread = threading.Thread(target=key_listener)
    key_thread.daemon = True
    key_thread.start()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')

if __name__ == '__main__':
    ouster = '/velodyne_points'
    static_bbox = '/bbox'
    image_color = '/image_jpeg/compressed/image'
    listener(image_color, ouster, static_bbox)
