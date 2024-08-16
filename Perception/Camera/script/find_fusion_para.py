#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import message_filters
import threading
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import math
import sensor_msgs.point_cloud2 as pc2

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
    't_mat': [0, 0, 0],
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
    lidarPosition = np.array([params_lidar.get(i) for i in (["X", "Y", "Z"])])
    camPosition = np.array([g_cmd["t_mat"][i] for i in range(3)])
    lidarRPY = np.array([params_lidar.get(i) for i in (["ROLL", "PITCH", "YAW"])])
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

    invTr = np.linalg.inv(Tr_cam_to_vehicle)
    Tr_lidar_to_cam = invTr.dot(Tr_lidar_to_vehicle).round(6)

    return Tr_lidar_to_cam

def draw_pts_img(img, points, distances):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
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

def callback(ouster, image, dynamicObejct, pcd_pub=None):
    global CV_BRIDGE, params_cam

    width = params_cam["WIDTH"]
    height = params_cam["HEIGHT"]

    TransformMat = matrixChanger(params_cam, params_lidar)
    CameraMat = getCameraMat(params_cam)
    RealCameraMat = camermatChanger()

    try:
        np_arr = np.frombuffer(image.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logerr("Failed to decode CompressedImage")
            return
        mask = np.zeros_like(img)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))
        return

    rospy.loginfo("Image and point cloud received")

    point_list = []
    distance_list = []
    for point in pc2.read_points(ouster, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        point_list.append((point[0], point[1], point[2], 1))
        distance_list.append(math.sqrt(point[0]**2 + point[1]**2 + point[2]**2))

    rospy.loginfo(f"Point list size: {len(point_list)}")

    pc_np = np.array(point_list, np.float32)
    distances = np.array(distance_list, np.float32)

    # 좌표 변환 및 사영
    filtered_xyz_p = pc_np[:, 0:3]
    filtered_xyz_p = filtered_xyz_p.T
    xyz_p = pc_np[:, 0:3]
    xyz_p = np.insert(xyz_p, 3, 1, axis=1).T

    xyz_c = np.dot(TransformMat, xyz_p)
    xy_i, filtered_xyz_c, filtered_xyz_p = np.dot(RealCameraMat, xyz_c[:3]), xyz_c[:3], filtered_xyz_p

    rospy.loginfo(f"Transformed points: {xyz_c.shape}")

    mat_xyz_p = filtered_xyz_p.T
    mat_xyz_c = filtered_xyz_c.T
    mat_xy_i = xy_i.T
    xy_i = xy_i.astype(np.int32)

    img = draw_pts_img(img, zip(xy_i[0, :], xy_i[1, :]), distances)

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

    filtered_xyz_p = pc_np[:, 0:3]
    filtered_xyz_p = filtered_xyz_p.T
    xyz_p = pc_np[:, 0:3]
    xyz_p = np.insert(xyz_p, 3, 1, axis=1).T

    xyz_c = np.dot(TransformMat, xyz_p)
    xy_i, filtered_xyz_c, filtered_xyz_p = np.dot(RealCameraMat, xyz_c[:3]), xyz_c[:3], filtered_xyz_p

    mat_xyz_p = filtered_xyz_p.T
    mat_xyz_c = filtered_xyz_c.T
    mat_xy_i = xy_i.T
    xy_i = xy_i.astype(np.int32)

    img = draw_pts_img(img, zip(xy_i[0, :], xy_i[1, :]), distances)

    pd_list = MarkerArray()
    for i, box in enumerate(box_list):
        inner_3d_point = []
        scale_x = (int(box['right_point'][0]) - int(box['left_point'][0]))
        scale_y = (int(box['right_point'][1]) - int(box['left_point'][1]))
        cv2.rectangle(img, (int(box['left_point'][0]), int(box['left_point'][1])), (int(box['right_point'][0]), int(box['right_point'][1])), (0, 0, 255), 2)

        for k, xy in enumerate(mat_xy_i):
            if xy[0] > (box['left_point'][0] + 0.45 * scale_x) and xy[0] < (box['right_point'][0] - 0.45 * scale_x) and xy[1] > (box['left_point'][1] + 0.1 * scale_y) and xy[1] < (box['right_point'][1] - 0.5 * scale_y):
                inner_3d_point.append(mat_xyz_p[k].tolist())

        if len(inner_3d_point) != 0:
            dist, position = calc_distance_position2(inner_3d_point)
            tmp_pd = Marker()
            tmp_pd.lifetime = rospy.Duration(0.1)
            tmp_pd.header.frame_id = "/base_link"
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

    rospy.loginfo(f"Processed frame with {len(box_list)} detections")
    cv2.namedWindow("LidartoCameraProjection", cv2.WINDOW_NORMAL)  # 창 크기 조정 가능하도록 설정
    cv2.resizeWindow("LidartoCameraProjection", 1280, 960)  # 원하는 크기로 창 조정
    cv2.imshow("LidartoCameraProjection", img)
    cv2.waitKey(1)

def key_listener():
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            key = key % 256
            handle_key_input(key)

def listener(image_color, ouster, static_bbox):
    rospy.init_node('fusion_camera_lidar', anonymous=True)

    staticObejct = message_filters.Subscriber(static_bbox, Detection2DArray)
    image_sub = message_filters.Subscriber(image_color, CompressedImage)
    velodyne_sub = message_filters.Subscriber(ouster, PointCloud2)
    pcd_pub = rospy.Publisher('/static_obstacles', MarkerArray, queue_size=10)

    ats = message_filters.ApproximateTimeSynchronizer(
        [velodyne_sub, image_sub, staticObejct], queue_size=5, slop=0.05)
    ats.registerCallback(callback, pcd_pub)

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
    image_color = '/image_jpeg/compressed'
    listener(image_color, ouster, static_bbox)
