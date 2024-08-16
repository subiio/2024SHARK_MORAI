#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2DArray
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import Bool

# 기존의 파라미터 정의 부분 유지

# 기존의 함수 정의 부분 유지



parameters_cam = {
    "WIDTH": 640, # image width
    "HEIGHT": 480, # image height
    "FOV": 90, # Field of view
    "X": 3.6, # meter
    "Y": 0,
    "Z":  0.58,
    "YAW": 0, # radian
    "PITCH": 0.0,
    "ROLL": 0
}
parameters_lidar = {
    "X": 1.42, # meter
    "Y": 0,
    "Z": 1.25,
    "YAW": 0, # - 7.5*math.pi/180.0, # radian
    "PITCH": 0,
    "ROLL": 0
}
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
    
    rotMat = rotYaw@rotPitch@rotRoll
    
    return rotMat


def getTransformMat(params_cam, params_lidar):
    #With Respect to Vehicle ISO Coordinate
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
    print(Tr_lidar_to_cam)
    return Tr_lidar_to_cam


def getCameraMat(params_cam):
    # Camera Intrinsic Parameters
    focalLength = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
    principalX = params_cam["WIDTH"]/2
    principalY = params_cam["HEIGHT"]/2
    CameraMat = np.array([focalLength,0.,principalX,0,focalLength,principalY,0,0,1]).reshape(3,3)
    print(CameraMat)
    return CameraMat

class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):        
        self.scan_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.scan_callback)
        # self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.img_callback)
        self.image_sub = rospy.Subscriber("/yolo_image", Image, self.img_callback)
        # self.bbox_sub = rospy.Subscriber("/bbox", Detection2DArray, self.bbox_callback)
        self.bbox_sub = rospy.Subscriber("/yolo_result", YoloResult, self.bbox_callback)
        self.cvbridge = CvBridge()
        self.pc_np = None
        self.img = None
        self.bboxes = []
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        self.TransformMat = getTransformMat(params_cam, params_lidar)
        self.CameraMat = getCameraMat(params_cam)
        ############### estop############################
        # self.estop_pub = rospy.Publisher("/estop", Bool, queue_size=10)
        # self.stop_sign = False

    def img_callback(self, msg):
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img = self.cvbridge.imgmsg_to_cv2(msg, "bgr8")
    def scan_callback(self, msg):
        point_list = []        
        for point in pc2.read_points(msg, skip_nans=True):      
            point_list.append((point[0], point[1], point[2], 1))
        self.pc_np = np.array(point_list, np.float32)        

    def bbox_callback(self, msg):
        self.detection = msg.detections.detections
        self.bboxes = [(detection.bbox.center.x - detection.bbox.size_x/2,  # 중심좌표
                        detection.bbox.center.y - detection.bbox.size_y/2,  # 
                        detection.bbox.size_x,  # w, h
                        detection.bbox.size_y,
                        detection.results[0].id) for detection in self.detection]

    def transformLiDARToCamera(self, pc_lidar):
        cam_temp = self.TransformMat.dot(pc_lidar)
        cam_temp = np.delete(cam_temp, 3, axis=0)
        return cam_temp
    
    def transformCameraToImage(self, pc_camera):
        cam_temp = self.CameraMat.dot(pc_camera)
        cam_temp = np.delete(cam_temp,np.where(cam_temp[2,:]<0),axis=1)
        cam_temp /= cam_temp[2,:]
        cam_temp = np.delete(cam_temp,np.where(cam_temp[0,:]>self.width),axis=1)
        cam_temp = np.delete(cam_temp,np.where(cam_temp[1,:]>self.height),axis=1)
        return cam_temp
    
    def filter_points_in_bbox(self, xy_i):
        # bbox에 포함된 포인트만 필터링
        filtered_points = []
        for bbox in self.bboxes:
            x_min, y_min, w, h,id = bbox
            x_max, y_max = x_min + w, y_min + h
            mask = (xy_i[0,:] >= x_min) & (xy_i[0,:] <= x_max) & (xy_i[1,:] >= y_min) & (xy_i[1,:] <= y_max)
            filtered_points.append(xy_i[:, mask])
        if len(filtered_points) > 0:
            # print(f'{id} , 탐지된 filter: {len(filtered_points)}')
            # self.stop_sign = True
            # self.estop_pub.publish(self.stop_sign)
            # print("stop!!!!!")
            
            return np.concatenate(filtered_points, axis=1)
        else:
            self.stop_sign = False
            # self.estop_pub.publish(self.stop_sign)
            # print("googogogo")
            return np.array([[], [], []], dtype=int)
        

def draw_pts_img(img, xi, yi):
    point_np = img    
    for ctr in zip(xi, yi):
        point_np = cv2.circle(point_np, ctr, 2, (0,255,0),-1)
    return point_np


if __name__ == '__main__':    
    rospy.init_node('ex_calib', anonymous=True)
    Transformer = LiDARToCameraTransform(parameters_cam, parameters_lidar)
    time.sleep(1)
    rate = rospy.Rate(10)
    estop = False
    estop_pub =rospy.Publisher("/estop", Bool, queue_size=10)
    while not rospy.is_shutdown():

        xyz_p = Transformer.pc_np[:, 0:3]
        xyz_p = np.insert(xyz_p,3,1,axis=1).T
        xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]<0),axis=1)
        xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]>12),axis=1)
        xyz_p = np.delete(xyz_p,np.where(xyz_p[2,:]<-1.4),axis=1) #Ground Filter

        xyz_c = Transformer.transformLiDARToCamera(xyz_p)
        
        xy_i = Transformer.transformCameraToImage(xyz_c)

        # bbox 영역 내의 포인트만 필터링
        xy_i_filtered = Transformer.filter_points_in_bbox(xy_i)

        if len(xy_i_filtered[0]) == 0 :
            estop_pub.publish(False)
            print("gogoogogog")
        else :
            print("stop!!!!")
            estop_pub.publish(True)

        xy_i_filtered = xy_i_filtered.astype(np.int32)
        projectionImage = draw_pts_img(Transformer.img, xy_i_filtered[0,:], xy_i_filtered[1,:])
                                            
        cv2.imshow("LidartoCameraProjection", projectionImage)
        cv2.waitKey(1)
