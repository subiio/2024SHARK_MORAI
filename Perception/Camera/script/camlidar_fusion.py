#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from ultralytics_ros.msg import YoloResult

# 카메라와 LiDAR 사이의 변환 및 회전 파라미터
CamearLidarTranslation = [2.18, 0.00, -0.67]
CameraLidarRotation = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 여기서는 회전이 없는 경우로 가정

pan = 0.0
tilt = 0.0

params_cam = {
    "WIDTH": 640,  # image width
    "HEIGHT": 480,  # image height
    "FOV": 90,  # Field of view
}

# 카메라 내부 파라미터 계산 함수
def getCameraMat(params_cam):
    focalLength_x = params_cam["WIDTH"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
    focalLength_y = params_cam["HEIGHT"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
    principalX = params_cam["WIDTH"] / 2  # 주점
    principalY = params_cam["HEIGHT"] / 2  # 주점
    CameraMat = np.array([focalLength_x, 0., principalX, 0, focalLength_y, principalY, 0, 0, 1]).reshape(3, 3)
    return CameraMat

# 카메라 외부 파라미터 계산 함수
def ExtrinsicMat(pan, tilt, CamearLidarTranslation):
    cosP = math.cos(pan * math.pi / 180)
    sinP = math.sin(pan * math.pi / 180)
    cosT = math.cos(tilt * math.pi / 180)
    sinT = math.sin(tilt * math.pi / 180)
    
    RotMat = np.array([
        [cosP, -sinP, 0],
        [sinP * sinT, cosP * sinT, -cosT],
        [sinP * cosT, cosP * cosT, sinT]
    ])
    
    CamLidTr = np.array(CamearLidarTranslation).reshape(3, 1)
    Rot_Tr = np.hstack((RotMat, CamLidTr))
    np_zero = np.array([0, 0, 0, 1])
    Extrinsic_param = np.vstack((Rot_Tr, np_zero))
    
    return Extrinsic_param

class LidarToImageProjector:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 카메라 내부 파라미터 설정
        self.camera_intrinsics = getCameraMat(params_cam)

        # 카메라 외부 파라미터 설정
        self.camera_extrinsics = ExtrinsicMat(pan, tilt, CamearLidarTranslation)

        rospy.init_node('lidar_to_image_projector', anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.point_cloud_callback)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("/yolo_result", YoloResult, self.detection_callback)  # YoloResult 메시지 구독
        
        self.pcd = None
        self.image = None
        self.detections = None

    def point_cloud_callback(self, msg):
        try:
            self.pcd = []
            for point in point_cloud2.read_points(msg, skip_nans=True):
                self.pcd.append([point[0], point[1], point[2]])
            self.pcd = np.array(self.pcd)
            rospy.loginfo("PointCloud data received and processed.")

            if self.image is not None and self.detections is not None:
                self.project_and_display()
        except Exception as e:
            rospy.logerr(f"Error in point_cloud_callback: {e}")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if self.image is None:
                rospy.logerr("Image data is empty or corrupted.")
            else:
                rospy.loginfo("Image data received and processed.")

            if self.pcd is not None and self.detections is not None:
                self.project_and_display()
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def detection_callback(self, msg):
        try:
            self.detections = msg.detections.detections  # Detection2DArray 객체 내의 detections 리스트
            rospy.loginfo("Detection data received and processed.")

            if self.image is not None and self.pcd is not None:
                self.project_and_display()
        except Exception as e:
            rospy.logerr(f"Error in detection_callback: {e}")

    def project_and_display(self):
        try:
            if self.image is None:
                rospy.logerr("Image is not available for projection.")
                return

            if self.pcd.size == 0:
                rospy.logerr("PointCloud data is not available for projection.")
                return

            pcd_camera_coords = self.camera_extrinsics[:3, :3] @ self.pcd.T + self.camera_extrinsics[:3, 3:]
            pcd_image_plane = self.camera_intrinsics @ pcd_camera_coords
            pcd_image_plane /= pcd_image_plane[2, :]

            points_2d = pcd_image_plane[:2, :].T

            for point in points_2d:
                cv2.circle(self.image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

            # 디텍션 결과를 이미지에 오버레이
            for detection in self.detections:
                bbox = detection.bbox
                x = int(bbox.center.x - bbox.size_x / 2)
                y = int(bbox.center.y - bbox.size_y / 2)
                w = int(bbox.size_x)
                h = int(bbox.size_y)
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                for result in detection.results:
                    label = f"ID: {result.id}, Score: {result.score:.2f}"
                    cv2.putText(self.image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("LiDAR Projection with Detections", self.image)
            cv2.waitKey(1)
            rospy.loginfo("Projection and display completed.")
        except Exception as e:
            rospy.logerr(f"Error in project_and_display: {e}")
        finally:
            cv2.destroyAllWindows()

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        projector = LidarToImageProjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
