#!/usr/bin/env python3


###########################################################
from cali import *   # callibration 관련 함수 및 인자
from marker import * # marker 관련 함수
from follow import * # follow 관렴 함수
#############################################################

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge,CvBridgeError
#############################################################
import message_filters
import cv2
import numpy as np
import rospy

class Camera:
    def __init__(self):
        
        self.Extrinsic_param = ExtrinsicMat(0,0,CamearLidarTranslation)
        self.cameraMat = getCameraMat(params_cam)
        self.CVBRIDGE = CvBridge()

        self.wall_follow = False

        self.ouster = '/fov_cloud'
        self.segmentation = 'seg_img'
        self.image_color = '/output_img'

        rospy.loginfo('PointCloud2 topic: {}'.format(self.ouster))
        rospy.loginfo('Segmentation topic: {}'.format(self.segmentation))
        rospy.loginfo('Image topic: {}'.format(self.image_color))


        self.lane_pub = rospy.Publisher('/lane_pub',MarkerArray,queue_size=10) #차선 좌표 찍기
        self.way_pub = rospy.Publisher('/way_pub',Marker,queue_size=10)  # 나아갈 좌표 찍기

        self.seg_sub = message_filters.Subscriber(self.segmentation,Image)
        self.image_sub = message_filters.Subscriber(self.image_color,Image)
        self.ouster_sub = message_filters.Subscriber(self.ouster, PointCloud2)
        self.ats = message_filters.ApproximateTimeSynchronizer(
        [self.seg_sub, self.image_sub, self.ouster_sub], queue_size=10, slop=10) # 3개를 한 번에 서브스크라이브해서 운용할 수 있게 해주는 ..
        self.ats.registerCallback(self.callback) 
    
    def get_lidar_to_camera(self,TransformMat, pointcloud):
        Lid2cam = TransformMat.dot(pointcloud)
        Lid2cam = np.delete(Lid2cam, 3, axis=0)
        return Lid2cam
    
    def seg_img_filter(self,seg_img):
        height, width = seg_img.shape
        threshold_height = int(3/4 * height)  #k-city 가서 확인해볼것

    # Create a kernel with 1 above the threshold height and 0 below
        kernel = np.zeros((height, width), dtype=np.uint8)
        kernel[threshold_height:, :] = 1

    # Apply the kernel using bitwise AND operation
        seg_img = cv2.bitwise_and(seg_img, kernel)
        return seg_img
    

    def lidar_lane(self, seg_img, img, fixel_coord, world_xyz, wall_follow) :

        seg_img = self.seg_img_filter(seg_img)
        lane_3d_pts = []
        lanes = []
        markers = []
        fixel_coord = fixel_coord.T # 전치행렬
        for i in range(fixel_coord.shape[0]):
            if int(fixel_coord[i][1]) < 720 and int(fixel_coord[i][1]) > 0 and int(fixel_coord[i][0]) > 0 and int(fixel_coord[i][0]) < 1280:
                if seg_img[int(fixel_coord[i][1])][int(fixel_coord[i][0])] == 1 and seg_img[int(fixel_coord[i][1])][int(fixel_coord[i][0])-5] == 0:
                    cv2.circle(img, (int(fixel_coord[i][0]),int(fixel_coord[i][1])), 5, (0,0,255), -1)
                    lane_3d_pts.append([world_xyz[i][0], world_xyz[i][1]])

        right, mid = mid_right_start(lane_3d_pts)
        right.sort(key=lambda x: (x[0], -x[1])) #종방향 기준 오름차순 정리(차량 기준 가장 가까운 점부터 먼 점 순서.)
        mid.sort(key=lambda x: (x[0], -x[1]))
        if len(right) < 3:
            wall_follow = True
        print(wall_follow)
        lanes.append(right)
        lanes.append(mid)
       
        color=[[255,0,0],[0,255,0],[0,0,255]]
        for k in range(2):
            for i in range(len(lane_3d_pts)):
                for j in range(len(lanes[k])):
                    if np.sqrt((lanes[k][j][0]- lane_3d_pts[i][0])**2 + (lanes[k][j][1]- lane_3d_pts[i][1])**2) < 0.2:
                        lanes[k][j][0] = lane_3d_pts[i][0]
                        lanes[k][j][1] = lane_3d_pts[i][1]
            lane_marker = points_rviz('lane',k, lanes[k],color[k])
            markers.append(lane_marker)
        if wall_follow == False:
            if len(mid) > 2 and len(right) > 2: # 점이 충분히 있다면...
                follow_points = follow_pts(right,mid)
            elif len(right) > 0: # 오른쪽 차선의 점이 하나라도 있다면, 오른쪽 차선을 기준으로 해야 차선 바깥으로 나가는 것을 막을 수 있다.

                follow_points = follow_pts_right(right)
            else:
  
                follow_points = follow_pts_mid(mid) # 왼쪽 차선에 대한 정보밖에 없는 상황
        else: #오른쪽 차선의 점이 너무 많은 상황
            follow_points = follow_pts_right(right)

        lane_marker = points_rviz('lane',4, follow_points,[125,0,125])
        mark = marker_array_rviz(markers)
        self.lane_pub.publish(mark)
        self.way_pub.publish(lane_marker)
        img = cv2.resize(img,(640,480))
        cv2.imshow('lane',img)
        cv2.waitKey(1)


    def callback(self, seg_img, image, ouster):
        try:
            self.img = self.CVBRIDGE.imgmsg_to_cv2(image, 'bgr8')
            self.seg_img = self.CVBRIDGE.imgmsg_to_cv2(seg_img, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

        try:
            world = []
            for n in point_cloud2.read_points(ouster, skip_nans=True): # point cloud를 통해 찍힌 점 인식
                world.append((n[0], n[1], n[2],1))

            world_xyz = np.array(world, np.float32).T # 전치행렬, 라이다 포인터들

            self.Ext = self.get_lidar_to_camera(self.Extrinsic_param, world_xyz)
            self.fixel_coord = self.cameraMat.dot(self.Ext)
            self.fixel_coord /= self.fixel_coord[2,:] # z축을 다 나눠줌  xyz 순서 , 그래야 2d 처럼 활용가능 
            self.lidar_lane(self.seg_img,self.img,self.fixel_coord, world_xyz.T,self.wall_follow)
        except:
            pass

def run():
    rospy.init_node('fusion_camera_lidar', anonymous=True)
    camera = Camera()
    rospy.spin()


if __name__ == '__main__':
    run()