import rospy
from sensor_msgs.msg import CameraInfo
import math

class CameraInfoPublisher:
    def __init__(self):
        # 노드 핸들러 초기화
        self.nh = rospy.init_node('camera_info_publisher', anonymous=True)
        
        # 카메라 정보 메시지 초기화
        self.camera_info_msg = CameraInfo()

        # 카메라 파라미터 설정
        self.camera_info_msg.header.frame_id = rospy.get_param('~camera_frame_id', 'camera_frame')
        self.camera_info_msg.width = rospy.get_param('~width', 640)
        self.camera_info_msg.height = rospy.get_param('~height', 480)

        # FOV를 사용하여 초점 거리 계산
        fov = 90.0  # FOV in degrees
        self.fx = self.camera_info_msg.width / (2 * math.tan(math.radians(fov) / 2))
        self.fy = self.fx  # Assuming square pixels
        self.cx = self.camera_info_msg.width / 2.0
        self.cy = self.camera_info_msg.height / 2.0

        # 카메라 매트릭스 설정
        self.camera_info_msg.K[0] = self.fx
        self.camera_info_msg.K[2] = self.cx
        self.camera_info_msg.K[4] = self.fy
        self.camera_info_msg.K[5] = self.cy
        self.camera_info_msg.K[8] = 1.0

        # 투영 행렬 설정
        self.camera_info_msg.P[0] = self.fx
        self.camera_info_msg.P[2] = self.cx
        self.camera_info_msg.P[5] = self.fy
        self.camera_info_msg.P[6] = self.cy
        self.camera_info_msg.P[10] = 1.0

        # 왜곡 계수 초기화 (기본적으로 0으로 설정)
        self.camera_info_msg.D = [0.0] * 5

        # 퍼블리셔 생성
        self.camera_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)

        # 주기적으로 CameraInfo 메시지를 퍼블리시
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.publish_camera_info)

    def publish_camera_info(self, event):
        self.camera_info_msg.header.stamp = rospy.Time.now()
        self.camera_info_pub.publish(self.camera_info_msg)

if __name__ == '__main__':
    CameraInfoPublisher()
    rospy.spin()
