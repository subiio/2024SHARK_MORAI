#! /usr/bin/env python3


import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

class TrafficLightDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        self.traffic_pub = rospy.Publisher("/traffic_sign", Bool, queue_size=1)
        self.latest_frame = None

    def image_callback(self, msg):
        try:
            # Convert the compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Define ROI (Region of Interest)
            x_start, x_end = 440, 640
            y_start, y_end = 200, 320
            self.latest_frame = image[y_start:y_end, x_start:x_end]

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def process_image(self):
        while not rospy.is_shutdown():
            if self.latest_frame is not None:
                roi_img = self.latest_frame.copy()
                
                # Fixed HSV values
                lower_hsv = np.array([0, 158, 181])
                upper_hsv = np.array([179, 255, 255])

                # Convert the ROI image to HSV
                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

                # Threshold the HSV image to get only the colors in the specified range
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                result = cv2.bitwise_and(roi_img, roi_img, mask=mask)

                # Display the original, mask, and result images
                cv2.imshow('ROI', roi_img)
                cv2.imshow('Mask', mask)
                cv2.imshow('Result', result)

                # Detect traffic light color
                self.detect_traffic_light(result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def detect_traffic_light(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red1 = np.array([0, 158, 181])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 158, 181])
        upper_red2 = np.array([179, 255, 255])

        # Green color range
        lower_green = np.array([40, 158, 181])
        upper_green = np.array([90, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        if cv2.countNonZero(red_mask) > 0:
            rospy.loginfo("Red light detected")
            self.traffic_pub.publish(Bool(data=False))
        elif cv2.countNonZero(green_mask) > 0:
            rospy.loginfo("Green light detected")
            self.traffic_pub.publish(Bool(data=True))
        else:
            rospy.loginfo("No clear traffic light detected")
            self.traffic_pub.publish(Bool(data=False))

if __name__ == '__main__':
    rospy.init_node('traffic_light_detector', anonymous=True)
    tld = TrafficLightDetector()
    try:
        tld.process_image()  # Start the image processing loop
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()
