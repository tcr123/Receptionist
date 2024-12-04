#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ultralytics import YOLO
import cv2
import numpy as np

included_classes = [0, 39, 45, 57, 73]

class ObstacleAvoidance:
    def __init__(self):
        rospy.init_node('obstacle_avoidance', anonymous=True)
        self.sub_color_bot = Subscriber('/camera/color/image_raw', Image)
        self.sub_depth_bot = Subscriber('/camera/depth/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer([self.sub_color_bot, self.sub_depth_bot], 10, 0.1)
        self.ts.registerCallback(self.images_callback)

        self.cv_image_bot = None
        self.img_depth_bot = None
        self.detect_obstacles = []

        self.model = YOLO("/home/mustar/catkin_ws/src/cr_receptionist/models/yolov8n.pt")

    def images_callback(self, msg_color, msg_depth):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            self.img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')

            self.avoid_obstacles()

        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return
        
    def avoid_obstacles(self):
        results = self.model.track(source=self.cv_image, conf=0.3, classes=included_classes, iou=0.5, persist=True, tracker='bytetrack.yaml')
        self.detect_obstacles.clear()

        for objects in results:
            for result in objects:
                a = result.boxes.cpu().numpy()
                print(a)
                x1, y1, x2, y2 = map(int, a.xyxy[0])
                current_depth_img = self.img_depth[int(y1):int(y2),int(x1):int(x2)]
                current_depth = np.median(current_depth_img)

                obj_info = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'depth': current_depth
                }
                
                # Append the dictionary to the list
                self.detect_obstacles.append(obj_info)

                cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(self.cv_image, f'Depth: {current_depth}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Tracked Image", self.cv_image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 27 is the ESC key
            self.follow = False
            cv2.destroyAllWindows()

        print(self.detect_obstacles)

if __name__ == "__main__":
    try:
        ObstacleAvoidance()
        rospy.spin()
    except Exception as e:
        print(f"Error with find_empty_seat.py: {e}")
