#!/usr/bin/env python3
import rospy
from ultralytics import YOLO
import cv2
import numpy as np
import json
import time
import os
from gtts import gTTS
import speech_recognition as sr

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from utils.speech_processor import SpeechProcessor

AUDIO_FILE_PATH = "/home/mustar/catkin_ws/src/cr_receptionist/models/main_audio.mp3"
INCLUDE_CLASSES = [0, 39, 45, 73]

# Helper Functions
def text_to_audio(text):
    """Convert text to speech and play the audio."""
    tts = gTTS(text)
    tts.save(AUDIO_FILE_PATH)
    os.system(f"mpg321 {AUDIO_FILE_PATH}")
    os.remove(AUDIO_FILE_PATH)

class ChairTracking:
    def __init__(self):
        rospy.init_node('chair_tracking', anonymous=True)
        rospy.Subscriber("start_tracking", String, self.callback)
        self.sub_color = Subscriber('/camera/color/image_raw', Image)
        self.sub_depth = Subscriber('/camera/depth/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer([self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)

        # TODO - need add one publisher to tell the main py done
        self.pub = rospy.Publisher("guest_seated", String, queue_size=10)
        self.pub_seat = rospy.Publisher("seat_available", String, queue_size=10)

        self.speech_processor = SpeechProcessor()

        # default speed and turn
        self.speed = 0.6
        self.turn = 1

        # image frame
        self.cv_image = None
        self.img_depth = None

        # tracking obj params
        self.model = YOLO("/home/mustar/catkin_ws/src/cr_receptionist/models/yolov8n.pt")
        # self.model_collision = YOLO("/home/mustar/catkin_ws/src/cr_receptionist/models/yolov8n.pt")
        self.detect_result = {}
        self.detect_obstacles = []
        
        # for seat checking
        self.checking = False
        self.max_checks = 5
        self.check_count = 0

        # for tracking object
        self.track = False
        self.follow = False
        self.follow_obj_id = -1
        self.follow_time = 0
        self.front = 0
        self.left = 0
        self.right = 0

        # moving
        self.move_cmd = Twist()
        self.control_speed = 0
        self.control_turn = 0
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)

    def callback(self, msg):
        data = msg.data
        if data == 'tracking':
            self.track = True
            self.follow = True
            self.follow_obj_id = -1
            self.follow_time = 0

            # temporary use
            self.change_speed({
                "x":1,
                "y":1
            })

            self.move()
        
        if data == 'checking_seat':
            self.checking = True
            self.check_count = 0

    def images_callback(self, msg_color, msg_depth):
        # rospy.loginfo("Images callback triggered")

        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            self.img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')

            self.collision_image = self.cv_image.copy()

            if self.checking == True:
                self.checkSeat()

            if self.track == True:
                self.trackObj()

            if self.follow == True:
                self.followObj()

        except CvBridgeError as e:
            rospy.logwarn(str(e))
            return
        
    def checkSeat(self):
        # person is 0, chair is 56
        results = self.model.predict(source=self.cv_image, classes=[56])

        if results:
            num = 0
            for objects in results:
                for result in objects:
                    a = result.boxes.cpu().numpy()
                    x1, y1, x2, y2 = map(int, a.xyxy[0])
                    num += 1

                    # Draw bounding boxes
                    cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Render the image with bounding boxes
            cv2.imshow('Empty seat', self.cv_image)
            cv2.waitKey(1)

            # check repeated 5 times to improve reliabity
            if self.check_count > self.max_checks:
                self.checking = False
                
                if num > 0:
                    rospy.loginfo(f"Chair detected: {num}")
                    self.pub_seat.publish("available")
                
                if num == 0:
                    rospy.loginfo("No chairs detected.")
                    self.pub_seat.publish("inavailable")

                rospy.loginfo("Finished checking for empty seats.")
                cv2.destroyAllWindows()

            if num > 0:
                rospy.loginfo(f"Chair detected: {num}")
                self.check_count += 1

            if num == 0:
                rospy.loginfo("No chairs detected.")
                self.check_count += 1

        else:
            rospy.loginfo("No chairs detected.")
            self.pub_seat.publish("inavailable")
        
    def trackObj(self):
        # person is 0, chair is 56
        results = self.model.track(source=self.cv_image, classes=[56], persist=True, tracker = 'bytetrack.yaml')

        self.detect_result = results[0].tojson()

        current_time = time.time()
        if self.follow_obj_id != -2:
            find_follow_obj = False

            for result in results:
                a = result.boxes.cpu().numpy()
                if a.id is not None and self.follow_obj_id in a.id:
                    index = np.where(a.id == self.follow_obj_id)[0][0]
                    x1, y1, x2, y2 = map(int, a.xyxy[index])
                    self.follow_time = current_time
                    find_follow_obj = True

                    cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(self.cv_image, f'ID: {a.id[index]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.imshow("Tracked Image", self.cv_image)
                
            if find_follow_obj == False:
                for result in results:
                    a = result.boxes.cpu().numpy()
                    if a.is_track == True:
                        x1, y1, x2, y2 = map(int, a.xyxy[0])
                        self.setFollowObj(a.id[0])
                        self.follow_time = current_time

                        cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(self.cv_image, f'ID: {a.id[0]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.imshow("Tracked Image", self.cv_image)
                        break
                
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 27 is the ESC key
                cv2.destroyAllWindows()

    # def avoid_obstacles(self):
    #     obstacles = self.model_collision.track(source=self.collision_image, classes=INCLUDE_CLASSES, persist=True, tracker='bytetrack.yaml')
    #     self.detect_obstacles.clear()

    #     for objects in obstacles:
    #         for result in objects:
    #             a = result.boxes.cpu().numpy()
    #             x1, y1, x2, y2 = map(int, a.xyxy[0])
    #             current_depth_img = self.img_depth[int(y1):int(y2),int(x1):int(x2)]
    #             current_depth = np.median(current_depth_img)

    #             obj_info = {
    #                 'x1': x1,
    #                 'y1': y1,
    #                 'x2': x2,
    #                 'y2': y2,
    #                 'depth': current_depth
    #             }
                
    #             # Append the dictionary to the list
    #             self.detect_obstacles.append(obj_info)
    #             cv2.rectangle(self.collision_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #     cv2.imshow("Obstacle Image", self.collision_image)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key in (ord('q'), 27):  # 27 is the ESC key
    #         cv2.destroyAllWindows()

    # set follow object id  
    def setFollowObj(self, id):
        self.follow = True
        self.follow_obj_id = id
        rospy.loginfo(f"Start follow the object: {id}")

    # follow obj
    def followObj(self):
        x1, y1, x2, y2 = -1, -1, -1, -1

        results = json.loads(self.detect_result)

        for result in results:
            if ("track_id" in result):
                if self.follow_obj_id == result["track_id"]:
                    x1 = result["box"]["x1"]
                    y1 = result["box"]["y1"]
                    x2 = result["box"]["x2"]
                    y2 = result["box"]["y2"]

        current_depth_img = self.img_depth[int(y1):int(y2),int(x1):int(x2)]
        current_depth = np.median(current_depth_img)

        depth_threshold = 1800

        spd_x = 0
        spd_y = 0

        if current_depth > (depth_threshold + 3000):
            spd_x = 0.6
        elif current_depth > depth_threshold:
            spd_x = 0.3
        else:
            spd_x = 0

        if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
            print("center")
            if self.control_speed != 0 or self.control_turn != 0:
                spd_x = 0
                spd_y = 0
        else:
            obj_center = (x1+x2)/2

            dif_center = obj_center - 320

            # right side
            if dif_center > 0:
                if dif_center < 20:
                    print("center")
                elif dif_center > 50:
                    spd_y=-0.4
                elif dif_center > 160:
                    spd_y=-0.6
                elif dif_center > 250:
                    spd_y=-1
                else:
                    spd_y=-((abs(dif_center)-20)/320)
            # left side
            else:
                if abs(dif_center) < 20:
                    print("center")
                elif abs(dif_center) > 50:
                    spd_y = 0.4
                elif abs(dif_center) > 160:
                    spd_y= 0.6 
                elif abs(dif_center) > 250:
                    spd_y=1 
                else:
                    spd_y=(abs(dif_center)-20)/320

        print("Current depth: ", current_depth)
        
        if current_depth < depth_threshold and current_depth > 500:
            rospy.loginfo("Reached the destination")
            self.pub.publish("done")
            self.track = False
            self.follow_obj_id = -2
            self.follow = False
            cv2.destroyAllWindows()
        
        self.change_speed({
            "x":spd_x,
            "y":spd_y
        })

        self.move()

    def change_speed(self, spd_info):
        print(spd_info)
        x = spd_info["x"]
        y = spd_info["y"]

        if abs(y) < 0.2:
            y = 0
        
        if abs(x) < 0.2:
            x = 0 

        target_speed = self.speed * x
        target_turn = self.turn * y

        if x == 0 and y == 0:
            self.control_speed = 0
            self.control_turn = 0

        if target_speed > self.control_speed:
            self.control_speed = min( target_speed, self.control_speed + 0.02 )
        elif target_speed <  self.control_speed:
            self.control_speed = max( target_speed,  self.control_speed - 0.02 )
        else:
            self.control_speed = target_speed

        if target_turn > self.control_turn:
            self.control_turn = min( target_turn, self.control_turn + 0.1 )
        elif target_turn < self.control_turn:
            self.control_turn = max( target_turn, self.control_turn - 0.1 )
        else:
            self.control_turn = target_turn

    # moving function
    def move(self):
        rospy.loginfo('Move....')
        if self.follow:
            twist = Twist()
            twist.linear.x = self.control_speed; twist.linear.y = 0; twist.linear.z = 0
            twist.angular.x = 0; twist.angular.y = 0; twist.angular.z =self.control_turn
            self.cmd_vel_pub.publish(twist)

if __name__ == "__main__":
    try:
        ChairTracking()
        rospy.spin()
    except Exception as e:
        print(f"Error with find_empty_seat.py: {e}")