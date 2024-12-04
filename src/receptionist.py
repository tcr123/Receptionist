#! /usr/bin/env python3

import rospy
import threading
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from utils.speech_processor import SpeechProcessor
from utils.face_recognition import FaceRecognition
from cv_bridge import CvBridge, CvBridgeError
import threading
from math import radians
import time
import json
import os
import cv2

class Receptionist:
    def __init__(self):
        self.RECORD = False
        self.count = 0
        self.speech_processor = SpeechProcessor()
        self.face_recognition = FaceRecognition()

        self.last_no_face_time = 0  # Timestamp of the last "face not detected" message
        self.no_face_interval = 10  # Minimum interval in seconds between "face not detected" messages
        self.max_checks = 10  # Maximum number of checks
        self.check_count = 0  # Counter for iterations
        self.processing_image = False
        self.lock = threading.Lock()

        rospy.sleep(2)

        rospy.init_node('receptionist_main', anonymous=True)
        self.human_result_pub = rospy.Publisher('human_detected', String, queue_size = 10)
        self.task_status_pub = rospy.Publisher('task_status', String, queue_size=10)
        self.nav_pub = rospy.Publisher("nav_cmd", String, queue_size=10)
        self.empty_seat_pub = rospy.Publisher("start_tracking", String, queue_size=10)
        
        rospy.Subscriber('camera/color/image_raw', Image, self.image_callback, queue_size=1)

        rospy.sleep(2)
        self.move_to_start_location()

    def move_to_start_location(self):
        rospy.loginfo("Move to the start area")
        self.nav_pub.publish("FindEmptySeat")
        rospy.wait_for_message("nav_feedback", String)

        rospy.sleep(1)

        self.empty_seat_pub.publish("checking_seat")

        seat_msg = rospy.wait_for_message("seat_available", String)

        # Access the data within the returned message
        feedback_data = seat_msg.data
        print(f'Check seat: {feedback_data}')

        if feedback_data == "inavailable":
            rospy.loginfo("No seat available now, stop serving")
            self.speech_processor.text2audio("No seat available now, I will stop serving")
            return
        else:
            rospy.loginfo("start serving now")
            self.speech_processor.text2audio("Empty seat available, will serving now")
            self.move_to_first_location()
            
    def move_to_first_location(self):
        rospy.loginfo("Move to the first location")
        self.nav_pub.publish("Entrance Recpt")
        rospy.wait_for_message("nav_feedback", String)
        self.RECORD = True
        self.count+=1
         
    # ------------------- Detect Human ----------------------
    def image_callback(self, msg):
        if not self.RECORD or self.processing_image:
            return
        
        rospy.loginfo("Start detecting human...")
        rate = rospy.Rate(5)

        try:
            # Ensure thread-safe access to the image
            with self.lock:
                local_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

            if local_image is None:
                rospy.logwarn("No valid image available for checking seats. Retrying...")
                rate.sleep()
                return

            cv_image = local_image.copy()

            rospy.loginfo("Start detecting human...")

            humans = self.face_recognition.detect_human(cv_image)
            faces = self.face_recognition.detect_faces(cv_image)

            cv2.imshow('Live Feed', cv_image)
            cv2.waitKey(1)

            if self.check_count > self.max_checks:
                if humans > 0 and len(faces) > 0:
                    rospy.loginfo("human and faces detected")
                    self.RECORD = False
                    # self.recognise_person(local_image)
                    threading.Thread(target=self.recognise_person, args=(local_image,)).start()
                    cv2.destroyAllWindows()

            if humans > 0 and len(faces) > 0:
                rospy.loginfo("human and faces detected")
                self.check_count += 1
                print(self.check_count)
                return
            elif humans > 0:
                # Check if enough time has passed since the last "face not detected" message
                current_time = time.time()
                if current_time - self.last_no_face_time >= self.no_face_interval:
                    rospy.loginfo("faces not detected")
                    # Trigger the speech processor without blocking image processing
                    threading.Thread(target=self.speech_processor.text2audio, args=("The face is not detected, please face toward me",)).start()
                    # Update the timestamp of the last message
                    self.last_no_face_time = current_time
                return
        except CvBridgeError as e:
            rospy.logwarn("CvBridge error: {}".format(e))
        finally:
            rate.sleep()

        # # Lock and processing flag
        # rate = rospy.Rate(5)
        # self.processing_image = True
        # try:
        #     with self.lock:
        #         curr_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

        #     cv_image = curr_image.copy()

        #     rospy.loginfo("Start detecting human...")

        #     humans = self.face_recognition.detect_human(cv_image)
        #     faces = self.face_recognition.detect_faces(cv_image)

        #     # Display the image
        #     cv2.imshow('Live Feed', cv_image)
        #     cv2.waitKey(1)

        #     if humans > 0 and len(faces) > 0:
        #         current_time = time.time()
        #         if current_time - self.last_no_face_time >= self.no_face_interval:    
        #             rospy.loginfo("human and faces detected")
        #             self.RECORD = False
        #             self.recognise_person(curr_image)
        #     elif humans > 0:
        #         # Check if enough time has passed since the last "face not detected" message
        #         current_time = time.time()
        #         if current_time - self.last_no_face_time >= self.no_face_interval:
        #             rospy.loginfo("faces not detected")
        #             # Trigger the speech processor without blocking image processing
        #             threading.Thread(target=self.speech_processor.text2audio, args=("The face is not detected, please face toward me",)).start()
        #             # Update the timestamp of the last message
        #             self.last_no_face_time = current_time
        
        # except CvBridgeError as e:
        #     rospy.logwarn("CvBridge error: {}".format(e))
        
        # finally:
        #     # Reset the flag after processing is complete
        #     self.processing_image = False
        #     rate.sleep()

    def cleanup(self):
        cv2.destroyAllWindows()

    # -------------- get Name --------------------- 
    def recognise_person(self, image):
        person_name = self.face_recognition.recognize_faces(image)
        print('Checking name')
        print(person_name)

        if person_name:
            hello_str = 'Hello {}, nice to meet you'.format(person_name)
            self.speech_processor.text2audio(hello_str)
        else:
            self.speech_processor.text2audio("Hello, what is your name?")
            name = self.speech_processor.audio2text()
            self.speech_processor.text2audio("Please face to me, I will take a picture")
            rospy.sleep(1)

            self.face_recognition.save_faces(image, name)

        # reset the check out
        self.check_count = 0
        rospy.loginfo("Finished checking for face recognition")
        
        # Move to corner to search for empty seat
        self.speech_processor.text2audio('Thanks for your waiting. Please follow me and stand on my left side')
        # TODO mapping
        self.nav_pub.publish("FindEmptySeat")
        rospy.wait_for_message("nav_feedback", String)

        # Scan for empty seat
        self.empty_seat_pub.publish("tracking")
        self.empty_seat_result = rospy.wait_for_message("guest_seated", String)

        if self.empty_seat_result.data == "done":
            # Tell the guest to sit
            direct_sit_str = 'Please sit the chair in front of me. thank you'.format(person_name)
            self.speech_processor.text2audio(direct_sit_str)
        
        self.move_to_start_location()
        self.RECORD = True
            
if __name__ == "__main__":
    try:
        f = Receptionist()
        rospy.spin()
        # f.set_up_waiting()

    except rospy.ROSInterruptException:
        rospy.loginfo("Main Error")
        f.cleanup()
        