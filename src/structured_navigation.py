#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

original = 0
start = 0

class NavToPoint:
    def __init__(self):
        rospy.on_shutdown(self.cleanup)

        # Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")

        # Wait for the action server to become available
        self.move_base.wait_for_server(rospy.Duration(120))
        rospy.loginfo("Connected to move base server.")

        # A variable to hold the initial pose of the robot to be set by the user in RViz
        initial_pose = PoseWithCovarianceStamped()
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped, self.update_initial_pose)

        # Get the initial pose from the user
        rospy.loginfo("*** Click the 2D Pose Estimate button in RViz to set the robot's initial pose...")
        rospy.wait_for_message('initialpose', PoseWithCovarianceStamped)
        
        # Make sure we have the initial pose
        while initial_pose.header.stamp == "":
           rospy.sleep(1)
        rospy.loginfo("Starting navigation node...")
        rospy.sleep(1)

        # List of locations
        self.locations = dict()
        self.locations['FindEmptySeat'] = [4.7540, 0.1029, 0.6527, 0.7576]
        self.locations['Entrance Recpt'] = [2.1818, -0.1560, -0.5962, 0.8028]

        # Subscribe to get target location
        self.target_location = rospy.Subscriber("nav_cmd", String, self.nav_callback)
        self.nav_feedback_pub = rospy.Publisher("nav_feedback", String, queue_size=1)

        # --------------------------------------------------------------------------

        # Get initial location
        # quaternion = quaternion_from_euler(0.0, 0.0, 0.0)
        # self.origin = Pose(Point(0, 0, 0), Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
        # --------------------------------------------------------------------------

    def nav_callback(self, target_location):
        targetted_location = str(target_location.data)
        print(f"targetted location: {targetted_location}")
        self.goal = MoveBaseGoal()
        rospy.loginfo("Ready to go.")
        global start

        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.header.stamp = rospy.Time.now()

        # Robot will go to the target location
        if targetted_location in self.locations.keys():
            coordinate = self.locations[targetted_location]
            destination = Pose(Point(coordinate[0], coordinate[1], 0.000), Quaternion(0.0, 0.0, coordinate[2], coordinate[3]))
            rospy.loginfo(f"Going to {targetted_location}")
            rospy.sleep(1)
            self.goal.target_pose.pose = destination
            self.move_base.send_goal(self.goal)
            waiting = self.move_base.wait_for_result(rospy.Duration(300))
            if waiting == 1:
                rospy.loginfo(f"Reached {targetted_location}")
                self.nav_feedback_pub.publish("Done")
                rospy.sleep(1)
                start = 0
        else: 
            rospy.loginfo("Invalid Location")

        rospy.Rate(2).sleep()


    def update_initial_pose(self, initial_pose):
        self.initial_pose = initial_pose
        global original
        if original == 0:
            self.origin = self.initial_pose.pose.pose
            original = 1

    def cleanup(self):
        rospy.loginfo("Shutting down navigation...")
        self.move_base.cancel_goal()

if __name__=="__main__":
    rospy.init_node('navi_point')
    try:
        NavToPoint()
        rospy.spin()
    except:
        pass
