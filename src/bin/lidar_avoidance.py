#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ObjectAvoidanceNode():
    def __init__(self):
        rospy.init_node('lidar_avoidance', anonymous=True)
        self.sub_lidar = Subscriber('/scan', LaserScan)
        self.ts = ApproximateTimeSynchronizer([self.sub_lidar], 10, 0.1)
        self.ts.registerCallback(self.lidar_callback)

        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.safe_distance = 0.8  # Meters
        rospy.loginfo('Object Avoidance Node Started')

    def lidar_callback(self, msg):
        ranges = msg.ranges
        front = min(min(ranges[0:30]), min(ranges[-30:]))  # Front section
        left = min(ranges[60:120])  # Left section
        right = min(ranges[-120:-60])  # Right section

        twist_msg = Twist()

        if front < self.safe_distance:
            # Obstacle in front, determine whether to turn left or right
            if left > right:
                rospy.loginfo('Turning left')
                twist_msg.angular.z = 0.6  # Turn left
            else:
                rospy.loginfo('Turning right')
                twist_msg.angular.z = -0.6  # Turn right
            twist_msg.linear.x = 0.0  # Stop forward motion
        else:
            # No obstacle in front, move forward
            rospy.loginfo('Moving forward')
            twist_msg.linear.x = 0.2
            twist_msg.angular.z = 0.0

        self.publisher.publish(twist_msg)

if __name__ == '__main__':
    try:
        ObjectAvoidanceNode()
        rospy.spin()
    except Exception as e:
        print(f"Error with find_empty_seat.py: {e}")
