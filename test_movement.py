#!/usr/bin/env python

from __future__ import print_function, division  # Python 2 compatibility

import rospy
from geometry_msgs.msg import Twist

class SimpleMovement:
    def __init__(self, forward_speed=0.3, rotation_speed=-1):
        # Initialize ROS node
        rospy.init_node('simple_movement', anonymous=False)
        
        # Set up parameters
        self.forward_speed = forward_speed
        self.rotation_speed = rotation_speed
        
        # Create publisher - using the navi topic which seems to work better
        self.cmd_vel = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        
        # Tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")
        
        # Register shutdown handler
        rospy.on_shutdown(self.shutdown)
    
    def move(self, linear, angular):
        """Send movement command"""
        rospy.loginfo("Moving - lin: {} ang: {}".format(linear, angular))
        
        # Create and publish Twist message
        move_cmd = Twist()
        move_cmd.linear.x = linear
        move_cmd.angular.z = angular
        self.cmd_vel.publish(move_cmd)
    
    def run_sequence(self):
        """Run the movement sequence using ROS time for better accuracy"""
        try:
            rospy.loginfo("Starting movement sequence")
            
            # Move forward for 1 second
            rospy.loginfo("Step 1: Moving forward for 3 seconds")
            t0 = rospy.get_rostime().secs
            while (t0 + 4.5 >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self.move(self.forward_speed, 0.0)
                rospy.sleep(0.1)  # 10Hz update rate
            
            # Stop for 1 second
            rospy.loginfo("Step 2: Stopping for 1 second")
            self.move(0.0, 0.0)  # Send stop command
            rospy.sleep(0.5)
            
            # NEW STEP: Rotate 90 degrees (counter-clockwise)
            rospy.loginfo("Step 3: Rotating 90 degrees")
            t0 = rospy.get_rostime().secs
            while (t0 + 2 >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self.move(0.0, self.rotation_speed)  # Positive = counter-clockwise
                rospy.sleep(0.1)  # 10Hz update rate
            
            # Stop briefly after rotation
            self.move(0.0, 0.0)  # Send stop command
            rospy.sleep(1.0)
            
            # Move forward for 3 seconds
            rospy.loginfo("Step 4: Moving forward for 5 seconds")
            t0 = rospy.get_rostime().secs
            while (t0 + 5 >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self.move(self.forward_speed, 0.0)
                rospy.sleep(0.1)  # 10Hz update rate
            
            # Final stop
            rospy.loginfo("Sequence completed")
            self.move(0.0, 0.0)  # Send stop command
            
        except Exception as e:
            rospy.logerr("Error during movement: {}".format(str(e)))
            self.move(0.0, 0.0)  # Make sure we stop on error

    def shutdown(self):
        """Stop TurtleBot when shutting down"""
        rospy.loginfo("Stop TurtleBot")
        self.cmd_vel.publish(Twist())  # Empty twist = all zeros = stop
        rospy.sleep(1)  # Give it time to stop

if __name__ == '__main__':
    try:
        mover = SimpleMovement()
        mover.run_sequence()
    except rospy.ROSInterruptException:
        pass

