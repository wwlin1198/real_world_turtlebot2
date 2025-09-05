#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import sys
import math

def send_goal(x, y, yaw=0.0):
    """Simple function to send robot to a waypoint"""
    
    # Initialize ROS node
    rospy.init_node('simple_goto', anonymous=True)
    
    # Create action client
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    print("Waiting for move_base server...")
    client.wait_for_server()
    print("Connected!")
    
    # Create goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    
    # Set position
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0.0
    
    # Set orientation (convert yaw to quaternion manually)
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
    goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
    
    # Send goal
    print(f"Going to: x={x}, y={y}, yaw={yaw}")
    client.send_goal(goal)
    
    # Wait for result
    result = client.wait_for_result(rospy.Duration(60.0))
    
    if result:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            print("✓ Successfully reached waypoint!")
        else:
            print(f"✗ Failed to reach waypoint. State: {state}")
    else:
        print("✗ Timeout - canceling goal")
        client.cancel_goal()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python simple_goto.py <x> <y> [yaw]")
        print("Example: python simple_goto.py 1.5 2.0 1.57")
        sys.exit(1)
    
    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        
        send_goal(x, y, yaw)
        
    except ValueError:
        print("Error: Please provide numeric values for coordinates")
    except KeyboardInterrupt:
        print("Interrupted by user")
