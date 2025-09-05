#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion
# Manual quaternion conversion to avoid tf import issues
import json
import math

class WaypointNavigator:
    def __init__(self):
        rospy.init_node('waypoint_navigator', anonymous=True)
        
        # Create action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        print("Waiting for move_base action server...")
        self.client.wait_for_server()
        print("Connected to move_base server!")
        
    def create_goal(self, x, y, yaw=0.0):
        """Create a MoveBaseGoal from x, y coordinates and yaw angle"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Set position
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion manually
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return goal
    
    def go_to_waypoint(self, x, y, yaw=0.0, timeout=60.0):
        """Send robot to specific waypoint"""
        goal = self.create_goal(x, y, yaw)
        
        print(f"Sending goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
        self.client.send_goal(goal)
        
        # Wait for result
        result = self.client.wait_for_result(rospy.Duration(timeout))
        
        if result:
            state = self.client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                print("✓ Waypoint reached successfully!")
                return True
            else:
                print(f"✗ Failed to reach waypoint. State: {state}")
                return False
        else:
            print("✗ Timeout waiting for waypoint")
            self.client.cancel_goal()
            return False
    
    def go_to_waypoints_from_file(self, filename):
        """Navigate through waypoints loaded from JSON file"""
        try:
            with open(filename, 'r') as f:
                waypoints = json.load(f)
            
            print(f"Loaded {len(waypoints)} waypoints from {filename}")
            
            for i, wp in enumerate(waypoints):
                print(f"\n--- Going to waypoint {i+1}/{len(waypoints)} ---")
                x = wp['position']['x']
                y = wp['position']['y']
                
                # Convert quaternion to yaw if available
                if 'orientation' in wp:
                    quat = wp['orientation']
                    yaw = self.quaternion_to_yaw(quat['x'], quat['y'], quat['z'], quat['w'])
                else:
                    yaw = 0.0
                
                success = self.go_to_waypoint(x, y, yaw)
                if not success:
                    print("Failed to reach waypoint. Stopping navigation.")
                    break
                    
                # Small delay between waypoints
                rospy.sleep(1.0)
                
        except Exception as e:
            print(f"Error loading waypoints: {e}")
    
    def quaternion_to_yaw(self, x, y, z, w):
        """Convert quaternion to yaw angle"""
        return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    
    def interactive_mode(self):
        """Interactive mode to send individual waypoints"""
        print("\n=== Interactive Waypoint Navigation ===")
        print("Enter waypoints as: x y [yaw]")
        print("Example: 1.5 2.0 1.57")
        print("Type 'quit' to exit")
        
        while not rospy.is_shutdown():
            try:
                user_input = raw_input("\nEnter waypoint (x y [yaw]): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                parts = user_input.split()
                if len(parts) < 2:
                    print("Please enter at least x and y coordinates")
                    continue
                
                x = float(parts[0])
                y = float(parts[1])
                yaw = float(parts[2]) if len(parts) > 2 else 0.0
                
                self.go_to_waypoint(x, y, yaw)
                
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    try:
        navigator = WaypointNavigator()
        
        print("\n=== Waypoint Navigator ===")
        print("1. Interactive mode - Enter waypoints manually")
        print("2. Load waypoints from file")
        print("3. Go to single waypoint")
        
        choice = raw_input("Choose mode (1/2/3): ").strip()
        
        if choice == '1':
            navigator.interactive_mode()
            
        elif choice == '2':
            filename = raw_input("Enter waypoint file path: ").strip()
            navigator.go_to_waypoints_from_file(filename)
            
        elif choice == '3':
            x = float(raw_input("Enter x coordinate: "))
            y = float(raw_input("Enter y coordinate: "))
            yaw = input("Enter yaw angle (default 0): ") or 0.0
            yaw = float(yaw)
            navigator.go_to_waypoint(x, y, yaw)
            
        else:
            print("Invalid choice")
            
    except rospy.ROSInterruptException:
        print("Navigation interrupted")

if __name__ == '__main__':
    main()
