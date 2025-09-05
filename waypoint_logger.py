#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import json
from datetime import datetime

class WaypointLogger:
    def __init__(self):
        rospy.init_node('waypoint_logger', anonymous=True)
        self.waypoints = []
        
        # Subscribe to AMCL pose
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        
        print("Waypoint Logger started. Press Ctrl+C to save waypoints.")
        
    def pose_callback(self, msg):
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Extract orientation (quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Store current pose (every 2 seconds to avoid spam)
        current_time = rospy.Time.now().to_sec()
        if not hasattr(self, 'last_log_time') or (current_time - self.last_log_time) > 2.0:
            waypoint = {
                'timestamp': datetime.now().isoformat(),
                'position': {'x': x, 'y': y, 'z': z},
                'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw}
            }
            self.waypoints.append(waypoint)
            print(f"Logged waypoint: x={x:.3f}, y={y:.3f}, yaw={self.quaternion_to_yaw(qx,qy,qz,qw):.3f}")
            self.last_log_time = current_time
    
    def quaternion_to_yaw(self, x, y, z, w):
        import math
        return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        
    def save_waypoints(self):
        filename = f"waypoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.waypoints, f, indent=2)
        print(f"Saved {len(self.waypoints)} waypoints to {filename}")

if __name__ == '__main__':
    try:
        logger = WaypointLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        logger.save_waypoints()
