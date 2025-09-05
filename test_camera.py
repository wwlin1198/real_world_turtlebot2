#!/usr/bin/env python

from __future__ import print_function, division  # Python 2 compatibility

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
import os
import threading
import queue

# Debug information to help diagnose the issue
print("Python version:", sys.version)
print("Python executable:", sys.executable)

# Alternative approach without cv_bridge
class RealsenseDepthViewer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('realsense_depth_viewer')
        
        # Check if display is available
        self.has_display = "DISPLAY" in os.environ
        if not self.has_display:
            rospy.logwarn("No display detected. Images will not be shown.")
        
        # Create a queue for thread-safe image passing
        self.image_queue = queue.Queue(maxsize=1)
        
        # Subscribe to the depth image topic from RealSense
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        
        # Add a publisher to publish the processed image if needed
        self.processed_pub = rospy.Publisher('/processed_depth_image', Image, queue_size=1)
        
        # Start display thread if display is available
        if self.has_display:
            self.display_thread = threading.Thread(target=self.display_loop)
            self.display_thread.daemon = True
            self.display_thread.start()
        
        rospy.loginfo("RealSense depth viewer initialized. Waiting for depth images...")
        
    def depth_callback(self, msg):
        try:
            # Convert ROS Image to numpy array without using cv_bridge
            if msg.encoding == '16UC1':
                depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            else:
                rospy.logwarn("Unexpected encoding: %s", msg.encoding)
                depth_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            
            # Normalize the depth image for display
            # Convert to 8-bit for visualization
            depth_image_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply colormap for better visualization
            depth_colormap = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
            
            # If display is available, put the image in the queue for the display thread
            if self.has_display:
                # Put the latest image in the queue, replacing any old one
                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.image_queue.put(depth_colormap)
            else:
                # Log some information about the image to confirm processing
                rospy.loginfo_throttle(5, "Processed depth image: shape=%s min=%s max=%s", 
                                      depth_array.shape, 
                                      np.min(depth_array) if depth_array.size > 0 else "N/A", 
                                      np.max(depth_array) if depth_array.size > 0 else "N/A")
                
                # Optionally publish the processed image
                # self.publish_processed_image(depth_colormap, msg.header)
            
        except Exception as e:
            rospy.logerr("Error processing depth image: %s", str(e))

    def display_loop(self):
        """Separate thread for displaying images"""
        while not rospy.is_shutdown():
            try:
                # Get the latest image from the queue with a timeout
                image = self.image_queue.get(timeout=0.1)
                cv2.imshow("RealSense D435i Depth", image)
                cv2.waitKey(1)
            except queue.Empty:
                # No image available, just continue
                pass
            except Exception as e:
                rospy.logerr("Error in display thread: %s", str(e))

    def publish_processed_image(self, image, header):
        """Publish the processed image as a ROS message"""
        msg = Image()
        msg.header = header
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = image.shape[1] * 3
        msg.data = image.tobytes()
        self.processed_pub.publish(msg)

    def run(self):
        # Keep the node running
        rospy.spin()
        
        # Close all windows when the node is shut down
        if self.has_display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        viewer = RealsenseDepthViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass