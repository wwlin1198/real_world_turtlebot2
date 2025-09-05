#!/usr/bin/env python  # Use Python 2.7

from __future__ import print_function, division  # Python 2 compatibility


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

# Instead of using cv_bridge, use numpy directly for 16-bit depth images
def imgmsg_to_cv2(img_msg):
    if img_msg.encoding == '16UC1':
        dtype = np.dtype("uint16")  # Use uint16 for 16-bit depth
    else:
        dtype = np.dtype("uint8")
        
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_np = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, -1)
    
    # If the image has only one channel (like depth)
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(axis=2)
    
    return image_np

class RealsenseDepthViewer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('realsense_depth_viewer')
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create directory for saving images
        self.save_dir = "/home/turtlebot02/turtlebot_ws/src/scripts/depth_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo("Created directory for depth images: %s", self.save_dir)
        
        # Flag to track if we've saved an image
        self.image_saved = False
        
        # Subscribe to the depth image topic from RealSense
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        
        # PyBullet depth camera parameters - MATCH THESE TO YOUR PYBULLET SETTINGS
        self.pybullet_near = 0.1  # Near plane in meters - match to your nearVal
        self.pybullet_far = 1.5  # Far plane in meters - match to your farVal
        
        rospy.loginfo("RealSense depth viewer initialized. Waiting for depth images...")
        
    def depth_callback(self, msg):
        try:
            # Skip if we've already saved an image
            if self.image_saved:
                return
                
            rospy.loginfo("Received depth image. Processing...")
            rospy.loginfo("Image encoding: %s", msg.encoding)
            
            # Convert ROS Image to OpenCV image
            try:
                # First try with cv_bridge (Python 2 compatible)
                if msg.encoding == '16UC1':
                    depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                else:
                    depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                rospy.loginfo("Successfully used cv_bridge")
            except Exception as bridge_error:
                # Fall back to direct numpy conversion
                rospy.logwarn("cv_bridge failed: %s. Using direct numpy conversion.", str(bridge_error))
                depth_image = imgmsg_to_cv2(msg)
            
            # Get image info
            rospy.loginfo("Depth image shape: %s, dtype: %s, min: %.2f, max: %.2f", 
                         str(depth_image.shape), 
                         str(depth_image.dtype),
                         np.min(depth_image),
                         np.max(depth_image))
            
            # Handle 2-channel depth image by extracting just the first channel
            if len(depth_image.shape) > 2 and depth_image.shape[2] == 2:
                rospy.loginfo("Detected 2-channel depth image, extracting first channel")
                depth_image_single = depth_image[:, :, 0].copy()
            else:
                depth_image_single = depth_image
            
            # Save the raw 16-bit depth image for reference
            timestamp = rospy.Time.now().to_sec()
            raw16_filename = os.path.join(self.save_dir, f"depth_raw16bit_{timestamp:.0f}.png")
            cv2.imwrite(raw16_filename, depth_image_single)
            rospy.loginfo("Saved raw 16-bit depth image to %s", raw16_filename)
            
            # Resize the depth image to match your PyBullet camera resolution
            depth_image_resized = cv2.resize(depth_image_single, (320, 240), interpolation=cv2.INTER_NEAREST)
            rospy.loginfo("Resized depth image to 320x240")
            
            # Convert RealSense depth values (typically in millimeters) to meters
            depth_meters = depth_image_resized.astype(np.float32) / 1000.0
            
            # Clip depth values to valid range
            depth_meters = np.clip(depth_meters, self.pybullet_near, self.pybullet_far)
            
            # Convert to PyBullet's depth buffer format
            # PyBullet's depth buffer is in range [0,1] where 1 is far and 0 is near
            # The formula comes from the OpenGL depth buffer calculation
            depth_pybullet = (self.pybullet_far + self.pybullet_near - (2.0 * self.pybullet_near * self.pybullet_far) / depth_meters) / (self.pybullet_far - self.pybullet_near)
            
            # Normalize to [0,1] range, matching PyBullet's depth_array format
            depth_pybullet = np.clip(depth_pybullet, 0.0, 1.0)
            
            # Convert to 8-bit for visualization and saving as image
            depth_pybullet_8bit = (depth_pybullet * 255).astype(np.uint8)
            
            # Save the PyBullet-style depth image as PNG
            raw_filename = os.path.join(self.save_dir, f"depth_pybullet_{timestamp:.0f}.png")
            success = cv2.imwrite(raw_filename, depth_pybullet_8bit)
            if success:
                rospy.loginfo("Successfully saved PyBullet-style depth image to %s", raw_filename)
            else:
                rospy.logerr("Failed to save depth image to %s", raw_filename)
            
            
            # Create and save a colormap for visualization
            try:
                # Create a colormap for better visualization
                depth_colormap = cv2.applyColorMap(depth_pybullet_8bit, cv2.COLORMAP_JET)
                
                # Save the colorized depth image
                color_filename = os.path.join(self.save_dir, f"depth_color_{timestamp:.0f}.png")
                success = cv2.imwrite(color_filename, depth_colormap)
                if success:
                    rospy.loginfo("Successfully saved colorized depth image to %s", color_filename)
                else:
                    rospy.logerr("Failed to save colorized depth image to %s", color_filename)
                    
                # Also create a colormap of the original depth (not PyBullet format)
                # This helps for comparison
                depth_orig_normalized = cv2.normalize(depth_meters, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_orig_colormap = cv2.applyColorMap(depth_orig_normalized, cv2.COLORMAP_JET)
                orig_color_filename = os.path.join(self.save_dir, f"depth_orig_color_{timestamp:.0f}.png")
                cv2.imwrite(orig_color_filename, depth_orig_colormap)
                rospy.loginfo("Saved original colorized depth image to %s", orig_color_filename)
                
            except Exception as e:
                rospy.logerr("Error creating colormap: %s", str(e))
            
            # Mark that we've saved an image
            self.image_saved = True
            rospy.loginfo("Image processing complete. Shutting down...")
            
            # Shutdown the node after saving one image
            rospy.signal_shutdown("Image saved successfully")
            
        except Exception as e:
            rospy.logerr("Error processing depth image: %s", str(e))
            import traceback
            rospy.logerr("Traceback: %s", traceback.format_exc())

    def run(self):
        # Keep the node running until we've saved an image
        rospy.spin()

if __name__ == '__main__':
    try:
        viewer = RealsenseDepthViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass