#!/usr/bin/env python

from __future__ import print_function, division  # Python 2 compatibility

import rospy
import numpy as np
import os
import time
import pickle
import signal
import sys
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist

class TeleopRecorder:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('teleop_recorder')
        
        # Parse arguments
        self.max_recording_time = rospy.get_param('~max_recording_time', 300)  # 5 minutes max by default
        self.output_dir = os.path.expanduser(rospy.get_param('~output_dir', '~/turtlebot_ws/src/scripts/teleop_actions'))
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"Created directory for teleop actions: {self.output_dir}")
        
        # Variables for storing actions
        self.action_list = []
        self.start_time = time.time()
        self.is_recording = False
        self.latest_action_time = 0
        self.action_timeout = 1.0  # seconds
        
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Subscribe to the teleop topic
        rospy.Subscriber('/cmd_vel_mux/input/teleop', Twist, self.teleop_callback, queue_size=1)
        
        rospy.loginfo("Teleop recorder initialized, waiting for teleop commands...")
        self.is_recording = True
    
    def teleop_callback(self, msg):
        """Callback function for teleop commands"""
        # Get linear and angular velocities
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z
        
        # Store the action
        self.action_list.append(np.array([linear_vel, angular_vel]))
        
        # Update latest action time
        self.latest_action_time = time.time()
        
        # Log occasionally
        if len(self.action_list) % 10 == 0:
            rospy.loginfo(f"Recorded {len(self.action_list)} teleop actions so far")
    
    def save_actions(self):
        """Save recorded actions to a pickle file"""
        if not self.action_list:
            rospy.logwarn("No actions recorded, nothing to save")
            return
        
        timestamp = int(time.time())
        filename = f"teleop_actions_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.action_list, f)
            rospy.loginfo(f"Saved {len(self.action_list)} teleop actions to {filepath}")
            return filepath
        except Exception as e:
            rospy.logerr(f"Error saving teleop actions: {str(e)}")
            return None
    
    def plot_histograms(self):
        """Plot histograms of the recorded actions"""
        if not self.action_list:
            rospy.logwarn("No actions recorded, nothing to plot")
            return
        
        # Convert list of actions to numpy array
        actions = np.array(self.action_list)
        
        # Create histogram directory if it doesn't exist
        histogram_dir = os.path.expanduser("~/turtlebot_ws/src/scripts/histograms")
        if not os.path.exists(histogram_dir):
            os.makedirs(histogram_dir)
        
        timestamp = int(time.time())
        
        # Plot histogram of linear velocities
        plt.figure(figsize=(10, 6))
        plt.hist(actions[:, 0], bins=30, alpha=0.7)
        plt.title('Histogram of Teleop Linear Velocities')
        plt.xlabel('Linear Velocity (m/s)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_linear = np.mean(actions[:, 0])
        std_linear = np.std(actions[:, 0])
        plt.axvline(mean_linear, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_linear:.3f}')
        plt.text(0.7, 0.85, f'Mean: {mean_linear:.3f}\nStd Dev: {std_linear:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.legend()
        
        linear_histogram_path = os.path.join(histogram_dir, f'teleop_linear_vel_histogram_{timestamp}.png')
        plt.savefig(linear_histogram_path)
        plt.close()
        
        # Plot histogram of angular velocities
        plt.figure(figsize=(10, 6))
        plt.hist(actions[:, 1], bins=30, alpha=0.7)
        plt.title('Histogram of Teleop Angular Velocities')
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_angular = np.mean(actions[:, 1])
        std_angular = np.std(actions[:, 1])
        plt.axvline(mean_angular, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_angular:.3f}')
        plt.text(0.7, 0.85, f'Mean: {mean_angular:.3f}\nStd Dev: {std_angular:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.legend()
        
        angular_histogram_path = os.path.join(histogram_dir, f'teleop_angular_vel_histogram_{timestamp}.png')
        plt.savefig(angular_histogram_path)
        plt.close()
        
        # Plot 2D histogram (heatmap) of linear vs angular velocities
        plt.figure(figsize=(10, 8))
        plt.hist2d(actions[:, 0], actions[:, 1], bins=30, cmap='viridis')
        plt.colorbar(label='Frequency')
        plt.title('2D Histogram of Teleop Linear vs Angular Velocities')
        plt.xlabel('Linear Velocity (m/s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid(True, alpha=0.3)
        
        combined_histogram_path = os.path.join(histogram_dir, f'teleop_combined_vel_histogram_{timestamp}.png')
        plt.savefig(combined_histogram_path)
        plt.close()
        
        rospy.loginfo(f"Saved teleop action histograms to {histogram_dir}")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C to save data before exiting"""
        if self.is_recording:
            rospy.loginfo("Recording stopped, saving data...")
            self.is_recording = False
            filepath = self.save_actions()
            if filepath:
                self.plot_histograms()
                rospy.loginfo("Done saving data, exiting")
            else:
                rospy.logerr("Failed to save data")
        sys.exit(0)
    
    def run(self):
        """Main loop for recording"""
        rospy.loginfo("Starting teleop recording. Use Ctrl+C to stop and save.")
        
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown() and self.is_recording:
            # Check if we've been recording too long
            if time.time() - self.start_time > self.max_recording_time:
                rospy.loginfo(f"Reached maximum recording time of {self.max_recording_time} seconds")
                break
            
            # Check if there's been no input for a while
            if self.action_list and time.time() - self.latest_action_time > self.action_timeout:
                rospy.loginfo("No teleop commands received recently, still recording...")
            
            rate.sleep()
        
        # Save data when loop exits
        if self.is_recording:  # If we didn't already stop recording in the signal handler
            rospy.loginfo("Recording stopped, saving data...")
            self.is_recording = False
            filepath = self.save_actions()
            if filepath:
                self.plot_histograms()
                rospy.loginfo("Done saving data, exiting")

if __name__ == '__main__':
    try:
        recorder = TeleopRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass 