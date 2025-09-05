#!/usr/bin/env python  # Use Python 2.7

from __future__ import print_function, division  # Python 2 compatibility

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import sys for path manipulation
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def save_frames_as_gif(frames, path='./', filename='depth_animation.gif', dpi=84, use_colormap=True, vmin=0, vmax=1):
    """
    Save a list of depth frames as a GIF animation.
    
    Args:
        frames: List of depth image arrays (expected to be in the range [0,1] or [0,255])
        path: Directory to save the GIF
        filename: Name of the output GIF file
        dpi: Resolution of the output GIF
        use_colormap: Whether to apply a colormap to the depth frames (set to False if frames are already colorized)
        vmin: Minimum value for normalization (if frames are already normalized [0,1], use 0)
        vmax: Maximum value for normalization (if frames are already normalized [0,1], use 1)
    
    Returns:
        Path to the saved GIF file
    """
    try:
        # Make sure the directory exists
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Create full output path
        output_path = os.path.join(path, filename)
        
        # Create a figure with exact 84x84 pixels (no resizing)
        plt.figure(figsize=(1, 1), dpi=int(dpi))
        
        # For the first frame, determine if we need to normalize
        sample_frame = frames[0]
        
        # Check if frames are already colorized (have 3 channels from cv2.applyColorMap)
        frames_are_colorized = len(sample_frame.shape) == 3 and sample_frame.shape[2] == 3
        
        if frames_are_colorized:
            # Convert BGR (OpenCV) to RGB (matplotlib)
            processed_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            # Normalize if needed
            if sample_frame.dtype == np.uint8:
                # No need to normalize further as imshow can handle uint8 RGB
                pass
        else:
            # Original grayscale depth processing
            if sample_frame.dtype == np.uint8 and vmax == 1:
                processed_frames = [frame.astype(np.float32) / 255.0 for frame in frames]
            else:
                processed_frames = frames
        
        # Apply colormap if requested and frames are not already colorized
        if use_colormap and not frames_are_colorized:
            # Create a colormap version for better visualization
            cmap = plt.get_cmap('jet')
            colored_frames = [cmap(frame) for frame in processed_frames]
            patch = plt.imshow(colored_frames[0])
        else:
            # Display already colored frames or grayscale
            if frames_are_colorized:
                patch = plt.imshow(processed_frames[0])
            else:
                patch = plt.imshow(processed_frames[0], cmap='gray', vmin=vmin, vmax=vmax)
            
        plt.axis('off')
        
        def animate(i):
            if use_colormap and not frames_are_colorized:
                patch.set_data(colored_frames[i])
            else:
                patch.set_data(processed_frames[i])
                
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(processed_frames), interval=50)
        anim.save(output_path, writer='pillow', fps=30)
        
        # Close the figure to free memory
        plt.close()
        
        rospy.loginfo("GIF saved to %s", output_path)
        return output_path
    
    except Exception as e:
        rospy.logerr("Error creating GIF: %s", str(e))
        return None

# Instead of using cv_bridge, use numpy directly
def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_np = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, -1)
    
    # If the image has only one channel (like depth)
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze(axis=2)
    
    return image_np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Make sure these match the environment's dimensions
        self.num_frames = 4 #if using policy2.pt, change num_frames to 4
        self.depth_height = 84
        self.depth_width = 84
        
        # Calculate the exact size of the input
        self.input_size = self.num_frames * self.depth_height * self.depth_width

        # CNN for depth processing (stacked frames only)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.num_frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Determine the output size of the CNN for a dummy input.
        with torch.no_grad():
            dummy_depth = torch.zeros(1, self.num_frames, self.depth_height, self.depth_width)
            cnn_output_size = self.cnn(dummy_depth).shape[1]

        # Critic head: accepts only the CNN depth features.
        # This is needed to match the saved model structure
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        # Actor head: accepts only the CNN depth features.
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def process_obs(self, x):
        """
        Process the observation containing only the stacked depth frames.
        The input `x` is expected to be a flattened tensor of size (batch, num_frames * depth_height * depth_width).
        """
        # Reshape x to (batch, num_frames, depth_height, depth_width)
        depth = x.reshape(-1, self.num_frames, self.depth_height, self.depth_width)
        depth_features = self.cnn(depth)
        return depth_features

    def get_value(self, x):
        depth_features = self.process_obs(x)
        return self.critic_head(depth_features)

    def get_action(self, x):
        depth_features = self.process_obs(x)
        action_mean = self.actor_head(depth_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.mean

    def get_action_and_value(self, x, action=None):
        depth_features = self.process_obs(x)
        action_mean = self.actor_head(depth_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)

class TurtlebotPolicyRunner:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_policy_runner')
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, default="~/turtlebot_ws/src/scripts/policy_nn/policy1.pt",
                           help="path to the saved model file")
        parser.add_argument("--tb2-speed", type=float, default=0.3,
                           help="maximum speed of the TB2 robot")
        parser.add_argument("--observation-dim", type=int, default=28224,  # 4*84*84
                           help="dimension of observation space")
        parser.add_argument("--action-dim", type=int, default=2,
                           help="dimension of action space")
        parser.add_argument("--record-video", action="store_true",
                           help="record a 10-second video of depth images")
        parser.add_argument("--record-gif", action="store_true",
                           help="record depth images and create a GIF")
        parser.add_argument("--num-frames", type=int, default=4,
                           help="number of frames to stack")
        parser.add_argument("--save-original-depth", action="store_true",
                           help="save original depth images")
        args = parser.parse_args(rospy.myargv()[1:])
        
        self.model_path = os.path.expanduser(args.model_path)
        self.max_speed = args.tb2_speed
        self.record_video = args.record_video
        self.record_gif = args.record_gif
        self.save_original_depth = args.save_original_depth
        # GIF recording variables
        self.gif_frames = []
        self.gif_recording = False
        self.max_gif_frames = 100  # Maximum number of frames to capture for GIF
        self.gif_capture_interval = 5  # Capture every 5th frame
        self.frame_count = 0
        self.gif_output_dir = os.path.expanduser("~/turtlebot_ws/src/scripts/gifs")
        
        
        # Video recording variables
        self.video_writer = None
        self.recording_start_time = None
        self.recording_duration = 10.0  # 10 seconds
        self.video_output_path = os.path.expanduser("/home/turtlebot02/turtlebot_ws/src/scripts/videos/turtlebot_depth_video.mp4")
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # State variables
        self.depth_frames = []
        self.num_frames = args.num_frames # Number of frames to stack
        self.depth_height = 84
        self.depth_width = 84
        self.last_observation_time = 0
        self.observation_timeout = 0.5  # seconds
        self.camera_timeout = 2.0  # seconds
        self.last_camera_time = 0
        self.stopped = False
        
        # Add these variables to the __init__ method
        self.last_frame_time = 0
        self.frame_add_interval = 1.0/30.0  # 30Hz - add a frame every ~33ms
        
        # Initialize agent
        self.obs_dim = args.observation_dim
        self.action_dim = args.action_dim
        self.agent = Agent(self.obs_dim, self.action_dim)
        self.load_model()
        
        # Safety parameters
        self.max_linear_speed = args.tb2_speed  # m/s
        self.max_angular_speed = 1  # rad/s
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback, queue_size=1)
        
        # Control rate (Hz)
        self.rate = rospy.Rate(30)  # 10 Hz
        
        rospy.loginfo("Policy runner initialized, waiting for sensor data...")
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                
                # If the checkpoint is the entire agent
                if isinstance(checkpoint, dict) and "actor" in checkpoint:
                    self.agent.load_state_dict(checkpoint)
                # If it's just state_dict
                else:
                    self.agent.load_state_dict(checkpoint)
                
                self.agent.eval()  # Set to evaluation mode
                rospy.loginfo("Successfully loaded policy from %s", self.model_path)
            else:
                rospy.logwarn("Policy file not found at %s, using random policy", self.model_path)
        except Exception as e:
            rospy.logerr("Failed to load policy: %s", str(e))
            rospy.logwarn("Using random policy instead")
            # Initialize with random weights
            for param in self.agent.parameters():
                param.data.normal_(0, 0.01)
                
    def start_gif_recording(self):
        """Start recording depth images for a GIF"""
        if not self.gif_recording:
            self.gif_frames = []
            self.frame_count = 0
            self.gif_recording = True
            rospy.loginfo("Started recording frames for GIF")
    
    def stop_gif_recording(self):
        """Stop recording frames and create a GIF"""
        if self.gif_recording and len(self.gif_frames) > 0:
            self.gif_recording = False
            timestamp = int(time.time())
            
            rospy.loginfo("Creating GIF with %d frames...", len(self.gif_frames))
            
            # Create GIF from frames
            filename = f"depth_gif_{timestamp}.gif"
            output_path = save_frames_as_gif(
                self.gif_frames,
                path=self.gif_output_dir,
                filename=filename,
                dpi=84,  # Match the depth image resolution
                use_colormap=False  # Don't apply colormap since we're already using cv2.applyColorMap
            )
            
            if output_path:
                rospy.loginfo("GIF saved to %s", output_path)
            else:
                rospy.logerr("Failed to save GIF")
                
            # Clear frames to free memory
            self.gif_frames = []
            
    def depth_callback(self, msg):
        try:
            # Check if the message is a 16-bit depth image
            is_16bit = msg.encoding == '16UC1'
            
            # Convert ROS Image to OpenCV image
            if is_16bit:
                # For 16-bit depth images
                depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            else:
                # For 8-bit images or other formats
                depth_array = imgmsg_to_cv2(msg)
            
            # Convert RealSense depth values (typically in millimeters) to meters
            depth_meters = depth_array.astype(np.float32) / 1000.0
            
            # PyBullet depth camera parameters - MATCH THESE TO YOUR PYBULLET SETTINGS
            pybullet_near = 0.1  # Near plane in meters
            pybullet_far = 10.0  # Far plane in meters
            
            # Clip depth values to valid range
            depth_meters = np.clip(depth_meters, pybullet_near, pybullet_far)
            
            # Convert to PyBullet's depth buffer format
            # PyBullet's depth buffer is in range [0,1] where 1 is far and 0 is near
            depth_pybullet = (pybullet_far + pybullet_near - (2.0 * pybullet_near * pybullet_far) / depth_meters) / (pybullet_far - pybullet_near)
            
            # Normalize to [0,1] range, matching PyBullet's depth_array format
            depth_pybullet = np.clip(depth_pybullet, 0.0, 1.0)
            
            # Convert to 8-bit for visualization and processing
            depth_image = (depth_pybullet * 255).astype(np.uint8)
            
            # Resize depth image to match the expected input size
            depth_image = cv2.resize(depth_image, (self.depth_width, self.depth_height))

            # Add frames at controlled rate
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_add_interval:
                # Add to frame stack
                self.depth_frames.append(depth_image)
                if len(self.depth_frames) > self.num_frames:
                    self.depth_frames.pop(0)
                self.last_frame_time = current_time
            
            self.last_camera_time = time.time()
            
            # Create a colormap version of the depth image for better visualization
            depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            # Create a resized version for video (if needed)
            depth_colormap_video = cv2.resize(depth_colormap, (320, 240))
            
            # Record GIF frame if we're recording GIFs
            if self.gif_recording:
                self.frame_count += 1
                # Capture every nth frame to avoid too many frames
                if self.frame_count % self.gif_capture_interval == 0:
                    # Use the original 84x84 depth_colormap for the GIF
                    self.gif_frames.append(depth_colormap.copy())
                    
                    # Check if we've reached the maximum number of frames
                    if len(self.gif_frames) >= self.max_gif_frames:
                        self.stop_gif_recording()
            
            # Record video frame if we're recording
            if self.recording_start_time is not None and self.video_writer is not None:
                # Write the resized colormap to the video
                self.video_writer.write(depth_colormap_video)
                
                # Check if we've recorded for the desired duration
                if time.time() - self.recording_start_time >= self.recording_duration:
                    self.stop_recording()
            
            # Log first frame and then every 5 seconds
            current_time = time.time()
            if len(self.depth_frames) == 1 or (int(current_time) % 5 == 0 and int(current_time * 10) % 10 == 0):
                rospy.loginfo("Received depth frame %d. Image shape: %s", 
                             len(self.depth_frames), 
                             str(depth_image.shape))
        except Exception as e:
            rospy.logerr("Error processing depth image: %s", str(e))
    
    def start_recording(self):
        """Start recording depth images to a video file"""
        if self.recording_start_time is None:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4 format
            self.video_writer = cv2.VideoWriter(
                self.video_output_path, 
                fourcc, 
                30.0,  # FPS
                (320, 240)
            )
            self.recording_start_time = time.time()
            rospy.loginfo("Started recording depth video to %s", self.video_output_path)
    
    def stop_recording(self):
        """Stop recording and release the video writer"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            recording_duration = time.time() - self.recording_start_time
            self.recording_start_time = None
            rospy.loginfo("Finished recording depth video (%.1f seconds) to %s", 
                         recording_duration, self.video_output_path)
    
    def get_observation(self):
        # Check if we have enough depth frames
        if len(self.depth_frames) < self.num_frames:
            return None
        
        # Check if we have recent camera data
        current_time = time.time()
        if current_time - self.last_camera_time > self.camera_timeout:
            if not self.stopped:
                rospy.logwarn("Camera timeout (%.1f sec). Stopping robot.", 
                            current_time - self.last_camera_time)
                self.stop_robot()
                self.stopped = True
            return None
        
        # Stack frames and format observation to match what was used during training
        stacked_frames = np.stack(self.depth_frames, axis=0)
        
        # Normalize the stacked frames to [0,1] range
        # The depth_frames are uint8 (0-255), so divide by 255.0
        stacked_frames_normalized = stacked_frames.astype(np.float32) / 255.0
        
        # Flatten the normalized frames
        observation = stacked_frames_normalized.flatten()
        
        # Log min and max values of the observation
        min_val = np.min(observation)
        max_val = np.max(observation)
        mean_val = np.mean(observation)
        std_val = np.std(observation)
        
        # Log every 5 seconds to avoid flooding the console
        if int(current_time) % 5 == 0 and int(current_time * 10) % 10 == 0:
            rospy.loginfo("Observation stats - min: %.3f, max: %.3f, mean: %.3f, std: %.3f", 
                         min_val, max_val, mean_val, std_val)
        
        self.last_observation_time = current_time
        self.stopped = False
        return observation
    
    def run(self):
        rospy.loginfo("Starting policy runner loop...")
        
        # Wait until we have enough frames
        rospy.loginfo("Waiting for depth frames...")
        while not rospy.is_shutdown() and len(self.depth_frames) < self.num_frames:
            rospy.sleep(0.1)
            
        if len(self.depth_frames) >= self.num_frames:
            rospy.loginfo("Received initial depth frames. Starting control loop.")
        else:
            rospy.logerr("Shutdown requested before receiving depth frames.")
            return
        
        # Start recording if requested
        if self.record_video:
            self.start_recording()
        # Start GIF recording if requested
        if self.record_gif:
            self.start_gif_recording()
            
        while not rospy.is_shutdown():
            # Get current observation
            obs = self.get_observation()
            
            current_time = time.time()
            if obs is not None:
                try:
                    # Convert observation to tensor
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
                    
                    # Get action from policy
                    with torch.no_grad():
                        action = self.agent.get_action(obs_tensor)
                    
                    # Convert action to numpy array
                    action_np = action.cpu().numpy().flatten()
                    
                    # Convert action to Twist message with safety limits
                    twist_msg = self.action_to_twist(action_np)
                    
                    # for _ in range(2):
                    #     self.cmd_vel_pub.publish(twist_msg)
                    #     rospy.sleep(0.1)
                    # Publish command
                    self.cmd_vel_pub.publish(twist_msg)
                    
                    # Log occasionally for debugging (once every ~5 seconds)
                    if int(current_time) % 1 == 0 and int(current_time * 10) % 10 == 0:
                        # Calculate wheel velocities from the action
                        wheel_radius = 0.038  # meters
                        max_wheel_velocity = self.max_linear_speed / wheel_radius  # rad/s
                        left_wheel_vel = float(action_np[0]) * max_wheel_velocity
                        right_wheel_vel = float(action_np[1]) * max_wheel_velocity
                
                except Exception as e:
                    rospy.logerr("Error in policy execution: %s", str(e))
                    self.stop_robot()
            elif current_time - self.last_observation_time > self.observation_timeout:
                # Stop the robot if we haven't had a good observation for a while
                self.stop_robot()
            
            self.rate.sleep()
        
        # Make sure to stop recording if we exit the loop
        if self.video_writer is not None:
            self.stop_recording()
                        
        if self.gif_recording:
            self.stop_gif_recording()
        rospy.loginfo("Policy runner loop ended")
    
    def action_to_twist(self, action):
        """
        Convert policy action to Twist message by first converting to wheel velocities,
        then back to linear and angular velocity.
        
        In the simulation:
        - action[0] controls left wheel velocity
        - action[1] controls right wheel velocity
        """
        # # Robot parameters (same as in simulation)
        wheel_separation = 0.23  # Distance between wheels in meters
        wheel_radius = 0.038     # Wheel radius in meters
        
        linear_vel = action[0]
        angular_vel = action[1]
        
        # Clip actions 
        linear_vel = np.clip(linear_vel, -self.max_linear_speed, self.max_linear_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        # Create Twist message
        twist = Twist()
        twist.linear.x = linear_vel 
        twist.angular.z = angular_vel 
        rospy.loginfo("Twist: [%.3f, %.3f]", twist.linear.x, twist.angular.z)
        
        # Log the raw action values, wheel velocities, and resulting robot velocities
        rospy.logdebug("Raw action: [%.3f, %.3f]", 
                      linear_vel, angular_vel)
        
        return twist
    
    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)

if __name__ == '__main__':
    try:
        runner = TurtlebotPolicyRunner()
        runner.run()
    except rospy.ROSInterruptException:
        # Make sure to stop recording if there's an exception
        if hasattr(runner, 'video_writer') and runner.video_writer is not None:
            runner.stop_recording()
        pass