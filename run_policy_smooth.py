#!/usr/bin/env python  # Use Python 2.7

from __future__ import print_function, division  # Python 2 compatibility

import os
import argparse
import numpy as np
import torch
import pickle
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
from image_smoother import ImageSmoother
import traceback


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
    def __init__(self, obs_dim, action_dim, num_frames):
        super().__init__()
        
        # Make sure these match the environment's dimensions
        self.num_frames = num_frames
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
        
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(self.num_frames, 16, 8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        #     nn.Conv2d(16, 32, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        #     # nn.Conv2d(32, 32, 5),
        #     # nn.ReLU(),
        #     # nn.MaxPool2d(2,2),
        #     nn.Flatten()
        # )
 

        # Determine the output size of the CNN for a dummy input.
        with torch.no_grad():
            dummy_depth = torch.zeros(1, self.num_frames, self.depth_height, self.depth_width)
            cnn_output_size = self.cnn(dummy_depth).shape[1]

        # Critic head: accepts only the CNN depth features.
        # Updated to match saved model's architecture
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        # Actor head: accepts only the CNN depth features.
        # Updated to match saved model's architecture
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 256)),
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

class LSTM_Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, num_frames):
        super().__init__()
        self.num_frames = num_frames
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
            print(cnn_output_size)

        self.lstm = nn.LSTM(cnn_output_size, 512)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Critic head: accepts only the CNN depth features.
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        # Actor head: accepts only the CNN depth features.
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 2), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))

    def process_obs(self, x):
        """
        Process the observation containing only the stacked depth frames.
        The input `x` is expected to be a flattened tensor of size (batch, num_frames * depth_height * depth_width).
        """
        # Check if the input size matches what we expect
        #print(x.shape)
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: got {x.shape[1]}, expected {self.input_size}")
            
        # Reshape x to (batch, num_frames, depth_height, depth_width)
        depth = x.reshape(-1, self.num_frames, self.depth_height, self.depth_width)
        depth_features = self.cnn(depth)
        #print(depth_features.shape)
        return depth_features

    def get_states(self, x, lstm_state, done):
        hidden = self.process_obs(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size)).long()
        new_hidden = []
        #print(done)
        for h, d in zip(hidden, done):
            #print(d)
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (~d).view(1, -1, 1) * lstm_state[0],
                    (~d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic_head(hidden)
    
    def get_mean_std(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        value = self.critic_head(hidden)
        
        action_mean = self.actor_head(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return action_mean, action_std
        

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        value = self.critic_head(hidden)

        action_mean = self.actor_head(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
            lstm_state
        )

class TurtlebotPolicyRunner:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_policy_runner')
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, default="~/turtlebot_ws/src/scripts/policy_nn/fewfew.pt",
                           help="path to the saved model file")
        parser.add_argument("--tb2-speed", type=float, default=0.3,
                           help="maximum speed of the TB2 robot")
        parser.add_argument("--observation-dim", type=int, default=28224,  # 4*84*84
                           help="dimension of observation space")
        parser.add_argument("--action-dim", type=int, default=2,
                           help="dimension of action space")
        parser.add_argument("--record-video", action="store_true",
                           help="record a 30-second video of RGB images")
        parser.add_argument("--record-gif", action="store_true",
                           help="record depth images and create a GIF")
        parser.add_argument("--save-original-depth", action="store_true",
                           help="save original depth images")
        parser.add_argument("--original-depth-dir", type=str, default="~/turtlebot_ws/src/scripts/original_depth_frames",
                           help="directory to save original depth images")
        parser.add_argument("--use-depth", action="store_true",
                           help="use depth edges")
        parser.add_argument("--smoothing", action="store_true",
                    help="smooth observations or not")
        parser.add_argument("--use-lstm", action="store_true",
                    help="use lstm or not")
        parser.add_argument("--max-angular-speed", type=float, default=0.2,
                    help="maximum angular speed of the TB2 robot")
        args = parser.parse_args(rospy.myargv()[1:])
        
        self.model_path = os.path.expanduser(args.model_path)
        self.max_speed = args.tb2_speed
        self.record_video = args.record_video
        self.save_original_depth = args.save_original_depth
        self.smoothing = args.smoothing
        self.record_gif = args.record_gif
        self.use_depth = args.use_depth
        self.use_lstm = args.use_lstm
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
        self.recording_duration = 30.0  # 10 seconds
        self.video_output_path = os.path.expanduser(f"/home/turtlebot02/turtlebot_ws/src/scripts/videos/turtlebot__video_{time.time()}.mp4")
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # State variables
        self.depth_frames = []
        self.num_frames = 8  # Number of frames to stack
        self.depth_height = 84
        self.depth_width = 84
        self.last_observation_time = 0
        self.observation_timeout = 0.5  # seconds
        self.camera_timeout = 2.0  # seconds
        self.last_camera_time = 0
        self.stopped = False
        
        # Initialize agent
        self.obs_dim = args.observation_dim
        self.action_dim = args.action_dim
        
        if self.use_lstm:
            print("Using LSTM")
            self.agent = LSTM_Agent(self.obs_dim, self.action_dim, self.num_frames)
        else:   
            self.agent = Agent(self.obs_dim, self.action_dim, self.num_frames)
            
        self.load_model()
        
        # Safety parameters
        self.max_linear_speed = args.tb2_speed  # m/s
        self.max_angular_speed = args.max_angular_speed  # rad/s
        
        # RGB video recording variables
        self.rgb_frame = None
        self.last_rgb_time = 0
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback, queue_size=1)
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback, queue_size=1)
        
        # Control rate (Hz)
        self.rate = rospy.Rate(30)  # 10 Hz
        
        
        # Additional arguments
        self.save_original_depth = args.save_original_depth
        self.original_depth_dir = os.path.expanduser(args.original_depth_dir)
        
        # Create original depth directory if needed
        if self.save_original_depth and not os.path.exists(self.original_depth_dir):
            os.makedirs(self.original_depth_dir)
            rospy.loginfo(f"Created directory for original depth images: {self.original_depth_dir}")
        
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
    
    def add_smoothed_frame_to_gif(self, smoothed_obs):
        """Add a smoothed observation frame to the GIF recording"""
        if self.gif_recording:
            self.frame_count += 1
            # Capture every nth frame to avoid too many frames
            if self.frame_count % self.gif_capture_interval == 0:
                try:
                    # Reshape smoothed observation back to frames
                    # Get only the last frame from the stacked frames
                    frame_size = self.depth_height * self.depth_width
                    last_frame = smoothed_obs[-frame_size:]
                    
                    # Reshape to 2D image
                    depth_img = last_frame.reshape(self.depth_height, self.depth_width)
                    
                    # Convert back to uint8
                    depth_img = (depth_img * 255).astype(np.uint8)
                    
                    # Apply colormap for visualization
                    depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
                    
                    # Add to frames
                    self.gif_frames.append(depth_colormap.copy())
                    
                    # Check if we've reached the maximum number of frames
                    if len(self.gif_frames) >= self.max_gif_frames:
                        self.stop_gif_recording()
                        
                except Exception as e:
                    rospy.logerr("Error adding smoothed frame to GIF: %s", str(e))
    
    def plot_action_histogram(self, action_list):
        """
        Plot histograms of linear and angular velocities and save them as image files.
        
        Args:
            action_list: List of actions, where each action is [linear_vel, angular_vel]
        """
        try:
            if not action_list:
                rospy.logwarn("No actions to plot histogram for")
                return
                
            # Convert list of actions to numpy array for easier handling
            actions = np.array(action_list)
            
            # Create output directory if it doesn't exist
            histogram_dir = os.path.expanduser("~/turtlebot_ws/src/scripts/histograms")
            if not os.path.exists(histogram_dir):
                os.makedirs(histogram_dir)
                
            timestamp = int(time.time())
            
            # Plot histogram of linear velocities
            plt.figure(figsize=(10, 6))
            plt.hist(actions[:, 0], bins=30, alpha=0.7, color='#440154')  # Dark purple from viridis
            plt.title('Histogram of Linear Velocities')
            plt.xlabel('Linear Velocity (m/s)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_linear = np.mean(actions[:, 0])
            std_linear = np.std(actions[:, 0])
            plt.axvline(mean_linear, color='#21918c', linestyle='dashed', linewidth=2, label=f'Mean: {mean_linear:.3f}')
            plt.text(0.7, 0.85, f'Mean: {mean_linear:.3f}\nStd Dev: {std_linear:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            plt.legend()
            
            linear_histogram_path = os.path.join(histogram_dir, f'linear_vel_histogram_{timestamp}.png')
            plt.savefig(linear_histogram_path)
            plt.close()
            
            # Plot histogram of angular velocities
            plt.figure(figsize=(10, 6))
            plt.hist(actions[:, 1], bins=30, alpha=0.7, color='#440154')  # Dark purple from viridis
            plt.title('Histogram of Angular Velocities')
            plt.xlabel('Angular Velocity (rad/s)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_angular = np.mean(actions[:, 1])
            std_angular = np.std(actions[:, 1])
            plt.axvline(mean_angular, color='#21918c', linestyle='dashed', linewidth=2, label=f'Mean: {mean_angular:.3f}')
            plt.text(0.7, 0.85, f'Mean: {mean_angular:.3f}\nStd Dev: {std_angular:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            plt.legend()
            
            angular_histogram_path = os.path.join(histogram_dir, f'angular_vel_histogram_{timestamp}.png')
            plt.savefig(angular_histogram_path)
            plt.close()
            
            # Plot 2D histogram (heatmap) of linear vs angular velocities using viridis colormap
            plt.figure(figsize=(10, 8))
            plt.hist2d(actions[:, 0], actions[:, 1], bins=30, cmap='viridis', norm=plt.Normalize(0, 30))
            plt.colorbar(label='Frequency')
            plt.title('2D Histogram of Linear vs Angular Velocities')
            plt.xlabel('Linear Velocity (m/s)')
            plt.ylabel('Angular Velocity (rad/s)')
            plt.grid(True, alpha=0.3)
            
            # Add mean point
            plt.plot(mean_linear, mean_angular, 'r+', markersize=15, label='Mean Action')
            plt.legend()
            
            combined_histogram_path = os.path.join(histogram_dir, f'combined_vel_histogram_{timestamp}.png')
            plt.savefig(combined_histogram_path)
            plt.close()
            
            rospy.loginfo(f"Saved action histograms to {histogram_dir}")
            
        except Exception as e:
            rospy.logerr(f"Error plotting action histogram: {str(e)}")

    def stop_gif_recording(self):
        """Stop recording frames and create a GIF"""
        if self.gif_recording and len(self.gif_frames) > 0:
            self.gif_recording = False
            timestamp = int(time.time())
            filename = f"depth_gif_smoothed_{timestamp}.gif"
            
            rospy.loginfo("Creating GIF with %d frames...", len(self.gif_frames))
            
            # Save the GIF
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
            
            # Plot and save action histograms
            if hasattr(self, 'action_store_list') and len(self.action_store_list) > 0:
                self.plot_action_histogram(self.action_store_list)
                
                # Save the actions to a pickle file
                try:
                    # Create directory if it doesn't exist
                    actions_dir = os.path.expanduser("~/turtlebot_ws/src/scripts/actions")
                    if not os.path.exists(actions_dir):
                        os.makedirs(actions_dir)
                        
                    action_file_path = os.path.join(actions_dir, f"actions_{timestamp}.pkl")
                    with open(action_file_path, 'wb') as f:
                        pickle.dump(self.action_store_list, f)
                    rospy.loginfo(f"Saved {len(self.action_store_list)} actions to {action_file_path}")
                except Exception as e:
                    rospy.logerr(f"Error saving actions to pickle file: {str(e)}")

    def depth_callback(self, msg):
        smoother = ImageSmoother()
        try:
            # Check if the message is a 16-bit depth image
            is_16bit = msg.encoding == '16UC1'
            
            # Convert ROS Image to OpenCV image
            if is_16bit:
                # For 16-bit depth images
                # depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(480, 848)
                # Save original depth image if requested
                if self.save_original_depth and not hasattr(self, 'original_saved'):
                    timestamp = int(time.time())
                    original_filename = os.path.join(self.original_depth_dir, f"original_depth_{timestamp}.npy")
                    np.save(original_filename, depth_array)
                    rospy.loginfo(f"Saved original depth array to {original_filename}")
                    self.original_saved = True
            else:
                # For 8-bit images or other formats
                depth_array = imgmsg_to_cv2(msg)
                
                # Save original depth image if requested
                if self.save_original_depth and not hasattr(self, 'original_saved'):
                    timestamp = int(time.time())
                    original_filename = os.path.join(self.original_depth_dir, f"original_depth_{timestamp}.npy")
                    np.save(original_filename, depth_array)
                    rospy.loginfo(f"Saved original depth array to {original_filename}")
                    self.original_saved = True
            
            # Get dimensions of the original image
            height, width = depth_array.shape
            
            # Crop to 480x480 from the center
            start_x = max(0, (width - 480) // 2)
            start_y = max(0, (height - 480) // 2)
            cropped_depth = depth_array[start_y:start_y+480, start_x:start_x+480]
            
            # Continue with the rest of your processing on the cropped image
            # Convert RealSense depth values (typically in millimeters) to meters
            depth_meters = cropped_depth.astype(np.float32) / 1000.0
            
            # PyBullet depth camera parameters - MATCH THESE TO YOUR PYBULLET SETTINGS
            pybullet_near = 0.1  # Near plane in meters
            pybullet_far = 8.0  # Far plane in meters
            
            # Clip depth values to valid range
            depth_meters = np.clip(depth_meters, pybullet_near, pybullet_far)
            
            # Convert to PyBullet's depth buffer format
            depth_pybullet = (pybullet_far + pybullet_near - (2.0 * pybullet_near * pybullet_far) / depth_meters) / (pybullet_far - pybullet_near)
            
            # Normalize to [0,1] range
            depth_pybullet = np.clip(depth_pybullet, 0.0, 1.0)
            
            # Convert to 8-bit for visualization and processing
            depth_image = (depth_pybullet * 255).astype(np.uint8)
            
            # Now resize the 480x480 cropped image to 84x84
            depth_image = cv2.resize(depth_image, (self.depth_width, self.depth_height))
            
            # # Save a single depth frame (you can add a condition to save only once)
            # if self.save_original_depth and not hasattr(self, 'frame_saved') or not self.frame_saved:
            #     # Create directory if it doesn't exist
            #     save_dir = os.path.expanduser("~/turtlebot_ws/src/scripts/depth_frames")
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     timestamp = int(time.time())
                    
            #     # Save both raw depth image and colorized version
            #     raw_filename = os.path.join(save_dir, f"depth_raw_{timestamp}.png")
            #     color_filename = os.path.join(save_dir, f"depth_color_{timestamp}.png")
            #     # Save raw depth image
            #     cv2.imwrite(raw_filename, depth_image)
                
            #     # Save colorized depth image
            #     depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            #     cv2.imwrite(color_filename, depth_colormap)
                
            #     rospy.loginfo(f"Saved depth frames to {raw_filename} and {color_filename}")
                
            #     # Set flag to avoid saving more frames
            #     self.frame_saved = True
            
            # Add to frame stack
            self.depth_frames.append(depth_image)
            if len(self.depth_frames) > self.num_frames:
                self.depth_frames.pop(0)
                
            self.last_camera_time = time.time()
            
            # Create a colormap version of the depth image for better visualization
            depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        except Exception as e:
            rospy.logerr("Error processing depth image: %s", str(e))
    
    def rgb_callback(self, msg):
        """Process RGB images for video recording"""
        try:
            # Convert ROS Image to OpenCV image using custom function to avoid cv_bridge issues
            rgb_image = imgmsg_to_cv2(msg)
            
            # Check if image is RGB and convert to BGR for OpenCV/video writing
            if msg.encoding == 'rgb8' and len(rgb_image.shape) == 3:
                # Convert RGB to BGR for OpenCV
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                # Already in BGR format, no conversion needed
                pass
            else:
                rospy.logwarn("Unexpected image encoding: %s", msg.encoding)
            
            # Store the latest RGB frame for video recording
            self.rgb_frame = rgb_image.copy()
            self.last_rgb_time = time.time()
            
            # Record video frame if we're recording
            if self.recording_start_time is not None and self.video_writer is not None:
                # Resize for video if needed
                rgb_resized = cv2.resize(rgb_image, (640, 480))  # Use higher resolution for RGB
                self.video_writer.write(rgb_resized)
                
                # Check if we've recorded for the desired duration
                if time.time() - self.recording_start_time >= self.recording_duration:
                    self.stop_recording()
                    
        except Exception as e:
            rospy.logerr("Error processing RGB image: %s", str(e))
            import traceback
            traceback.print_exc()
    
    def start_recording(self):
        """Start recording RGB images to a video file"""
        if self.recording_start_time is None:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4 format
            self.video_writer = cv2.VideoWriter(
                self.video_output_path, 
                fourcc, 
                30.0,  # FPS
                (640, 480)  # Higher resolution for RGB video
            )
            self.recording_start_time = time.time()
            rospy.loginfo("Started recording RGB video to %s", self.video_output_path)
    
    def stop_recording(self):
        """Stop recording and release the video writer"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            recording_duration = time.time() - self.recording_start_time
            self.recording_start_time = None
            rospy.loginfo("Finished recording RGB video (%.1f seconds) to %s", 
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
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Robot stopped")

    def run(self):
        self.action_store_list = []  # Initialize action store list
        rospy.loginfo("Starting policy runner loop...")
        smoother = ImageSmoother()
        
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
            
        if self.use_lstm:
            lstm_state = (
                torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size),
                torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size),
            ) 
            
        while not rospy.is_shutdown():
            # Get current observation
            done = torch.zeros(1, 1, dtype=torch.long)  # Initialize done as a tensor
            obs = self.get_observation()
            if obs is not None:
                if self.smoothing:
                    # Apply smoothing
                    smoothed_obs = smoother.Smooth(obs, use_depth=self.use_depth)
                else:
                    smoothed_obs = obs
                
                # Prepare the latest frame for plus sign detection
                # Extract the last frame from smoothed_obs (which is flattened and normalized)
                frame_size = self.depth_height * self.depth_width
                latest_frame_flat = smoothed_obs[-frame_size:]
                latest_frame_2d_normalized = latest_frame_flat.reshape(self.depth_height, self.depth_width)
                # Convert to uint8 image (0-255 range)
                latest_frame_uint8 = (latest_frame_2d_normalized * 255).astype(np.uint8)
                
                # Create a colormap version for plus sign detection
                # This is better than grayscale for color-based detection
                depth_colormap = cv2.applyColorMap(latest_frame_uint8, cv2.COLORMAP_JET)
                
                # Try to detect a red plus sign in the colormap image
                # sign = self.detect_red_plus_sign(depth_colormap)
                
                # if sign:
                #     rospy.loginfo("Plus sign detected!")
                # else:
                #     rospy.loginfo("No plus sign detected")
                    
                # Record GIF frame if we're recording GIFs - now using smoothed observation
                if self.gif_recording:
                    self.add_smoothed_frame_to_gif(smoothed_obs)
                
                try:
                    # Convert observation to tensor
                    obs_tensor = torch.FloatTensor(smoothed_obs).unsqueeze(0)  # Add batch dimension
                    
                    # Get action from policy
                    with torch.no_grad():
                        if self.use_lstm:
                            action, _, _, _, lstm_state = self.agent.get_action_and_value(obs_tensor, lstm_state, done)
                        else:
                            action = self.agent.get_action(obs_tensor)
                    
                    # Convert action to numpy array
                    action_np = action.cpu().numpy().flatten()
                    
                    # Convert action to Twist message with safety limits
                    twist_msg = self.action_to_twist(action_np)
                    # twist_msg = self.action_to_twist_old(action_np)
                    
                    # Publish command
                    self.cmd_vel_pub.publish(twist_msg)
                    
                    # Log occasionally for debugging (once every ~5 seconds)
                    current_time = time.time()
                    if int(current_time) % 1 == 0 and int(current_time * 10) % 10 == 0:
                        linear_vel = np.clip(action_np[0], -self.max_linear_speed, self.max_linear_speed)
                        angular_vel = np.clip(action_np[1], -self.max_angular_speed, self.max_angular_speed)
                        rospy.loginfo("Action: linear=%.2f, angular=%.2f", linear_vel, angular_vel)
                        rospy.loginfo("Twist: linear=%.2f, angular=%.2f", twist_msg.linear.x, twist_msg.angular.z)
        
                except Exception as e:
                    rospy.logerr("Error in policy execution: %s", str(e))
                    self.stop_robot()
            elif time.time() - self.last_observation_time > self.observation_timeout:
                # Stop the robot if we haven't had a good observation for a while
                self.stop_robot()
                done = torch.ones(1, 1, dtype=torch.long)  # Set done to True when timeout occurs
            self.rate.sleep()
        
        # Make sure to stop recording if we exit the loop
        if self.video_writer is not None:
            self.stop_recording()
            
        if self.gif_recording:
            self.stop_gif_recording()
            """Save the stored actions to a pickle file"""
            # try:
            #     with open('/home/turtlebot02/turtlebot_ws/src/scripts/observations/actions.pkl', 'wb') as f:
            #         pickle.dump(self.action_store_list, f)
            #     rospy.loginfo(f"Saved {len(self.action_store_list)} actions to /home/turtlebot02/turtlebot_ws/src/scripts/observations/actions.pkl")
            # except Exception as e:
            #     rospy.logerr(f"Error saving actions: {str(e)}")

        rospy.loginfo("Policy runner loop ended")
    # old action to twist
    def action_to_twist_old(self, action):
        """
        Convert policy action to Twist message by first converting to wheel velocities,
        then back to linear and angular velocity.
        
        In the simulation:
        - action[0] controls left wheel velocity
        - action[1] controls right wheel velocity
        """
        # # Robot parameters (same as in simulation)
        # wheel_separation = 0.23  # Distance between wheels in meters
        wheel_radius = 0.038     # Wheel radius in meters
        
        linear_vel = action[0]
        angular_vel = action[1] 
        
        
        # Log the raw action values, wheel velocities, and resulting robot velocities
        rospy.loginfo("Raw action: [%.2f, %.2f]", 
                      linear_vel, angular_vel)
        
        # Clip actions 
        linear_vel = np.clip(linear_vel, -self.max_linear_speed, self.max_linear_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        # Store the clipped values in action_store_list
        self.action_store_list.append(np.array([linear_vel, angular_vel]))
        
        # linear_vel = 0.3
        # angular_vel = 0
        
        # Create Twist message
        # right is negative, left is positive
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel 

        
        # Log the raw action values, wheel velocities, and resulting robot velocities
        rospy.loginfo("Clipped action: [%.3f, %.3f]", 
                      linear_vel, angular_vel)
        
        return twist       
    def action_to_twist(self, action):
        """
        Convert policy action to Twist message by converting wheel velocities
        to linear and angular velocity.
        
        The policy outputs:
        - action[0]: raw wheel velocity (needs normalization)
        - action[1]: raw wheel velocity (needs normalization)
        """
        # Robot parameters (MUST match simulation exactly)
        wheel_separation = 0.287  # Distance between wheels in meters (match simulation)
        wheel_radius = 0.038     # Wheel radius in meters
        
        # Maximum wheel velocity in rad/s (match simulation scaling)
        max_wheel_vel = 7.4 
        
        # First normalize the raw actions to [-1, 1] range
        left_action_normalized = np.clip(action[0], -1.0, 1.0)
        right_action_normalized = np.clip(action[1], -1.0, 1.0)
        
        # Convert normalized actions from [-1, 1] to wheel velocities in rad/s
        left_wheel_vel_rads = left_action_normalized * max_wheel_vel   # rad/s
        right_wheel_vel_rads = right_action_normalized * max_wheel_vel  # rad/s
        
        # Log the raw actions, normalized actions, and converted wheel velocities
        rospy.loginfo("Raw actions: left=%.3f, right=%.3f", action[0], action[1])
        rospy.loginfo("Normalized actions: left=%.3f, right=%.3f", 
                      left_action_normalized, right_action_normalized)
        rospy.loginfo("Raw wheel velocities (rad/s): left=%.2f, right=%.2f", 
                      left_wheel_vel_rads, right_wheel_vel_rads)
        
        # Convert wheel velocities from rad/s to m/s
        left_wheel_vel_ms = left_wheel_vel_rads * wheel_radius
        right_wheel_vel_ms = right_wheel_vel_rads * wheel_radius
        
        # Log individual wheel velocities in m/s
        rospy.loginfo("Wheel velocities (m/s): left=%.3f, right=%.3f", 
                      left_wheel_vel_ms, right_wheel_vel_ms)
        
        # Convert wheel velocities to linear and angular velocity using differential drive kinematics
        linear_vel = (left_wheel_vel_ms + right_wheel_vel_ms) / 2.0 #m/s
        angular_vel = (right_wheel_vel_ms - left_wheel_vel_ms) / wheel_separation #rad/s
        
        # Log before clipping
        rospy.loginfo("Before clipping: linear=%.3f, angular=%.3f", 
                      linear_vel, angular_vel)
        
        # Clip velocities to safety limits
        linear_vel = np.clip(linear_vel, -self.max_linear_speed, self.max_linear_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        # Log after clipping
        rospy.loginfo("After clipping: linear=%.3f, angular=%.3f", 
                      linear_vel, angular_vel)
        
        # Store the clipped values in action_store_list
        self.action_store_list.append(np.array([linear_vel, angular_vel]))
        
        # Create Twist message
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel 
        
        return twist
if __name__ == '__main__':
    try:
        runner = TurtlebotPolicyRunner()
        runner.run()
    except rospy.ROSInterruptException:
        # Make sure to stop recording if there's an exception
        if hasattr(runner, 'video_writer') and runner.video_writer is not None:
            runner.stop_recording()
        pass