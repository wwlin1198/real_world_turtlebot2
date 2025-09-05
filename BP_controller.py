#!/usr/bin/env python

"""
BP_controller.py - TurtleBot2 Base Platform Controller

This is the controller for the Turtlebot2 base. It is capable of controlling the robot's base movement.
It is designed to be used in a multi-robot system as namespace is used to identify the robot. Before use,
make sure the namespace is set correctly for each robot so the ros_topics are differentiated 
(e.g. /robot1/cmd_vel_mux/input/navi, /robot2/cmd_vel_mux/input/navi).
Threading is used to send commands when both robots sample actions at the same time. Otherwise,
the commands will be sent sequentially. Claude is used to help with the development of this module.

Author: Wo Wei Lin
Date: 2025-08-27
"""

from __future__ import print_function, division
import rospy
from geometry_msgs.msg import Twist
import threading


class TurtleBotController:
    """
    TurtleBot2 Base Controller Class
    
    Provides functions to control TurtleBot2 movement including:
    - Basic movements (forward, backward, turn left/right, stop)
    - Velocity control with safety limits
    - Timed movements
    """
    
    def __init__(self, node_name='turtlebot_controller', namespace=''):
        """
        Initialize the TurtleBot controller
        
        Args:
            node_name (str): Name for the ROS node
            namespace (str): Robot namespace (e.g., 'robot1', 'robot2', or '' for no namespace)
        """
        # Initialize ROS node (only if not already initialized)
        try:
            rospy.init_node(node_name, anonymous=True)
        except rospy.exceptions.ROSException:
            # Node already initialized, which is fine for multiple controllers
            pass
        
        # Store namespace for this robot
        self.namespace = namespace
        if namespace and not namespace.startswith('/'):
            self.namespace = '/' + namespace
        
        # Create publisher for velocity commands with namespace
        cmd_vel_topic = self.namespace + '/cmd_vel_mux/input/navi' if namespace else '/cmd_vel_mux/input/navi'
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        
        # Safety parameters (m/s and rad/s)
        self.max_linear_speed = 0.3   # Maximum forward/backward speed
        self.max_angular_speed = 0.8  # Maximum rotation speed
        
        # Default movement speeds (conservative for smooth operation)
        self.default_linear_speed = 0.15  # Default forward/backward speed
        self.default_angular_speed = 1.17  # Default rotation speed
        
        # Angular movement calibration - compensate for actual vs commanded angular velocity
        self.angular_calibration_factor = 2.1  # Actual rotation is about half of commanded
        
        # Turn factors removed - each robot may have different characteristics
        
        # Rate for publishing commands
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Wait for publisher to be ready
        rospy.sleep(0.1)
        
        if namespace:
            rospy.loginfo("TurtleBot Controller initialized for %s (topic: %s)", namespace, cmd_vel_topic)
        else:
            rospy.loginfo("TurtleBot Controller initialized (topic: %s)", cmd_vel_topic)
    
    def _publish_velocity(self, linear_x=0.0, angular_z=0.0):
        """
        Publish velocity command to the robot
        
        Args:
            linear_x (float): Linear velocity in x direction (m/s)
            angular_z (float): Angular velocity around z axis (rad/s)
        """
        # Apply safety limits
        linear_x = max(-self.max_linear_speed, min(self.max_linear_speed, linear_x))
        angular_z = max(-self.max_angular_speed, min(self.max_angular_speed, angular_z))
        
        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = angular_z
        
        self.cmd_vel_pub.publish(twist_msg)
        
        rospy.logdebug("Published velocity: linear_x=%.2f, angular_z=%.2f", linear_x, angular_z)
    
    def forward(self, speed=None, duration=None):
        """
        Move the robot forward
        
        Args:
            speed (float, optional): Forward speed in m/s. If None, uses default speed.
            duration (float, optional): Duration to move in seconds. If None, moves continuously.
        """
        if speed is None:
            speed = self.default_linear_speed
        
        rospy.loginfo("Moving forward at %.2f m/s", speed)
        
        if duration is None:
            # Continuous movement
            self._publish_velocity(linear_x=speed)
        else:
            # Timed movement using ROS time for better accuracy
            start_time = rospy.get_rostime().secs
            while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self._publish_velocity(linear_x=speed)
                rospy.sleep(0.1)  # 10Hz update rate like test_movement
            self.stop()
    
    def backward(self, speed=None, duration=None):
        """
        Move the robot backward
        
        Args:
            speed (float, optional): Backward speed in m/s (positive value). If None, uses default speed.
            duration (float, optional): Duration to move in seconds. If None, moves continuously.
        """
        if speed is None:
            speed = self.default_linear_speed
        
        rospy.loginfo("Moving backward at %.2f m/s", speed)
        
        if duration is None:
            # Continuous movement
            self._publish_velocity(linear_x=-speed)
        else:
            # Timed movement using ROS time for better accuracy
            start_time = rospy.get_rostime().secs
            while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self._publish_velocity(linear_x=-speed)
                rospy.sleep(0.1)  # 10Hz update rate like test_movement
            self.stop()
    
    def turn_left(self, speed=None, duration=None):
        """
        Turn the robot left (counter-clockwise)
        
        Args:
            speed (float, optional): Rotation speed in rad/s (positive value). If None, uses default speed.
            duration (float, optional): Duration to turn in seconds. If None, turns continuously.
        """
        if speed is None:
            speed = self.default_angular_speed
        
        rospy.loginfo("Turning left at %.2f rad/s", speed)
        
        if duration is None:
            # Continuous movement
            self._publish_velocity(angular_z=speed)
        else:
            # Timed movement using ROS time for better accuracy
            start_time = rospy.get_rostime().secs
            while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self._publish_velocity(angular_z=speed)
                rospy.sleep(0.1)  # 10Hz update rate like test_movement
            self.stop()
    
    def turn_right(self, speed=None, duration=None):
        """
        Turn the robot right (clockwise)
        
        Args:
            speed (float, optional): Rotation speed in rad/s (positive value). If None, uses default speed.
            duration (float, optional): Duration to turn in seconds. If None, turns continuously.
        """
        if speed is None:
            speed = self.default_angular_speed
        
        rospy.loginfo("Turning right at %.2f rad/s", speed)
        
        if duration is None:
            # Continuous movement
            self._publish_velocity(angular_z=-speed)
        else:
            # Timed movement using ROS time for better accuracy
            start_time = rospy.get_rostime().secs
            while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
                self._publish_velocity(angular_z=-speed)
                rospy.sleep(0.1)  # 10Hz update rate like test_movement
            self.stop()
    
    def stop(self):
        """
        Stop the robot immediately
        """
        rospy.loginfo("Stopping robot")
        
        # Send stop command like test_movement.py
        self._publish_velocity(linear_x=0.0, angular_z=0.0)
        rospy.sleep(0.1)  # Brief pause to ensure command is processed
    
    def set_velocity(self, linear_x=0.0, angular_z=0.0):
        """
        Set custom velocity for the robot
        
        Args:
            linear_x (float): Linear velocity in x direction (m/s)
            angular_z (float): Angular velocity around z axis (rad/s)
        """
        rospy.loginfo("Setting velocity: linear_x=%.2f, angular_z=%.2f", linear_x, angular_z)
        self._publish_velocity(linear_x=linear_x, angular_z=angular_z)
    
    def set_angular_calibration(self, factor):
        """
        Adjust the angular calibration factor for rotate_angle function
        
        Args:
            factor (float): Calibration factor (2.0 means robot turns half as fast as commanded)
        """
        self.angular_calibration_factor = factor
        rospy.loginfo("Angular calibration factor set to %.2f", factor)
    
    def move_distance(self, distance, speed=None):
        """
        Move forward or backward for a specific distance (approximate)
        
        Args:
            distance (float): Distance to move in meters (positive=forward, negative=backward)
            speed (float, optional): Speed in m/s. If None, uses default speed.
        """
        if speed is None:
            speed = self.default_linear_speed
        
        if distance == 0:
            return
        
        # Calculate duration based on distance and speed
        duration = abs(distance) / speed
        direction = 1 if distance > 0 else -1
        
        rospy.loginfo("Moving %.2f meters at %.2f m/s (%.2f seconds)", distance, speed, duration)
        
        start_time = rospy.get_rostime().secs
        while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
            self._publish_velocity(linear_x=direction * speed)
            rospy.sleep(0.1)  # 10Hz update rate like test_movement
        
        self.stop()
    
    def rotate_angle(self, angle, speed=None):
        """
        Rotate by a specific angle (calibrated for actual robot performance)
        
        Args:
            angle (float): Angle to rotate in radians (positive=left/CCW, negative=right/CW)
            speed (float, optional): Rotation speed in rad/s. If None, uses default speed.
        """
        if speed is None:
            speed = self.default_angular_speed
        
        if angle == 0:
            return
        
        # Calculate duration based on angle and speed, with calibration factor
        # The calibration factor compensates for actual vs commanded angular velocity
        duration = (abs(angle) / speed) * self.angular_calibration_factor
        direction = 1 if angle > 0 else -1
        
        rospy.loginfo("Rotating %.2f radians at %.2f rad/s (%.2f seconds, calibrated)", angle, speed, duration)
        
        start_time = rospy.get_rostime().secs
        while (start_time + duration >= rospy.get_rostime().secs) and not rospy.is_shutdown():
            self._publish_velocity(angular_z=direction * speed)
            rospy.sleep(0.1)  # 10Hz update rate like test_movement
        
        self.stop()


# Robot controller management
_robot_controllers = {}  # Dictionary to store controllers for different robots

def get_robot_controller(namespace):
    """
    Get or create a controller for a specific robot namespace
    
    Args:
        namespace (str): Robot namespace (e.g., 'robot1', 'robot2')
    
    Returns:
        TurtleBotController: Controller instance for the specified robot
    """
    global _robot_controllers
    if namespace not in _robot_controllers:
        _robot_controllers[namespace] = TurtleBotController(namespace=namespace)
    return _robot_controllers[namespace]


# Async multi-robot control functions


# Advanced synchronization functions
def execute_synchronized(*robot_commands, **options):
    """
    Execute multiple robot commands simultaneously using threading
    
    Args:
        *robot_commands: Tuples of (robot_namespace, command_func, *args)
        **options: Execution options:
            - wait_for_completion (bool): If True, wait for all to finish. Default: True
            - timeout (float): Max time to wait in seconds. Default: None (no timeout)
            - return_threads (bool): If True, return thread objects. Default: False
    
    Returns:
        list: Thread objects if return_threads=True, otherwise None
    
    Examples:
        # Wait for all to complete (default behavior)
        execute_synchronized(
            ('robot1', 'forward', 0.2, 2.0),
            ('robot2', 'turn_left', None, 1.5)
        )
        
        # Fire-and-forget (don't wait)
        execute_synchronized(
            ('robot1', 'forward', 0.2, 2.0),
            ('robot2', 'turn_left', None, 1.5),
            wait_for_completion=False
        )
        
        # Wait with timeout
        execute_synchronized(
            ('robot1', 'forward', 0.2, 5.0),
            ('robot2', 'turn_left', None, 3.0),
            timeout=2.0
        )
        
        # Get thread objects for manual control
        threads = execute_synchronized(
            ('robot1', 'forward', 0.2, 2.0),
            ('robot2', 'turn_left', None, 1.5),
            return_threads=True
        )
    """
    # Parse options
    wait_for_completion = options.get('wait_for_completion', True)
    timeout = options.get('timeout', None)
    return_threads = options.get('return_threads', False)
    
    threads = []
    
    for command in robot_commands:
        namespace = command[0]
        command_func = command[1]
        args = command[2:] if len(command) > 2 else ()
        
        robot = get_robot_controller(namespace)
        
        def execute_command(robot_ctrl, func_name, cmd_args):
            getattr(robot_ctrl, func_name)(*cmd_args)
        
        thread = threading.Thread(target=execute_command, args=(robot, command_func, args))
        threads.append(thread)
        thread.start()
    
    # Handle different waiting strategies
    if return_threads:
        return threads
    elif wait_for_completion:
        if timeout is None:
            # Wait for all commands to complete
            for thread in threads:
                thread.join()
        else:
            # Wait with timeout
            start_time = rospy.Time.now()
            for thread in threads:
                remaining_time = timeout - (rospy.Time.now() - start_time).to_sec()
                if remaining_time > 0:
                    thread.join(remaining_time)
                else:
                    rospy.logwarn("Timeout reached, some robots may still be moving")
                    break
    # If wait_for_completion=False, return immediately (fire-and-forget)

# Note: Use execute_async() or execute_synchronized() for flexible multi-robot control

def execute_async(*robot_commands):
    """
    Execute multiple robot commands asynchronously (fire-and-forget)
    
    Args:
        *robot_commands: Tuples of (robot_namespace, command_func, *args)
    
    Returns:
        list: Thread objects for manual control if needed
    
    Example:
        # Start robots moving but don't wait
        threads = execute_async(
            ('robot1', 'forward', 0.2, 5.0),
            ('robot2', 'turn_left', None, 3.0)
        )
        
        # Do other work here...
        
        # Check if still running (optional)
        if threads[0].is_alive():
            print("Robot1 still moving")
    """
    return execute_synchronized(*robot_commands, wait_for_completion=False, return_threads=True)

def execute_with_timeout(*robot_commands, **kwargs):
    """
    Execute robot commands with a timeout
    
    Args:
        *robot_commands: Tuples of (robot_namespace, command_func, *args)
        timeout (float): Maximum time to wait in seconds
    
    Example:
        # Wait maximum 2 seconds, then proceed regardless
        execute_with_timeout(
            ('robot1', 'forward', 0.2, 5.0),  # Would normally take 5 seconds
            ('robot2', 'turn_left', None, 3.0),  # Would normally take 3 seconds
            timeout=2.0  # But we only wait 2 seconds
        )
    """
    timeout = kwargs.get('timeout', 5.0)  # Default 5 second timeout
    return execute_synchronized(*robot_commands, timeout=timeout)

def check_robots_status(threads):
    """
    Check the status of robot command threads
    
    Args:
        threads (list): List of thread objects from execute_async()
    
    Returns:
        dict: Status information for each thread
    """
    status = {}
    for i, thread in enumerate(threads):
        status['robot_{}'.format(i+1)] = {
            'is_alive': thread.is_alive(),
            'name': thread.name if hasattr(thread, 'name') else 'Thread-{}'.format(i+1)
        }
    return status

def wait_for_any_robot(threads, timeout=None):
    """
    Wait for ANY robot to finish (not all)
    
    Args:
        threads (list): List of thread objects
        timeout (float): Max time to wait
    
    Returns:
        int: Index of first thread that finished, or -1 if timeout
    """
    start_time = rospy.Time.now()
    
    while True:
        for i, thread in enumerate(threads):
            if not thread.is_alive():
                return i
        
        if timeout and (rospy.Time.now() - start_time).to_sec() > timeout:
            return -1
            
        rospy.sleep(0.1)  # Check every 100ms


# Example usage and testing functions
def demo_basic_movements():
    """
    Demonstration of basic movement functions
    """
    print("=== TurtleBot Basic Movement Demo ===")
    
    controller = TurtleBotController()
    
    try:
        print("Turning left...")
        controller.turn_left(duration=3)
        rospy.sleep(1)
        
                
        print("Move Forward for 2 seconds...")
        controller.forward(speed=0.3,duration=1.9)
        rospy.sleep(1)
        
        print("Turn Right...")
        controller.turn_right(duration=3)
        rospy.sleep(1)
        
        print("Move Forward for 7 seconds...")
        controller.forward(speed=0.3,duration=7)
        rospy.sleep(1)
        
        print("Stopping robot...")
        controller.stop()
        
        print("Demo completed!")
        
    except rospy.ROSInterruptException:
        print("Demo interrupted")
        controller.stop()

def demo_distance_movements():
    """
    Demonstration of distance-based movements
    """
    print("=== TurtleBot Distance Movement Demo ===")
    
    controller = TurtleBotController()
    
    try:
        print("Rotating 90 degrees left...")
        controller.rotate_angle(1.57)  # 90 degrees
        rospy.sleep(1)
        
        print("Moving forward 1 meters...")
        controller.move_distance(1)
        rospy.sleep(1)
        
        
        print("Moving forward 1.5 meters...")
        controller.move_distance(1.5)
        rospy.sleep(1)
        

        
        print("Demo completed!")
        
    except rospy.ROSInterruptException:
        print("Demo interrupted")
        controller.stop()

def demo_test_movement_style():
    """
    Demonstration using test_movement.py style movements
    """
    print("=== TurtleBot Test Movement Style Demo ===")
    
    controller = TurtleBotController()
    
    try:
        print("Step 1: Moving forward for 3 seconds")
        controller.forward(speed=0.3, duration=3.0)
        controller.stop()
        rospy.sleep(0.5)
        
        print("Step 2: Rotating for 2 seconds")
        controller.turn_left(speed=0.5, duration=2.0)
        controller.stop()
        rospy.sleep(1.0)
        
        print("Step 3: Moving forward for 2 seconds")
        controller.forward(speed=0.3, duration=2.0)
        controller.stop()
        
        print("Demo completed!")
        
    except rospy.ROSInterruptException:
        print("Demo interrupted")
        controller.stop()

def demo_multi_robot():
    """
    Demonstration of async thread control
    """
    print("=== Go to Large Box ===")
    
    try:
        print("Starting robots asynchronously (non-blocking)...")
        
        # Start robots but don't wait
        threads = execute_async(
            ('robot1', 'turn_left', None, 3.0),  
            ('robot2', 'turn_right', None, 3.0) 
        )
        
        print("Commands sent! Function returned immediately.")
        print("Robots are moving in background...")
        
        # Do other work while robots are moving
        for i in range(4):
            rospy.sleep(0.8)
            
            # Check status
            status = check_robots_status(threads)
            active_robots = [robot for robot, info in status.items() if info['is_alive']]
            
            if active_robots:
                print("Still moving: {}".format(', '.join(active_robots)))
            else:
                print("All robots finished!")
                break
        
        # Wait for any remaining
        for thread in threads:
            thread.join()
        
        print("\n Part 2: Wait for first robot to finish...")
        threads = execute_async(
            ('robot1', 'forward', 0.25, 1.0), 
            ('robot2', 'forward', 0.25, 1.0)   
        )
    
        first_done = wait_for_any_robot(threads)
        print("Robot {} finished first!".format(first_done + 1))
        
        # Wait for remaining
        for thread in threads:
            thread.join()
            
        print("\n Part 3: Wait for all robots to finish...")
        threads = execute_async(
            ('robot1', 'turn_right', None, 2.8),  # 4 seconds
            ('robot2', 'turn_left', None, 2.7)   # 1.5 seconds (finishes first)
        )
        # rospy.spin()
                # Wait for remaining
        for thread in threads:
            thread.join()
            
        
        threads = execute_async(
            ('robot1', 'forward', 0.3, 7),  # 4 seconds
            ('robot2', 'forward', 0.3, 7)   # 1.5 seconds (finishes first)
        )
        
        
        for thread in threads:
            thread.join()
        print("Async control demo completed!")
        
    except rospy.ROSInterruptException:
        print("Async demo interrupted")
        execute_async(('robot1', 'stop'), ('robot2', 'stop'))

def demo_async_control():
    """
    Demonstration of async thread control
    """
    print("=== Async Thread Control Demo ===")
    
    try:
        print("Starting robots asynchronously (non-blocking)...")
        
        # Start robots but don't wait
        threads = execute_async(
            ('robot1', 'turn_right', None, 3.0),  
            ('robot2', 'turn_left', None, 3.0) 
        )
        
        print("Commands sent! Function returned immediately.")
        print("Robots are moving in background...")
        
        # Do other work while robots are moving
        for i in range(4):
            rospy.sleep(0.8)
            
            # Check status
            status = check_robots_status(threads)
            active_robots = [robot for robot, info in status.items() if info['is_alive']]
            
            if active_robots:
                print("Still moving: {}".format(', '.join(active_robots)))
            else:
                print("All robots finished!")
                break
        
        # Wait for any remaining
        for thread in threads:
            thread.join()
        
        print("\n Part 2: Wait for first robot to finish...")
        threads = execute_async(
            ('robot1', 'forward', 0.25, 2), 
            ('robot2', 'forward', 0.25, 2)   
        )
    
        first_done = wait_for_any_robot(threads)
        print("Robot {} finished first!".format(first_done + 1))
        
        # Wait for remaining
        for thread in threads:
            thread.join()
            
        print("\n Part 3: Wait for all robots to finish...")
        threads = execute_async(
            ('robot1', 'turn_left', None, 2.8),  # 4 seconds
            ('robot2', 'turn_right', None, 2.3)   # 1.5 seconds (finishes first)
        )
    
                # Wait for remaining
        for thread in threads:
            thread.join()
            
        
        threads = execute_async(
            ('robot1', 'forward', 0.3, 7),  # 4 seconds
            ('robot2', 'forward', 0.3, 7)   # 1.5 seconds (finishes first)
        )
        
        
        for thread in threads:
            thread.join()
        print("Async control demo completed!")
        
    except rospy.ROSInterruptException:
        print("Async demo interrupted")
        execute_async(('robot1', 'stop'), ('robot2', 'stop'))


if __name__ == '__main__':
    try:
        import sys
        
        if len(sys.argv) > 1:
            if sys.argv[1] == 'demo':
                demo_basic_movements()
            elif sys.argv[1] == 'distance_demo':
                demo_distance_movements()
            elif sys.argv[1] == 'smooth_demo':
                demo_test_movement_style()
            elif sys.argv[1] == 'multi_robot_demo':
                demo_multi_robot()
            elif sys.argv[1] == 'async_demo':
                demo_async_control()
            else:
                print("Usage: python BP_controller.py [demo|distance_demo|smooth_demo|multi_robot_demo|async_demo]")
        else:
            print("TurtleBot Controller module loaded.")
            print("Available demos:")
            print("  python BP_controller.py demo              - Basic movement demo")
            print("  python BP_controller.py smooth_demo       - Smooth movement demo (recommended)")
            print("  python BP_controller.py distance_demo     - Distance-based movement demo")
            print("  python BP_controller.py multi_robot_demo  - Multi-robot control demo")
            print("  python BP_controller.py async_demo        - Async thread control demo")
            print("")
            print("Async multi-robot usage examples:")
            print("  import BP_controller as bp")
            print("  robot1 = bp.get_robot_controller('robot1')")
            print("  robot2 = bp.get_robot_controller('robot2')")
            print("  # Async execution (fire-and-forget)")
            print("  threads = bp.execute_async(")
            print("      ('robot1', 'forward', 0.2, 2.0),")
            print("      ('robot2', 'turn_left', None, 1.5)")
            print("  )")
            print("  # Check status: bp.check_robots_status(threads)")
            print("  # Wait for first: bp.wait_for_any_robot(threads)")
            
    except rospy.ROSInterruptException:
        print("Program interrupted")
