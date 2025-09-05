#!/usr/bin/env python

"""
Test script for async thread control functionality
"""

import rospy
import time
import BP_controller as bp

def test_async_control():
    """Test async execution with thread control"""
    
    print("=== Testing Async Thread Control ===")
    
    try:
        # Test 1: Basic async execution
        print("\nTest 1: Starting robots asynchronously...")
        threads = bp.execute_async(
            ('robot1', 'forward', 0.2, 4.0),   # 4 second movement
            ('robot2', 'turn_left', None, 2.0) # 2 second turn
        )
        
        print("Started {} robots".format(len(threads)))
        print("Function returned immediately - robots running in background")
        
        # Monitor status while they're running
        for i in range(5):  # Check status 5 times
            rospy.sleep(1.0)
            status = bp.check_robots_status(threads)
            print("Status check {}: {}".format(i+1, status))
            
            # Check individual robots
            if not threads[0].is_alive():
                print("  -> Robot1 finished!")
            if not threads[1].is_alive():
                print("  -> Robot2 finished!")
        
        # Wait for any remaining
        print("\nWaiting for remaining robots to finish...")
        for thread in threads:
            thread.join()
        print("All robots finished!")
        
        # Test 2: Wait for first robot to finish
        print("\nTest 2: Wait for first robot to finish...")
        threads = bp.execute_async(
            ('robot1', 'turn_right', None, 3.0),  # 3 seconds
            ('robot2', 'forward', 0.1, 1.0)      # 1 second (should finish first)
        )
        
        first_done = bp.wait_for_any_robot(threads, timeout=5.0)
        if first_done >= 0:
            print("Robot {} finished first!".format(first_done + 1))
        else:
            print("Timeout reached")
        
        # Stop any remaining
        print("Stopping remaining robots...")
        for i, thread in enumerate(threads):
            if thread.is_alive():
                print("Robot {} still running, will finish naturally".format(i+1))
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
            
        print("\nAsync control test completed!")
        
    except rospy.ROSInterruptException:
        print("Test interrupted")
    except Exception as e:
        print("Error during test: {}".format(e))

def test_overlapping_commands():
    """Test overlapping async commands"""
    
    print("\n=== Testing Overlapping Commands ===")
    
    try:
        # Start long movements
        print("Starting long movements...")
        long_threads = bp.execute_async(
            ('robot1', 'forward', 0.15, 6.0),  # 6 second forward
            ('robot2', 'forward', 0.15, 6.0)   # 6 second forward
        )
        
        # After 2 seconds, give correction commands
        print("Waiting 2 seconds...")
        rospy.sleep(2.0)
        
        print("Giving correction commands...")
        correction_threads = bp.execute_async(
            ('robot1', 'turn_left', None, 0.5),   # Quick left turn
            ('robot2', 'turn_right', None, 0.5)   # Quick right turn
        )
        
        # Monitor both sets of commands
        print("Monitoring commands...")
        all_threads = long_threads + correction_threads
        
        while any(thread.is_alive() for thread in all_threads):
            status = bp.check_robots_status(all_threads)
            active_count = sum(1 for robot, info in status.items() if info['is_alive'])
            print("Active threads: {}".format(active_count))
            rospy.sleep(1.0)
        
        print("All overlapping commands completed!")
        
    except rospy.ROSInterruptException:
        print("Overlapping test interrupted")

def interactive_test():
    """Interactive test for manual control"""
    
    print("\n=== Interactive Async Test ===")
    print("Commands:")
    print("  1 - Start robot1 forward (5 sec)")
    print("  2 - Start robot2 turn left (3 sec)")
    print("  3 - Start both robots different commands")
    print("  s - Check status")
    print("  q - Quit")
    
    active_threads = []
    
    try:
        while not rospy.is_shutdown():
            try:
                cmd = raw_input("\nEnter command: ").strip().lower()
                
                if cmd == '1':
                    print("Starting robot1 forward...")
                    threads = bp.execute_async(('robot1', 'forward', 0.2, 5.0))
                    active_threads.extend(threads)
                    
                elif cmd == '2':
                    print("Starting robot2 turn left...")
                    threads = bp.execute_async(('robot2', 'turn_left', None, 3.0))
                    active_threads.extend(threads)
                    
                elif cmd == '3':
                    print("Starting both robots...")
                    threads = bp.execute_async(
                        ('robot1', 'turn_right', None, 2.0),
                        ('robot2', 'backward', 0.15, 3.0)
                    )
                    active_threads.extend(threads)
                    
                elif cmd == 's':
                    if active_threads:
                        # Clean up finished threads
                        active_threads = [t for t in active_threads if t.is_alive()]
                        if active_threads:
                            status = bp.check_robots_status(active_threads)
                            print("Status:", status)
                        else:
                            print("No active threads")
                    else:
                        print("No threads started yet")
                        
                elif cmd == 'q':
                    break
                    
                else:
                    print("Unknown command")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        # Wait for any remaining threads
        if active_threads:
            print("Waiting for remaining movements to finish...")
            for thread in active_threads:
                if thread.is_alive():
                    thread.join()
        
    except Exception as e:
        print("Interactive test error: {}".format(e))

if __name__ == '__main__':
    try:
        print("Starting async control tests...")
        print("Make sure robot1 and robot2 are available!")
        
        # Run tests
        test_async_control()
        rospy.sleep(2)
        
        test_overlapping_commands()
        rospy.sleep(2)
        
        interactive_test()
        
    except rospy.ROSInterruptException:
        print("Tests interrupted")
    except Exception as e:
        print("Test error: {}".format(e))
