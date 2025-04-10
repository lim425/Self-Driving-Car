#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class VisionSave(Node):

    def __init__(self):
        super().__init__('vision_save_node')

        self.subscriber = self.create_subscription(Image, '/camera/output', self.process_recording_data, 10)
        self.out = cv2.VideoWriter('/home/junyi/output_video3.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
        self.get_logger().info("Subscribing Image Feed and video recording")
        self.bridge = CvBridge()

    def process_recording_data(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        frame_resized = cv2.resize(frame, (1280, 720))  # Ensure consistent frame size
        self.out.write(frame_resized)  # Record video
        cv2.imshow("Frame_recording", frame_resized)
        cv2.waitKey(1)

    def __del__(self):
        self.out.release()  # Ensure the video file is properly closed
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = VisionSave()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly and release resources
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
