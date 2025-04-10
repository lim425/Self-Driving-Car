#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# venv lib path
from sdc_venv_pkg.Drive_Bot import Car

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # Velocity publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.send_cmd_vel)
        self.velocity = Twist()

        # Raw image subscriber
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.process_data, 10)
        self.get_logger().info("Subscribing Image Feed")
        self.bridge = CvBridge()
        self.subscriber  # prevent unused variable warning

        # Image publisher for processed images (for output video recording purpose)
        self.image_publisher = self.create_publisher(Image, '/camera/output', 10)

        self.Car = Car()

    def send_cmd_vel(self):
        self.publisher_.publish(self.velocity)

    def process_data(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        Angle, Speed, img = self.Car.drive_car(frame)
        
        # Speed = 0.0
        self.velocity.angular.z = Angle
        self.velocity.linear.x = Speed

        # Publish the processed image (for output video recording purpose)
        self.publish_processed_image(img)

        cv2.imshow("Frame", img)

        # display_img = img.copy()
        # display_img_resized = cv2.resize(display_img, (1280, 720))
        # cv2.imshow("original_frame", frame)
        # cv2.imshow("display_img", display_img_resized)
        cv2.waitKey(1)

    def publish_processed_image(self, img):
        # Converts the OpenCV image to a ROS 2 Image message and publishes it.
        ros_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.image_publisher.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = VisionNode()

    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()