
# ROS2 Self Driving Car with Deep Learning and Computer Vision in Gazebo Simulation
This repository contains my work from the **[ROS2 Self Driving Car with Deep Learning and Computer Vision](https://www.udemy.com/course/ros2-self-driving-car-with-deep-learning-and-computer-vision/)** Udemy course by Muhammad Luqman and Haider Najeeb. 

Key features include:
- **Lane Detection**: Utilizes computer vision techniques with OpenCV to detect and follow lane lines.
- **Sign Classification**: Implements a custom-built Convolutional Neural Network (CNN) to classify traffic signs.
- **Sign Tracking**: Employs optical flow for tracking traffic signs in real-time.
- **Cruise Control**: Implements basic speed regulation to maintain smooth and safe navigation.





## Acknowledgements

Special thanks to Muhammad Luqman and Haider Najeeb for the excellent "ROS2 Self Driving Car with Deep Learning and Computer Vision" course.

 - [Udemy course](https://www.udemy.com/course/ros2-self-driving-car-with-deep-learning-and-computer-vision/)
  - [Original Course Github repo](https://github.com/noshluk2/ROS2-Self-Driving-Car-AI-using-OpenCV/tree/main)




## Usage

Step1: Activate the virtual environment (in dev_ws), and start the Gazebo simulation

```bash
source ./venv/bin/activate
ros2 launch sdc_venv_pkg world_gazebo.launch.py
```

Step2: Run the computer vision node (in new terminal)

```bash
source ./venv/bin/activate
export PYTHONPATH=/home/junyi/potbot_venv_ws/venv/lib/python3.10/site-packages:$PYTHONPATH
ros2 run sdc_venv_pkg vision_node 
```




## Appendix
![Image Alt]().


