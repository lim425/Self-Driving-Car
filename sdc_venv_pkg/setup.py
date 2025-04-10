# from setuptools import find_packages, setup
# import os
# from glob import glob

# package_name = 'sdc_venv_pkg'
# config_module = 'sdc_venv_pkg/config'
# detection_module = 'sdc_venv_pkg/Detection'
# detection_lane_module = 'sdc_venv_pkg/Detection/Lanes'
# detection_sign_module = 'sdc_venv_pkg/Detection/Signs'

# setup(
#     name=package_name,
#     version='0.0.0',
#     # packages=find_packages(exclude=['test']),
#     packages=[package_name, config_module, detection_module, detection_lane_module, detection_sign_module],
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         (os.path.join('share', package_name,'launch'), glob('launch/*')),
#         (os.path.join('share', package_name,'worlds'), glob('worlds/*')),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='junyi',
#     maintainer_email='limjunyi425@gmail.com',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'vision_node = sdc_venv_pkg.vision_node:main'
#         ],
#     },
# )



# original venv setup.py


from setuptools import find_packages, setup

package_name = 'sdc_venv_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junyi',
    maintainer_email='limjunyi425@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = sdc_venv_pkg.vision_node:main',
            'video_save = sdc_venv_pkg.video_save:main'
        ],
    },
)
