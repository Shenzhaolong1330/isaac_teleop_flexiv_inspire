from setuptools import setup
package_name = "flexiv_inspire_cameras"
setup(name=package_name, version="0.3.0", packages=["flexiv_inspire_isaac.cameras"], package_dir={"flexiv_inspire_isaac.cameras": "."},
 data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ['realsense_rgb.yaml']),
        ("share/" + package_name + "/launch", ['launch/triple_rgb.launch.py']),
 ], install_requires=["setuptools"], zip_safe=True,
 maintainer="hb", maintainer_email="hb@localhost", description="flexiv_inspire_cameras", license="Apache-2.0",
 entry_points={"console_scripts": ['camera_node = flexiv_inspire_isaac.cameras.ros_node:main', 'camera_verify = flexiv_inspire_isaac.cameras.verify:main']})
