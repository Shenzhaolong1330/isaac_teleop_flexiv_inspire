from setuptools import find_packages, setup
package_name = "flexiv_inspire_policy_bridge"
setup(name=package_name, version="0.3.0", packages=find_packages(),
 data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]), ("share/" + package_name, ["package.xml"])],
 install_requires=["setuptools"], zip_safe=True, maintainer="hb", maintainer_email="hb@localhost",
 description="ROS to TLS gRPC policy adapter", license="Apache-2.0", entry_points={"console_scripts": ["policy_bridge = flexiv_inspire_policy_bridge.server:main"]})
