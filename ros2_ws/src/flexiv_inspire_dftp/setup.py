from setuptools import setup
package_name = "flexiv_inspire_dftp"
setup(name=package_name, version="0.3.0", packages=["flexiv_inspire_isaac.dftp"], package_dir={"flexiv_inspire_isaac.dftp": "."},
 data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ['dual_hands.yaml']),
        ("share/" + package_name + "/launch", ['launch/dual_dftp.launch.py']),
 ], install_requires=["setuptools"], zip_safe=True,
 maintainer="hb", maintainer_email="hb@localhost", description="flexiv_inspire_dftp", license="Apache-2.0",
 entry_points={"console_scripts": ['dftp_node = flexiv_inspire_isaac.dftp.ros_node:main', 'dftp_read_only = flexiv_inspire_isaac.dftp.read_only_check:main']})
