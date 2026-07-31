from setuptools import setup
package_name = "flexiv_inspire_rerun"
setup(name=package_name, version="0.3.0", packages=["flexiv_inspire_isaac.rerun_viz"], package_dir={"flexiv_inspire_isaac.rerun_viz": "."},
 data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
 ], install_requires=["setuptools"], zip_safe=True,
 maintainer="hb", maintainer_email="hb@localhost", description="flexiv_inspire_rerun", license="Apache-2.0",
 entry_points={"console_scripts": ['rerun = flexiv_inspire_isaac.rerun_viz.cli:main']})
