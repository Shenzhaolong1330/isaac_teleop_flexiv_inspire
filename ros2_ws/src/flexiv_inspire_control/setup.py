from setuptools import find_packages, setup

package_name = "flexiv_inspire_control"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            [
                "config/control_bridge.yaml",
                "config/teleop.yaml",
                "config/manus_calibration_template.yaml",
                "config/manus_ergonomics_calibration_template.yaml",
                "config/manus_ergonomics_bootstrap.yaml",
            ],
        ),
        (
            f"share/{package_name}/launch", ["launch/teleop_control.launch.py"]
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hb",
    maintainer_email="hb@localhost",
    description="Independent fail-closed Flexiv/Inspire control bridge",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "control_bridge = flexiv_inspire_control.node:main",
            "authorize_control = flexiv_inspire_control.authorize_control:main",
            "authorize_home = flexiv_inspire_control.authorize_home:main",
            "home = flexiv_inspire_control.authorize_home:request_main",
            "zero_ft_local = flexiv_inspire_control.zero_ft_local:main",
            "teleop_input = flexiv_inspire_control.teleop_input_node:main",
            "manus_ergonomics_source = flexiv_inspire_control.manus_ergonomics_source:main",
            "manus_calibrate = flexiv_inspire_control.manus_calibration:main",
        ],
    },
)
