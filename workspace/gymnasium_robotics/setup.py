from setuptools import setup, find_packages

subpackages = find_packages(where=".")

setup(
    name="gymnasium-robotics",
    version="1.2.0",
    description="Local editable Gymnasium Robotics",
    packages=["gymnasium_robotics"] + [
        f"gymnasium_robotics.{pkg}" for pkg in subpackages
    ],
    package_dir={
        "gymnasium_robotics": ".",
    },
    include_package_data=True,
    package_data={
        "gymnasium_robotics": [
            "envs/assets/**/*",
            "envs/**/*.xml",
            "envs/**/*.stl",
            "envs/**/*.png",
            "envs/**/*.json",
        ],
    },
    install_requires=[
        "gymnasium",
        "numpy",
        "mujoco",
        "imageio",
        "pettingzoo",
    ],
    python_requires=">=3.8",
)