from setuptools import setup, find_packages

setup(
    name="Bohemien_Motor_Designer",
    version="2.0.0",
    description="Electric motor design toolkit — 1 kW to 1 MW, 0–1500 V DC bus",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "Bohemien_Motor_Designer-gui = Bohemien_Motor_Designer.gui.app:main",
        ],
    },
)
