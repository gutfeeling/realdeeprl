#!/usr/bin/env python

from setuptools import setup

setup(name="inventory_management_rl",
      version="0.1.0",
      description="Solves a simple inventory management problem using Deep Reinforcement Learning",
      author="Dibya Chakravorty",
      author_email="dibyachakravorty@gmail.com",
      packages=["inventory_env"],
      python_requires="~=3.9",
      install_requires=["gym==0.21.0",
                        "ray[rllib]==1.11",
                        "tensorflow; sys_platform != 'darwin' or platform_machine != 'arm64'",
                        "tensorflow-macos==2.10.0; sys_platform == 'darwin' and platform_machine == 'arm64'"
                        ],
      )
