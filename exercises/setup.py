#!/usr/bin/env python

from setuptools import setup

setup(name="exercises_real_world_deep_rl",
      version="0.1.0",
      description="Coding Exercises for the Real World Deep RL course",
      author="Dibya Chakravorty",
      author_email="dibyachakravorty@gmail.com",
      packages=["inventory_env_hard"],
      python_requires="~=3.9",
      install_requires=["gym==0.21.0",
                        "ray[rllib]==1.11",
                        "tensorflow; sys_platform != 'darwin' or platform_machine != 'arm64'",
                        "tensorflow-macos==2.10.0; sys_platform == 'darwin' and platform_machine == 'arm64'"
                        ],
      )