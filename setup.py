from setuptools import find_packages, setup

setup(
    name="task_tracker",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        "torch",  # example dependency
    ],
)
