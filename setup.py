from setuptools import setup, find_packages

setup(
    name="vehicle-damage-detection",
    version="0.1.0",
    description="Modular vehicle damage detection pipeline with swappable training strategies",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pycocotools>=2.0.7",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "tabulate>=0.9.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
        "serve": ["fastapi>=0.100.0", "uvicorn>=0.23.0"],
        "yolo": ["ultralytics>=8.0.0"],
        "wandb": ["wandb"],
    },
)
