import setuptools

requirements = [
    "numpy==1.23.5",
    "gym==0.21.0",
    "imitation==0.3.1",
    "protobuf==3.12.0",
    "pyglet==2.0.4",
    "pymap2d==0.1.15",
    "pytest==7.2.1",
    "fire==0.5.0",
    "cython",
    "scikit-image==0.19.3",
    "tensorboard==2.11.2",
    "hydra-core==1.2.0",
    "seaborn==0.11.2",
    "hydra-optuna-sweeper==1.2.0",
    "numba==0.56.4",
    "packaging==22.0",
    "mlflow==2.1.1",
    "autorom==0.4.2",
    "gym[atari]==0.21.0",
    "opencv-python==4.7.0.68",
]

setuptools.setup(
    name="navigation_stack_py",
    version="0.0.0",
    author="Kohei Honda",
    author_email="honda.kohei.b0@s.mail.nagoya-u.ac.jp",
    packages=setuptools.find_packages(),
    python_requires=">=3.8.10",
    platforms=["Linux"],
    install_requires=requirements,
    include_package_data=True,
)
