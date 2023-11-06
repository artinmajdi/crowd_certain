import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized setuptools install command to use conda if available."""

    def run(self):

        conda_available = False
        mamba_available = False

        try:

            # Check if conda is available
            subprocess.check_call(['conda', '--version'])
            conda_available = True

            # Check if mamba is available
            subprocess.check_call(['mamba', '--version'])
            mamba_available = True

        except subprocess.CalledProcessError:
            pass

        if conda_available:

            if not mamba_available:
                print('Mamba is not available. Installing mamba using conda.')
                subprocess.check_call(['conda', 'install', '-c', 'conda-forge', 'mamba', '-y'])

            print('Installing using mamba')
            subprocess.check_call(['mamba', 'env', 'update', '--file', 'requirements.yml'])

        else:
            print('Conda is not available. Installing using pip')
            # If conda is not available, use pip to install requirements
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

        # Proceed with the standard installation
        install.run(self)

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="crowd-certain",
    version="1.0.0",
    author="Artin Majdi",
    author_email="msm2024@gmail.com",
    description="Crowd Sourced Label Augmentation",
    url="https://github.com/artinmajdi/taxonomy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires='>=3.10',
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_dir={'crowd_certain': 'crowd_certain'},
    package_data={'crowd_certain': ['crowd_certain/config.json']},
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': ['myutility=crowd_certain.main:main']},
    cmdclass={'install': CustomInstallCommand}
)
