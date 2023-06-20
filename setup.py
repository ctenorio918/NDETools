from setuptools import setup, find_packages

setup(
    name='NDETools',
    version='0.1.0',
    description='Basic tools for NDE&T applications',
    url='https://github.com/ctenorio918/NDETools',
    author='Charles "Nate" Tenorio',
    author_email='ctenorio@gatech.edu',
    packages=['NDETools'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'pysimplegui',
                      'pathlib',
                      'pyvisa',
                      'json',
                      'pandas',
                      'PySimpleGUI',
                      'pywt',
                      'tftb'
                      ]
)