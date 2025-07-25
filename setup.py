from setuptools import setup, find_packages

setup(
    name='HNGeoNatomyX',
    version='0.1.0',
    description='A package to extract geometrical metrics to describe HNC anatomy',
    author='Odette Rios-Ibacache',
    author_email='odette.riosibacache@mail.mcgill.ca',
    packages=find_packages(),
    install_requires=[
        'pydicom', 
        'scipy',
        'numpy',
        'pyvista', 
        'scikit-learn',
        'shapely',
        'opencv-python',
    ]
   
)
