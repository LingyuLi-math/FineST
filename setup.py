"""
FineST: Fine-grained Spatial Transcriptomic
See: https://github.com/LingyuLi-math/FineST
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from pathlib import Path

here = path.abspath(path.dirname(__file__))

# Set __version__ for the project.
exec(open("./FineST/version.py").read())

# Get the long description from the relevant file
long_description=Path("README.rst").read_text("utf-8")
long_description_content_type='text/x-rst'

setup(
    name='FineST',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='FineST: Fine-grained Spatial Transcriptomic',
    long_description=long_description,
    long_description_content_type='text/x-rst',

    # The project's main homepage.
    url='https://github.com/LingyuLi-math/FineST',

    # Author details
    author=['Lingyu Li', 'Yuanhua Huang'],
    author_email='lingyuli@hku.hk',

    # Choose your license
    license='Apache-2.0',

    # What does your project relate to?
    keywords=[
        'Spatial transcriptomics', 
        'HE stained images',
        'Cell-cell communication', 
        'Ligand-recptor interaction'
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # entry_points={
    #       'console_scripts': [
    #           ],
    #       }, 

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    
    # install_requires=reqs,
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],

    include_package_data=True,
    package_data={'': ['datasets/*.csv', 'datasets/*.csv.gz']},

    extras_require={
        'docs': [
            #'sphinx == 1.8.3',
            'sphinx_bootstrap_theme']},

    py_modules = ['FineST']

    # buid the distribution: python setup.py sdist
    # upload to pypi: twine upload dist/...
)