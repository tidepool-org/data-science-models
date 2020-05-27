from distutils.core import setup

setup(
    name='Tidepool Data Science Models',
    version='0.1dev',
    author="Cameron Summers, Ed Nykaza",
    author_email="cameron@tidepool.org",
    packages=['tidepool_data_science_models', 'tidepool_data_science_models.models'],
    package_dir={'tidepool_data_science_models': 'tidepool_data_science_models'},
    license="BSD-2",
    long_description=open('README.md').read(),
    python_requires='>=3.6',
)