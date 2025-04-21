from setuptools import setup, find_packages

setup(
    name='SROsignal',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    license='GPLv3',
    description='Extract ordering info. from the spatial/frequency domain of atomic signals.',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'opencv-python', 'matplotlib', 'pickle', 'scikit-learn', 'json'],
    url='https://github.com/wzetto/SROsignal',
    author='Zhi Wang',
    author_email='wang.zhi.48u@st.kyoto-u.ac.jp',
)