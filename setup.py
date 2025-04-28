from setuptools import setup, find_packages

setup(
    name="saliencytools",
    version="0.1.0",
    author="Valerio Bonsignori",
    author_email="valerio.bonsignori@phd.unipi.it",
    description="A collection of metrics to compare saliency maps, validated using KNN-like classifiers on MNIST.",
    long_description= open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/valevalerio/saliencytools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.4",
        "pandas==1.5.3",
        "matplotlib==3.7.5",
        "seaborn==0.13.2",
        "scikit-learn==1.3.2",
        "scikit-multilearn==0.2.0",
        "scipy==1.10.1",
        "torch",
        "sklearn-image"
    ],
)
