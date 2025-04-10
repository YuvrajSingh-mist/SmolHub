from setuptools import setup, find_packages

setup(
    name="smolhub",
    version="0.5.2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "smolhub": ["config/*.yaml"],
    },
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "wandb",
        "tqdm",
        "pyyaml",
        "numpy",
    ],
    author="Yuvraj Singh",
    author_email="yuvraj.mist@gmail.com",
        long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    include_package_data=True,
)