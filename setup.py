import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ProGED", # Replace with your own username
    version="0.8.4",
    author="Jure Brence, Boštjan Gec, Nina Omejc, Sebastian Mežnar",
    author_email="jure.brence@ijs.si",
    description="Probabilistic generative equation discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brencej/ProGED",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ["numpy", 
                        "pandas", 
                        "scipy", 
                        "sympy", 
                        "nltk",
                        # "scikit-tda",
                        # "scikit-learn",
                        # "hyperopt",
                        # "pytest",
                       ]
)
