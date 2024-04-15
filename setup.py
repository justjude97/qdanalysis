from setuptools import setup, find_packages

descr = """package for MultiScript Handwriting Identification"""

if __name__ == "__main__":
    setup(
    name = "qdanalysis",
    version='0.1.0',
    packages=find_packages(include=['qdanalysis', 'qdanalysis.*'])
    )