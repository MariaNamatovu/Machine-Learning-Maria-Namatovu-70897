#prints python and every package version in the requirements.txt file
import sys
import pkg_resources

print("Python version:", sys.version)
with open("requirements.txt") as f:
    for line in f:
        package = line.strip().split("==")[0]
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"package: {package}, version: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"package: {package} is not installed")