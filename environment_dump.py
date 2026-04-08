import sys
import pkg_resources
from pathlib import Path


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

with open(OUTPUT_DIR / "environment.txt", "w") as f:
    f.write(f"Python version: {sys.version}\n\n")
    f.write("Installed packages:\n")
    for pkg in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
        f.write(f"{pkg.project_name}=={pkg.version}\n")

print("Saved environment.txt")

reqs = Path("requirements.txt")

with open(reqs, "w") as f:
    for pkg in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
        f.write(f"{pkg.project_name}=={pkg.version}\n")

print("Saved requirements.txt")