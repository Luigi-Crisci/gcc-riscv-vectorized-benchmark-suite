import sys
import os
import subprocess
import math

sizes = []
for i in range(10,27):
    sizes.append(2**i)

os.makedirs("res", exist_ok=True)
os.remove("res/axpy_res.csv")

with open("res/axpy_res.csv", "w+") as fd:
    for size in sizes:
        print(f"Running exp with size 2^{math.log2(size)} = {size}...")
        result = subprocess.run(["./bin/axpy_nostore", f"{size}"], capture_output=True, text=True)
        fd.write(result.stdout)