import numpy as np

b = 30
for i in range(99, 0, -1):
    if i < b:
        b = i
        with open("ttt.txt", "w+") as f:
            f.write(str(b))