import numpy as np

data = [[0,1,2,3,4]]

data = np.array(data)

new_data = data[:, :4]
m = np.argmax(data)
n = np.argmax(new_data)
print(m)
print(n)