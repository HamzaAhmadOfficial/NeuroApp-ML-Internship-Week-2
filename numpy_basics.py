import numpy as np

# 1. Array Creation
a = np.array([1,2,3])
b = np.zeros((2,3))
c = np.ones((3,3))
d = np.arange(1,10)


# 2, Reshaping Arrays
reshaped = d.reshape(3,3)

# 3. Slicing
slice_arr = reshaped[0:2, 1:3]

# 4. Mathematical Operations
add = a + 5
multiply = a * 2
dot_product = np.dot(a, a)

# 5. Statistical Functions
mean = reshaped.mean()
median = np.median(reshaped)
std = reshaped.std()
var = reshaped.var()

# 6. Broadcasting
broadcasted = reshaped + np.array([1, 2, 3])

print("Array a:", a)
print("Zeros Array b:\n", b)
print("Ones Array c:\n", c)
print("Reshaped Array d:\n", reshaped)
print("Sliced Array:\n", slice_arr)
print("Add:", add)
print("Multiply:", multiply)
print("Dot Product:", dot_product)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std)
print("Variance:", var)
print("Broadcasted Array:\n", broadcasted)