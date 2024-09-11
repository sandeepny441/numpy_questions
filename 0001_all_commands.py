import numpy as np

# 1. Array Creation
a = np.array([1, 2, 3])
b = np.zeros((3, 3))
c = np.ones((2, 2))
d = np.empty((2, 3))
e = np.arange(10)
f = np.linspace(0, 1, 5)

# 2. Indexing
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[0, 1])  # Access element
print(arr[:, 1])  # Access column

# 3. Slicing
print(arr[0:2, 1:3])  # Slice rows 0-1 and columns 1-2

# 4. Math Operations
g = np.add(a, b)
h = np.subtract(a, b)
i = np.multiply(a, b)
j = np.divide(a, b)
k = np.power(a, 2)

# 5. Array Broadcasting
broadcast_arr = np.array([1, 2, 3]) + np.array([[1], [2], [3]])

# 6. Shape Manipulation
reshaped = arr.reshape((1, 9))
transposed = arr.T

# 7. Stacking
vertical_stack = np.vstack((a, b))
horizontal_stack = np.hstack((a, b))

# 8. Splitting
split_arr = np.split(arr, 3)

# 9. Universal Functions (ufuncs)
sin_arr = np.sin(arr)
exp_arr = np.exp(arr)

# 10. Vectorization
vectorized_func = np.vectorize(lambda x: x**2)
vectorized_result = vectorized_func(arr)

# 11. Filtering and Masking
mask = arr > 5
filtered_arr = arr[mask]

# 12. Random Module
random_arr = np.random.rand(3, 3)
random_int = np.random.randint(0, 10, size=(3, 3))

# 13. Statistics
mean = np.mean(arr)
median = np.median(arr)
std_dev = np.std(arr)

# 14. Linear Algebra
eigenvalues, eigenvectors = np.linalg.eig(arr)
inverse = np.linalg.inv(arr)

# 15. Fourier Transforms
fft_result = np.fft.fft(arr)

# 16. Loading Data
# Assuming 'data.txt' exists
loaded_data = np.loadtxt('data.txt')

# 17. Advanced Indexing
fancy_indexing = arr[[0, 2], [1, 2]]

# 18. Custom Vectorization
def custom_func(x, y):
    return x * y + 2

vectorized_custom = np.frompyfunc(custom_func, 2, 1)

# 19. Strides
strided_view = np.lib.stride_tricks.as_strided(arr, shape=(2,2), strides=(arr.itemsize*3, arr.itemsize))

# 20. GPU Operations (requires additional setup)
# Example: arr_gpu = cp.asarray(arr)  # Using CuPy for GPU operations

# 21. Multi-threading
# NumPy uses multi-threading for many operations by default
# You can control it with:
np.set_num_threads(4)  # Set number of threads