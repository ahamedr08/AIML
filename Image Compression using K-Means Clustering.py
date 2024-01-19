import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin_min
import os

# Load an example image (you can replace this with your own image)
china = load_sample_image("china.jpg")

# Convert the image to a numpy array
china = china / 255.0  # Scale pixel values to the range [0, 1]
w, h, d = original_shape = tuple(china.shape)
image_array = np.reshape(china, (w * h, d))

# Number of colors to quantize the image to
n_colors = 2

# Fit K-means model to the image data
model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
labels = model.predict(image_array)
centers = model.cluster_centers_

# Create a compressed image by mapping each pixel to its nearest cluster center
compressed_image = centers[labels].reshape(china.shape)

# Display the original and compressed images
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(china)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Compressed image ({} colors)'.format(n_colors))
plt.imshow(compressed_image)
plt.show()

# Define the file paths for the original and compressed images
original_image_path = "original_image.jpg"
compressed_image_path = "compressed_image.jpg"

# Save the original and compressed images to file
plt.imsave(original_image_path, china)
plt.imsave(compressed_image_path, compressed_image)

# Get the file sizes
original_image_size = os.path.getsize(original_image_path)  # in bytes
compressed_image_size = os.path.getsize(compressed_image_path)  # in bytes

# Print the file sizes
print("Original Image Size: {:.2f} KB".format(original_image_size / 1024))
print("Compressed Image Size: {:.2f} KB".format(compressed_image_size / 1024))

# Remove the temporary image files
os.remove(original_image_path)
os.remove(compressed_image_path)
