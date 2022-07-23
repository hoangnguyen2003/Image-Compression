import sys
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from PIL import Image

original_image_name = sys.argv[1]
compressed_image_name = "Compressed_" + original_image_name

# Read image
image = io.imread(original_image_name)

# Preprocessing
height, width = image.shape[0], image.shape[1]
image = image.reshape(height*width, 3)

print("Compressing image...")

# Model
k = 13
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(image)
clusters = kmeans.cluster_centers_
clusters = clusters.astype(int)

# Create new image (compressed image)
new_image = np.zeros_like(image)
for i in range(len(new_image)):
    new_image[i] = clusters[labels[i]]
    
new_image = new_image.reshape(height, width, 3)

# Save numpy array as an image
img = Image.fromarray(new_image)
img.save(compressed_image_name)

print("Your image have been compressed!")