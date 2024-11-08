import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh và ảnh thứ 2
image = cv2.imread('images3.jpeg')
image1 = cv2.imread('3182197_1.jpg')

# Chuyển ảnh sang không gian RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Chuyển ảnh thứ 2 sang RGB

# Phân cụm ảnh 'image'
original_shape_image = image_rgb.shape
pixel_values_image = image_rgb.reshape((-1, 3))
pixel_values_image = np.float32(pixel_values_image)

# Áp dụng KMeans cho ảnh 'image'
kmeans_image = KMeans(n_clusters=3, random_state=42)
labels_image = kmeans_image.fit_predict(pixel_values_image)

segmented_image = kmeans_image.cluster_centers_[labels_image]
segmented_image = segmented_image.reshape(original_shape_image)

# Phân cụm ảnh 'image1'
original_shape_image1 = image1_rgb.shape
pixel_values_image1 = image1_rgb.reshape((-1, 3))
pixel_values_image1 = np.float32(pixel_values_image1)

# Áp dụng KMeans cho ảnh 'image1'
kmeans_image1 = KMeans(n_clusters=3, random_state=42)
labels_image1 = kmeans_image1.fit_predict(pixel_values_image1)

segmented_image1 = kmeans_image1.cluster_centers_[labels_image1]
segmented_image1 = segmented_image1.reshape(original_shape_image1)

# Hiển thị ảnh gốc và ảnh phân cụm của cả 2 ảnh
plt.figure(figsize=(20, 7))

# Ảnh gốc và phân cụm của ảnh 'image'
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Ảnh Gốc image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image.astype(np.uint8))
plt.title('Ảnh Sau Khi Phân Cụm K-Means (image)')
plt.axis('off')

# Ảnh gốc và phân cụm của ảnh 'image1'
plt.subplot(2, 2, 3)
plt.imshow(image1_rgb)
plt.title('Ảnh Gốc image1')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(segmented_image1.astype(np.uint8))
plt.title('Ảnh Sau Khi Phân Cụm K-Means (image1)')
plt.axis('off')

plt.show()
