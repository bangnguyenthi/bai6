import numpy as np
import cv2
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển đổi sang không gian màu xám
image = cv2.imread('3182197_1.jpg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('images3.jpeg', cv2.IMREAD_GRAYSCALE)  # Thay đổi với đường dẫn ảnh thứ 2

# Số lượng cụm
k = 3  # Thay đổi số lượng cụm theo yêu cầu

# Chuyển ảnh thành ma trận 1D
image_flat = image.flatten()
image1_flat = image1.flatten()

# Áp dụng FCM cho ảnh 'image'
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data=np.expand_dims(image_flat, axis=0),
    c=k,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None
)

# Lấy nhãn cụm cho từng pixel của ảnh 'image'
cluster_labels = np.argmax(u, axis=0)
segmented_image = cluster_labels.reshape(image.shape)

# Áp dụng FCM cho ảnh 'image1'
cntr1, u1, _, _, _, _, _ = fuzz.cluster.cmeans(
    data=np.expand_dims(image1_flat, axis=0),
    c=k,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None
)

# Lấy nhãn cụm cho từng pixel của ảnh 'image1'
cluster_labels1 = np.argmax(u1, axis=0)
segmented_image1 = cluster_labels1.reshape(image1.shape)

# Hiển thị ảnh gốc và ảnh phân cụm của cả 2 ảnh
plt.figure(figsize=(20, 10))

# Ảnh gốc và phân cụm của ảnh 'image'
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Ảnh Sau Khi Phân Cụm FCM (image)')
plt.axis('off')

# Ảnh gốc và phân cụm của ảnh 'image1'
plt.subplot(2, 2, 3)
plt.imshow(image1, cmap='gray')
plt.title('Ảnh Gốc image1')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(segmented_image1, cmap='gray')
plt.title('Ảnh Sau Khi Phân Cụm FCM (image1)')
plt.axis('off')

plt.show()
