import dlib
import cv2
import numpy as np

image_path = "training-images/Myself/myimg1.jpg"
  # path to your image

# Step 1: Read with OpenCV (BGR)
bgr = cv2.imread(image_path)
if bgr is None:
    print("[ERROR] Failed to load image!")
    exit()

# Step 2: Convert to RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Step 3: Ensure it's contiguous (required by dlib)
rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

# Debug info
print("[DEBUG] Shape:", rgb.shape, ", dtype:", rgb.dtype, ", contiguous:", rgb.flags['C_CONTIGUOUS'])

# Step 4: Detect faces
detector = dlib.get_frontal_face_detector()
faces = detector(rgb, 1)

# Step 5: Results
print(f"[INFO] Found {len(faces)} face(s).")
for i, face in enumerate(faces):
    print(f"Face {i+1}: {face}")
