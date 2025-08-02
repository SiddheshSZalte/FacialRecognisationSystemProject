from PIL import Image
import numpy as np
import face_recognition_api
import matplotlib.pyplot as plt

img_path = "training-images/Myself/myimg1.jpg"

# Load the image
image = face_recognition_api.load_image_file(img_path)

# Convert to PIL image just to visualize and confirm type
pil_image = Image.fromarray(image)
print(f"Mode: {pil_image.mode}, Size: {pil_image.size}, Type: {image.dtype}")

# Show image to confirm it looks okay
plt.imshow(pil_image)
plt.title("Check This Image")
plt.axis("off")
plt.show()

# Try to detect face
locations = face_recognition_api.face_locations(image)
if len(locations) == 0:
    print("❌ No face detected.")
else:
    print(f"✅ Detected {len(locations)} face(s).")
