import requests
import numpy as np
from PIL import Image
import io

# Create a simple test image (valid knee X-ray-like grayscale image)
img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
img = Image.fromarray(img_array, mode='L')

# Convert to bytes
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test the endpoint
try:
    response = requests.post(
        'http://localhost:8000/api/predict',
        files={'file': ('test_image.png', img_bytes, 'image/png')},
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body:\n{response.text}")
    if response.status_code == 200:
        print("\n✓ SUCCESS - Response JSON:")
        print(response.json())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
