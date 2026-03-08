import requests
import json

# Test 1: Get available images
print("=" * 60)
print("TEST 1: Get available images from test split")
print("=" * 60)
try:
    response = requests.get('http://localhost:8000/api/available-images', 
                           params={'split': 'test', 'limit': 10})
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Split: {data['split']}")
    print(f"Images found: {data['count']}")
    print(f"First 3 images:")
    for img in data['images'][:3]:
        print(f"  - {img['filename']} (KL-{img['kl_grade']}, {img['file_size']} bytes)")
    print(f"\nResponse includes: {list(data.keys())}")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: Get dataset stats
print("\n" + "=" * 60)
print("TEST 2: Get dataset statistics")
print("=" * 60)
try:
    response = requests.get('http://localhost:8000/api/dataset-stats',
                           params={'split': 'test'})
    print(f"Status: {response.status_code}")
    stats = response.json()
    print(json.dumps(stats, indent=2))
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: Try to access training set (should be blocked)
print("\n" + "=" * 60)
print("TEST 3: Attempt to access TRAINING split (should fail)")
print("=" * 60)
try:
    response = requests.get('http://localhost:8000/api/available-images',
                           params={'split': 'train'})
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Images returned: {data['count']}")
    if data['count'] == 0:
        print("✓ GOOD: Training data access BLOCKED")
    else:
        print("❌ WARNING: Training data was returned!")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 4: Try validation split
print("\n" + "=" * 60)
print("TEST 4: Get available images from val split")
print("=" * 60)
try:
    response = requests.get('http://localhost:8000/api/available-images',
                           params={'split': 'val', 'limit': 5})
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Validation images found: {data['count']}")
    if data['count'] > 0:
        print(f"✓ Sample: {data['images'][0]['filename']}")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 60)
print("✓ All endpoint tests completed!")
print("=" * 60)
