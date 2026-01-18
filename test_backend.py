#!/usr/bin/env python3
"""
Test script for Grounding DINO API
Run this to verify your backend is working correctly
"""

import requests
import base64
import json
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000/grounding-dino"
TEST_PROMPT = "car . person"

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (640, 480), color='white')
    
    # Save to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint...")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Make sure the backend server is running:")
        print("   python grounding_dino_hf_server.py")
        return False

def test_detection():
    """Test detection endpoint"""
    print("\n" + "=" * 60)
    print("Testing Detection Endpoint...")
    print("=" * 60)
    
    # Create test image
    print("📸 Creating test image...")
    test_image = create_test_image()
    
    # Prepare request
    payload = {
        "image": test_image,
        "prompt": TEST_PROMPT,
        "box_threshold": 0.35,
        "text_threshold": 0.25
    }
    
    print(f"📝 Prompt: {TEST_PROMPT}")
    print(f"🌐 Sending POST request to: {API_URL}")
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received:")
            print(f"   Model mode: {result.get('model_mode', 'unknown')}")
            print(f"   Detections: {len(result.get('detections', []))}")
            
            if result.get('detections'):
                print("\n📦 Detected objects:")
                for i, det in enumerate(result['detections'][:5]):  # Show first 5
                    print(f"   {i+1}. {det.get('label')} - confidence: {det.get('confidence', 0):.2f}")
            else:
                print("⚠️  No detections (this is normal for demo mode with blank image)")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("\n⚠️  Backend server not responding.")
        print("   Make sure you're running:")
        print("   python grounding_dino_hf_server.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request Timeout!")
        print("   The server is taking too long to respond.")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("\n" + "=" * 60)
    print("Testing CORS Configuration...")
    print("=" * 60)
    
    try:
        response = requests.options(
            API_URL,
            headers={"Origin": "http://localhost:5173"}
        )
        
        cors_header = response.headers.get('Access-Control-Allow-Origin')
        
        if cors_header:
            print(f"✅ CORS enabled: {cors_header}")
            return True
        else:
            print("⚠️  CORS headers not found")
            print("   Make sure flask-cors is installed:")
            print("   pip install flask-cors")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n🧪 Grounding DINO Backend Test Suite")
    print("=" * 60)
    
    # Run tests
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Fix this first!")
        return
    
    cors_ok = test_cors()
    detection_ok = test_detection()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Health Check:  {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"CORS Check:    {'✅ PASS' if cors_ok else '⚠️  WARNING'}")
    print(f"Detection:     {'✅ PASS' if detection_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if health_ok and detection_ok:
        print("\n🎉 All tests passed! Your backend is working correctly.")
        print("\nNow try using it in your React app:")
        print("1. Make sure your React app is running (npm run dev)")
        print("2. Click Auto Label → Grounding DINO")
        print("3. Enter prompt and click 'Label Current'")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
