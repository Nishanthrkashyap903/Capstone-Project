import requests

# Replace with your own API Key and API Secret
api_key = "hQW4wps9Jvc28jKo9dqfyBFdw_ZgG0n6"
api_secret = "YLkoWFP173lRDiJJ8zGBqpMlypW1WUAw"
image_path = "Angry.jpg"  # Local image path or a URL

# API endpoint for emotion detection
faceplusplus_url = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Prepare the data for the request
data = {
    'api_key': api_key,
    'api_secret': api_secret,
    'return_attributes': 'emotion'
}

# Open the image file and send it in the request
with open(image_path, 'rb') as image_file:
    files = {
        'image_file': image_file
    }
    
    # Make the request
    response = requests.post(faceplusplus_url, data=data, files=files)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Emotion Detection Result:", result['faces'][0]['attributes']['emotion'])
else:
    print(f"Error: {response.status_code}, {response.text}")
