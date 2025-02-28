import requests

# Call the API
url = "http://127.0.0.1:8000/query/"
question = "How many planets are there?"
response = requests.get(url + question.replace(" ", "%20"))  # URL-encode spaces
print(response.json())