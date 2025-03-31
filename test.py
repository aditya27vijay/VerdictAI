import requests

url = "https://api.deepinfra.com/v1/openai/models"
headers = {"Authorization": "Bearer AMENPRd2GjmGxy6itRSOGgbkimbuimwX"}

response = requests.get(url, headers=headers)
print(response.json())  # This should list available models
