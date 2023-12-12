import requests

url = 'https://evprice-1ba58eb40b90.herokuapp.com/predict'

data = {
        "efficiency": 180,
        "fast_charge": 450,
        "battery": 65,
        "range": 375,
        "top_speed": 180,
        "acceleration0to100": 5.5
            }

response = requests.post(url, json=data).json()
print(response)