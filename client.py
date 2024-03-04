import requests
import json
import time

url = "https://autumn8functions.default.aws.autumn8.ai"
deepseekcoder_6_7 = "/inference/4373e4e3-2502-4f28-b96a-125c2bc6faeb%2B293%2Bq5k6v6egxqi65gt1ojg9%2Bg5-2xlarge%2Bmar"
deepseekcoder_33_5bit = "/inference/aa1caf72-8e03-481e-96a9-4a14ff289798%2B293%2Bmk1qjyoeawzn0ifdjb3u%2Bg5-4xlarge%2Bmar"
deepseekcoder_33_8bit = "/inference/bea7c4e4-2b46-400a-b51e-7058a0ae639a%2B293%2Bshlb15o4kbkehlwlpgno%2Bg5-4xlarge%2Bmar"
mixtral_5bit = "/inference/fe80279d-6261-47f7-b527-2ad827c911c1%2B293%2Bw29oxt3arylsrrd9laq3%2Bg5-4xlarge%2Bmar"
mixtral_8bit = "/inference/e7c2a547-c385-47c9-96ad-e6993674c7d3%2B293%2B922ohce8dyg762kuktfa%2Bg5-4xlarge%2Bmar"

payload = {
    "input": """In what order will the logs below show up in the console?
console.log('First')

setTimeout(function () {
  console.log('Second')
}, 0)

new Promise(function (res) {
  res('Third')
}).then(console.log)

console.log('Fourth')"""
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': "Basic MjkzOllyV1NZd0NQZEotQTJNYXRpNUFZc25xWDJsVGxoSlE0"
}

# Convert the data to JSON format
json_data = json.dumps(payload)

total_time = 0
rounds = 5

for i in range(rounds):
    print(f"### Round {i + 1} ###")
    # Measure the start time of the request
    start_time = time.time()

    # Send the POST request
    response = requests.post(url + mixtral_8bit, headers=headers, data=json_data, timeout=60000)

    # Measure the end time of the request
    end_time = time.time()

    # Calculate the request duration
    request_duration = end_time - start_time

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful!")
        print("Response:")
        print(response.json())
        print("Request duration:", request_duration, "seconds")
    else:
        print("Error occurred: ", response.status_code)
        print(response.json())

    if i != 0:
        total_time += request_duration

print("Average response time: ", total_time / (rounds - 1))
