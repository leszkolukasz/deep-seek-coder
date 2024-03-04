import http.client
import json
import time

conn = http.client.HTTPSConnection("autumn8functions.default.aws.autumn8.ai")

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

start = time.time()
headers = {'Authorization': "Basic MjkzOllyV1NZd0NQZEotQTJNYXRpNUFZc25xWDJsVGxoSlE0"}

conn.request("POST", "/inference/4373e4e3-2502-4f28-b96a-125c2bc6faeb%2B293%2Bq5k6v6egxqi65gt1ojg9%2Bg5-2xlarge%2Bmar",
             json.dumps(payload), headers)

res = conn.getresponse()
data = res.read()
print(time.time() - start)

output = json.loads(data.decode("utf-8"))["message"]["output"]
# output = json.loads(data.decode("utf-8"))
print(output)
