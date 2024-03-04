import httpx
import json
import time

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

headers = {'Authorization': "Basic MjkzOllyV1NZd0NQZEotQTJNYXRpNUFZc25xWDJsVGxoSlE0"}

timeout = httpx.Timeout(None, connect=None, read=None, write=None)

start = time.time()

r = httpx.post(
    "https://autumn8functions.default.aws.autumn8.ai/inference/aa1caf72-8e03-481e-96a9-4a14ff289798%2B293%2Bmk1qjyoeawzn0ifdjb3u%2Bg5-4xlarge%2Bmar",
    json=payload, headers=headers, timeout=timeout)

print(time.time() - start)

print(r.json())
