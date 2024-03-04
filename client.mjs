import axios from "axios";

const url = "https://autumn8functions.default.aws.autumn8.ai"
const deepseekcoder_6_7 = "/inference/4373e4e3-2502-4f28-b96a-125c2bc6faeb%2B293%2Bq5k6v6egxqi65gt1ojg9%2Bg5-2xlarge%2Bmar"
const deepseekcoder_33_5bit = "/inference/aa1caf72-8e03-481e-96a9-4a14ff289798%2B293%2Bmk1qjyoeawzn0ifdjb3u%2Bg5-4xlarge%2Bmar"
const deepseekcoder_33_8bit = "/inference/bea7c4e4-2b46-400a-b51e-7058a0ae639a%2B293%2Bshlb15o4kbkehlwlpgno%2Bg5-4xlarge%2Bmar"
const mixtral_5bit = "/inference/fe80279d-6261-47f7-b527-2ad827c911c1%2B293%2Bw29oxt3arylsrrd9laq3%2Bg5-4xlarge%2Bmar"
const mixtral_8bit = "/inference/e7c2a547-c385-47c9-96ad-e6993674c7d3%2B293%2B922ohce8dyg762kuktfa%2Bg5-4xlarge%2Bmar"


const rounds = 5;

const payload = {
    input: `
How can this code be improved?
fetch("/user")
  .then((res) => res.json())
  .then((user) => {

})
    `,
};

const headers = {
    "Content-Type": "application/json",
    Authorization: "Basic MjkzOllyV1NZd0NQZEotQTJNYXRpNUFZc25xWDJsVGxoSlE0",
};

async function sendRequest() {
    let totalDuration = 0;

    for (let i = 0; i < rounds; i++) {
        console.log(`### Round ${i + 1} ###`);

        const startTime = Date.now();

        try {
            const response = await axios.post(url + deepseekcoder_33_5bit, payload, {
                headers: headers,
            });

            const endTime = Date.now();
            const requestDuration = (endTime - startTime) / 1000;

            console.log("POST request successful!");
            console.log("Response:");
            console.log(response.data);
            console.log(response.data["message"]["output"])
            console.log("Request duration:", requestDuration, "seconds");

            if (i !== 0) {
                totalDuration += requestDuration;
            }
        } catch (error) {
            console.log("Error occurred:", error.message);
        }
    }

    console.log("Average response time:", totalDuration / (rounds - 1));
}

sendRequest();
