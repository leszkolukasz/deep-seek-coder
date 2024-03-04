import logging
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

logger = logging.getLogger(__name__)

GPU_ENABLED = True
SIZE = "33"
QUANTIZATION = "8"

CONFIG = {
    "33": {
        "8": {
            "fullname": "Q8_0",
            "offload": 20
        },
        "5": {
            "fullname": "Q5_K_M",
            "offload": 40
        }
    },
    "6.7": {
        "8": {
            "fullname": "Q8_0",
            "offload": -1
        }
    }
}

config = CONFIG[SIZE][QUANTIZATION]


class DeepSeekCoderHandler(BaseHandler):
    HUF_REPO = f"TheBloke/deepseek-coder-{SIZE}B-instruct-GGUF"
    MODEL_NAME = f"deepseek-coder-{SIZE}b-instruct.{config['fullname']}.gguf"
    TOKENIZER_NAME = f"deepseek-ai/deepseek-coder-{SIZE}b-instruct"
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    map_location: str
    device: torch.device

    def initialize(self, context):
        properties = context.system_properties
        manifest = context.manifest

        logger.info(f"Properties: {properties}")
        logger.info(f"Manifest: {manifest}")

        if (
                torch.cuda.is_available()
                and properties.get("gpu_id") is not None
                and GPU_ENABLED
        ):
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        logger.info(f"Found device {self.device}")

        model_path = self.download_model()

        self.model = Llama(
            model_path=model_path,
            n_ctx=0,
            main_gpu=properties.get("gpu_id"),
            n_gpu_layers=config['offload'] if self.map_location == "cuda" else 0,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_NAME, trust_remote_code=True
        )

    def preprocess(self, data):
        if "input" in data:
            data = data["input"]

        if isinstance(data, list):
            data = data[0]

        if data is None:
            raise PredictionException("Input is None")

        logger.info(f'Received: "{input}". Begin tokenization')

        message = [{"role": "user", "content": data}]
        prompt = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )

        logger.info("Tokenization process completed")

        return prompt

    def inference(self, prompt, *args, **kwargs):
        logger.info("Begin inference")

        with torch.inference_mode():
            output = self.model(
                prompt,
                max_tokens=-1,
                temperature=0.2,
                repeat_penalty=1.1,
                top_k=50,
                top_p=0.95,
                echo=False,
            )

        logger.info("Inference completed")

        return output

    def postprocess(self, data):
        output = data["choices"][0]["text"].strip()

        logger.info(f"Postprocessing completed. Output: {output}")

        return output

    def handle(self, data, context):
        input = self.preprocess(data)
        output = self.inference(input)
        output = self.postprocess(output)

        return output

    def download_model(self):
        logger.info("Downloading model")
        model_path = hf_hub_download(
            repo_id=self.HUF_REPO,
            filename=self.MODEL_NAME)
        logger.info("Model downloaded successfully")
        return model_path
