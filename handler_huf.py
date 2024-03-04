import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

logger = logging.getLogger(__name__)

GPU_ENABLED = True
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True


class DeepSeekCoderHandler(BaseHandler):
    MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
    manifest: any
    model: any
    tokenizer: any
    map_location: str
    device: any

    def initialize(self, context):
        properties = context.system_properties
        self.manifest = context.manifest

        logger.info(f"Properties: {properties}")
        logger.info(f"Manifest: {self.manifest}")

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

        logger.info(f"Using device {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )

        # Qunatization requires GPU, accelerate and bitsandbytes
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        )

        if not LOAD_IN_4BIT and not LOAD_IN_8BIT:
            self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        input = data

        if "input" in data:
            input = data["input"]

        if isinstance(input, list):
            input = input[0]

        if input is None:
            raise PredictionException("Input is None")

        logger.info(f'Received: "{input}". Begin tokenizing')

        message = [{"role": "user", "content": input}]
        tokenized_input = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        logger.info("Tokenization process completed")

        return tokenized_input

    def inference(self, data, *args, **kwargs):
        logger.info("Begin inference")

        with torch.inference_mode():
            output = self.model.generate(
                data,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=32021,
            )

        logger.info("Inference completed")

        return output

    def postprocess(self, data, input_len):
        output = self.tokenizer.decode(data[0][input_len:], skip_special_tokens=True)

        logger.info(f"Postprocessing completed. Output: {output}")

        return output

    def handle(self, data, context):
        input = self.preprocess(data)
        output = self.inference(input)
        output = self.postprocess(output, input.shape[1])

        return output
