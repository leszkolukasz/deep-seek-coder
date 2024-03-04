import logging
from pathlib import Path
import torch
from accelerate import (
    Accelerator,
    init_empty_weights,
    infer_auto_device_map,
)
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizerBase
from huggingface_hub import snapshot_download
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

logger = logging.getLogger(__name__)

LOAD_IN_8BIT = True
LOAD_IN_4BIT = False


class DeepSeekCoderHandler(BaseHandler):
    MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    accelerator: Accelerator
    model_dir: Path

    def initialize(self, context):
        properties = context.system_properties
        manifest = context.manifest
        self.model_dir = Path(properties.get("model_dir"))

        logger.info(f"Properties: {properties}")
        logger.info(f"Manifest: {manifest}")

        self.accelerator = Accelerator()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)

        device_map = infer_auto_device_map(self.model)
        logger.info(f"Device map: {device_map}")

        weights_location = self._download_weights()

        logger.info("Loading model")
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=LOAD_IN_8BIT, load_in_4bit=LOAD_IN_4BIT
        )
        self.model = load_and_quantize_model(
            self.model,
            weights_location=weights_location,
            device_map="auto",
            bnb_quantization_config=bnb_quantization_config,
            offload_folder=self.model_dir / "offload",
        )
        logger.info("Model loading completed")

        if not self.quantized_model_weights_dir.exists():
            logger.info("Saving quantized weights")
            self.accelerator.save_model(self.model, self.quantized_model_weights_dir)

        logger.info(f"Memory footprint: {self.model.get_memory_footprint()}")

        self.model.eval()

    def preprocess(self, data):
        if "input" in data:
            data = data["input"]

        if isinstance(data, list):
            data = data[0]

        if data is None:
            raise PredictionException("Input is None")

        logger.info(f'Received: "{data}". Begin tokenizing')

        message = [{"role": "user", "content": data}]
        tokenized_input = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt"
        ).to(self.accelerator.device)

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

    def _download_weights(self):
        if self.quantized_model_weights_dir.exists():
            logger.info("Quantized weights already exist")
            return self.quantized_model_weights_dir

        logger.info("Downloading weights")
        weights_location = snapshot_download(
            repo_id=self.MODEL_NAME,
            allow_patterns=["*.json", "*.bin"],
            ignore_patterns=["*.safetensors.index.json"],
        )
        logger.info(f"Weights location: {weights_location}")
        logger.info("Weights will be quantized and saved to the model directory")

        return weights_location

    @property
    def quantized_model_weights_dir(self):
        return self.model_dir / "quantized_weights"
