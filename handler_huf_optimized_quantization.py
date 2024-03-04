import logging
from pathlib import Path
import psutil
import torch
from accelerate import (
    Accelerator,
    init_empty_weights,
)
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedTokenizerBase,
)
from huggingface_hub import snapshot_download
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

logger = logging.getLogger(__name__)

FETCH_QUANITZED = (
    True  # Fetch quantized weights from repo instead of quantizing on initialization
)
LOAD_IN_8BIT = True
LOAD_IN_4BIT = False

if LOAD_IN_4BIT and LOAD_IN_8BIT:
    raise Exception("Invalid config")


class DeepSeekCoderHandler(BaseHandler):
    MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
    QUANTIZED_WEIGHTS_URL = "whistleroosh/deepseek-coder-33b-instruct-8bit"
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

        logger.info("Loading model")
        weights_location, is_quantized = self._download_weights()
        if FETCH_QUANITZED or is_quantized:
            self._load_model(weights_location)
        else:
            self._quantize_and_load_model(weights_location)
        logger.info("Model loading completed")

        logger.info(f"Device map: {self.model.hf_device_map}")
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
        if not FETCH_QUANITZED and self.quantized_model_weights_dir.exists():
            logger.info("Quantized weights already exist")
            return self.quantized_model_weights_dir, True

        logger.info("Downloading weights")
        weights_location = snapshot_download(
            repo_id=self.QUANTIZED_WEIGHTS_URL if FETCH_QUANITZED else self.MODEL_NAME,
            allow_patterns=["*.json", "*.safetensors"]
            if FETCH_QUANITZED
            else ["*.json", "*.bin"],
            ignore_patterns=["*.bin.index.json"]
            if FETCH_QUANITZED
            else ["*.safetensors.index.json"],
        )
        logger.info(f"Weights location: {weights_location}")

        return weights_location, FETCH_QUANITZED

    def _quantize_and_load_model(self, weights_location):
        logger.info("Weights will be quantized and saved to the model directory")

        self._load_model(weights_location)

        logger.info("Saving quantized weights")
        self.accelerator.save_model(self.model, self.quantized_model_weights_dir)

    def _load_model(self, weights_location):
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=LOAD_IN_8BIT, load_in_4bit=LOAD_IN_4BIT
        )

        VRAM = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        RAM = psutil.virtual_memory().total / (1024**3) - 10

        self.model = load_and_quantize_model(
            self.model,
            weights_location=weights_location,
            device_map="auto",
            bnb_quantization_config=bnb_quantization_config,
            offload_folder=self.model_dir / "offload",
            offload_state_dict=True,
            no_split_module_classes=self.model._no_split_modules,
            max_memory={"cpu": f"{RAM:.2f}GiB", 0: f"{VRAM:.2f}GiB"},
        )

    @property
    def quantized_model_weights_dir(self):
        return self.model_dir / "quantized_weights"
