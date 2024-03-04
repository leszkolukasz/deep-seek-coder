import logging
from pathlib import Path
import torch
import psutil
from accelerate import (
    Accelerator,
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizerBase
from huggingface_hub import snapshot_download
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

logger = logging.getLogger(__name__)


class DeepSeekCoderHandler(BaseHandler):
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
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
        VRAM = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        RAM = psutil.virtual_memory().total / (1024 ** 3) - 6
        self.model = load_checkpoint_and_dispatch(
            self.model,
            weights_location,
            device_map="auto",
            offload_folder=self.model_dir / "offload",
            offload_state_dict=True,
            no_split_module_classes=self.model._no_split_modules,
            # dtype=torch.float,
            max_memory={"cpu": f"{RAM:.2f}GiB", 0: f"{VRAM:.2f}GiB"}
        )
        logger.info("Model loading completed")
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
        logger.info("Downloading weights")
        weights_location = snapshot_download(
            repo_id=self.MODEL_NAME,
            allow_patterns=["*.json", "*.bin"],
            ignore_patterns=["*.safetensors.index.json"],
        )
        logger.info(f"Weights location: {weights_location}")

        return weights_location
