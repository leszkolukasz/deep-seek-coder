from transformers import AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained(
    "TheBloke/deepseek-coder-1.3b-instruct-GGUF",
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,
)
