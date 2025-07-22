# test_llama3.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    pipeline,
)

# the HF repo that actually exists and is public:
MODEL_ID = "MaziyarPanahi/Llama-3-13B-Instruct-v0.1"

print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Loading model…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Wiring up streamer & pipeline…")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
)

# build a little chat‐style prompt
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always speaks in pirate‑speak."},
    {"role": "user",   "content": "Who are you?"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

print("=== Prompt ===")
print(prompt)
print("=== Generation ===")
outs = pipe(
    prompt,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
)
# the streamer will have already printed token by token
# but pipeline still returns the final dict:
print("\n\nFinal output:")
print(outs[0]["generated_text"][len(prompt):])
