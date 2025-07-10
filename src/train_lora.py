"""
python src/train_lora.py feedback/negative.tsv
Produces: models/lora-adapter/
"""
import sys, pathlib, csv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer  # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "unsloth/llama-3-13b-instruct"    # base HF model (same family as GGUF)
neg_path = pathlib.Path(sys.argv[1])

records = []
with neg_path.open() as f:
    for ts, code, q, bad, correct in csv.reader(f, delimiter="\t"):
        prompt  = f"<|user|> {q}\n<|assistant|> {correct}"
        records.append({"text": prompt})

ds = Dataset.from_list(records)

tok   = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, load_in_4bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_cfg)

args = TrainingArguments(
    output_dir="models/lora-adapter",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=True,
)
Trainer(model, args, train_dataset=ds, tokenizer=tok).train()
model.save_pretrained("models/lora-adapter")
tok.save_pretrained("models/lora-adapter")
print("✅ LoRA adapter saved → models/lora-adapter/")