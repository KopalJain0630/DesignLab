from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gzip
import xml.etree.ElementTree as ET
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3.5-mini-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

for name, module in model.named_modules():
    print(name)

lora_config = LoraConfig(
    r=8,  # Low-rank adaptation matrix size
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    target_modules=[
        "self_attn.qkv_proj",  # Attention Query-Key-Value Projection
        "self_attn.o_proj",    # Attention Output Projection
        "mlp.gate_up_proj",    # MLP Gate + Up Projection
        "mlp.down_proj"        # MLP Down Projection
    ],  # Layers to fine-tune
    task_type=TaskType.CAUSAL_LM  # Task: Causal Language Modeling
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def parse_pubmed_xml(xml_file):
    with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
    abstracts = []
    for pubmed_article in root.findall('.//PubmedArticle'):
        article = pubmed_article.find('.//Article')
        if article is not None:
            abstract_tag = article.find('.//Abstract')
            if abstract_tag is not None:
                # Extracting the text from the abstract
                abstract_text = ''.join(abstract_tag.itertext()).strip()
                if abstract_text:
                    abstracts.append(abstract_text)
    return abstracts

pubmed_file = 'pubmed25n0001.xml.gz'
abstracts = parse_pubmed_xml(pubmed_file)
print(f"Extracted {len(abstracts)} abstracts.")
abstracts

data = {"text": abstracts}  # abstracts = Small subset of PubMed data
dataset = Dataset.from_dict(data)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

def tokenize_function(example):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels for causal LM
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Use 1 epoch for testing; increase for better results
    per_device_train_batch_size=1,  # Small batch size due to CPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate larger batch size
    save_strategy="epoch",  # Save at the end of each epoch
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    report_to="none",  # Disable wandb, tensorboard logs
    load_best_model_at_end=True,
    optim="adamw_torch"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
