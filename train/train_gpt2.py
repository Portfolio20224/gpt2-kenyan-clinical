from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv("train_with_responses.csv") 
dataset = Dataset.from_pandas(df)



tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")


special_tokens = {
    "additional_special_tokens": [
        "[CLINICIAN]",
        "[PATIENT]",
        "[CONSTRAINTS]",
        "[QUESTIONS]",
        "→ Decision :"
    ]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token 
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):

    full_text = []
    for i in range(len(examples["llm_response"])):
        text = f"Question: {examples['llm_response'][i]}\n\nAnswer: {examples['Clinician'][i]}"
        full_text.append(text)

    model_inputs = tokenizer(
        full_text,
        max_length=512,  
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="gpt2-kenyan-clinical",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    logging_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    report_to="none",  
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # Add eval_dataset if you have validation data
)

trainer.train()

