import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from datasets import Dataset

model_path = "gpt2-kenyan-clinical"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-kenyan-clinical2")
model = GPT2LMHeadModel.from_pretrained(model_path)

model.eval()

def generate_clinical_response(prompt_text, max_length=512, temperature=0.7, top_p=0.9):
    """
    Generate a clinical response from a prompt
    """
    formatted_prompt = f"Question: {prompt_text}\n\nAnswer:"

    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in generated_text:
        response = generated_text.split("Answer:", 1)[1].strip()
        return response
    else:
        return generated_text


def batch_inference(prompts_list):
    """
    Generate response from multiple promt
    """
    results = []
    for i, prompt in enumerate(prompts_list):
        print(f"Processing prompt {i+1}/{len(prompts_list)}...")
        response = generate_clinical_response(prompt)
        results.append({
            'prompt': prompt,
            'response': response
        })
    return results

def save_inference_results(results, filename="inference_results.json"):
    """
    Save result
    """
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":

    df_eval = pd.read_csv("test_with_responses.csv") 
    eval_dataset = Dataset.from_pandas(df_eval)
    test_prompts = eval_dataset["llm_response"]

    print("\n" + "="*50)
    print("BATCH INFERENCE TEST")
    print("="*50)

    results = batch_inference(test_prompts)

    save_inference_results(results)

    print("\nInference completed!")