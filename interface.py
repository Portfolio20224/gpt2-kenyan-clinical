import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "gpt2-kenyan-clinical"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-kenyan-clinical2")
model = GPT2LMHeadModel.from_pretrained(model_path)


MEDICAL_CHECKS = {
    "antibiotics": ["infection", "bacterial", "sepsis"],
    "antipyretics": ["fever", "temperature >38"],
    "referral": ["fracture", "deep foreign body", "respiratory distress"]
}

def validate_response(response, prompt):
    """Check """
    warnings = []
    
    for med, keywords in MEDICAL_CHECKS.items():
        if med in response.lower() and not any(kw in prompt.lower() for kw in keywords):
            warnings.append(f"⚠️ {med.capitalize()} suggéré sans indication claire")
    
    if "female" in prompt.lower() and " male " in response.lower():
        warnings.append("⚠️ Incohérence de genre détectée")
    
    return warnings

def generate_response(prompt, max_length=300, temperature=0.7):
    """Génère une réponse clinique sécurisée"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        warnings = validate_response(response, prompt)
        
        if warnings:
            response = f"{response}\n\n{' '.join(warnings)}"
        
        return response
    
    except Exception as e:
        return f"Erreur: {str(e)}"

css = """
.gradio-container {max-width: 800px !important}
footer {visibility: hidden}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""
    ## 🇰🇪 Clinical Decision Support - Kenya
    *Entrez un cas clinique structuré comme pendant l'entraînement*
    """)
    
    with gr.Row():
        with gr.Column():
            input_prompt = gr.Textbox(
                label="Cas clinique",
                placeholder="""[CLINICIAN] Nurse, 5y exp, Sub-county Hospital... 
[PATIENT] 30F, fever 39°C...
[CONSTRAINTS] No lab tests...""",
                lines=7
            )
            generate_btn = gr.Button("Générer la décision")
            
        with gr.Column():
            output = gr.Textbox(
                label="Décision clinique",
                interactive=False,
                lines=10
            )
    
    examples = [
        ["[CLINICIAN] Nurse, 2y exp, Dispensary\n[PATIENT] 3mo febrile, pale\n[CONSTRAINTS] No lab tests\n[QUESTIONS]\n1. Likely diagnosis?\n→ Decision:"],
        ["[CLINICIAN] Nurse, 10y exp, Sub-county Hospital\n[PATIENT] 28F, lower abdominal pain, 38/40 weeks\n[CONSTRAINTS] Limited ultrasound\n→ Decision:"]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=input_prompt,
        label="Exemples cliniques"
    )
    
    with gr.Accordion("Paramètres avancés", open=False):
        max_length = gr.Slider(100, 500, value=300, label="Longueur max")
        temperature = gr.Slider(0.1, 1.0, value=0.7, label="Créativité (température)")
    
    generate_btn.click(
        fn=generate_response,
        inputs=[input_prompt, max_length, temperature],
        outputs=output
    )
    
    gr.Markdown("""
    **Structure recommandée:**
    - [CLINICIAN] Expérience, établissement
    - [PATIENT] Âge, sexe, symptômes
    - [CONSTRAINTS] Limites matérielles
    - [QUESTIONS] Questions spécifiques
    """)

demo.launch()