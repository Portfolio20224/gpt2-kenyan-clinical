from tqdm import tqdm
from langchain_ollama import ChatOllama
import pandas as pd

TEMPLATE = """You are a medical information extraction assistant.

Extract the following structured data from the clinical vignette:
- [CLINICIAN]: Summarize the clinician's experience, facility, and county.
- [PATIENT]: Include the patient's age, symptoms, and a brief summary.
- [CONSTRAINTS]: Identify any resource limitations.
- [QUESTIONS]: List all clinician's questions in numbered form.

Here are two examples:
--- EXAMPLE 1 ---

Text:
I am a nurse with 12 years of experience in General nursing working in a Sub-county Hospitals and Nursing Homes in Kiambu county in Kenya. Forty-seven years, old man, came to Casualty, supported by two men, screaming, because of pain. Upon inquiry, he reported that he had severe abdominal pain, upper abdominal pain, in the gastric area, which had started the previous night. He had not slept. He also reported that this was not the first time that it was happening, and he reported a history of PUD. This time it was severe. So on observations, the PUDs were, the vitals were within normal range. The questions I had were, an analgesic first? or this patient was sent for labs? or to do a scan first?

Structured Output:
[CLINICIAN] Nurse, 12 years' experience in General Nursing, Sub-county Hospitals and Nursing Homes, Kiambu county, Kenya
[PATIENT] 47M, severe upper abdominal pain (gastric area), history of PUD, no sleep due to pain, vitals normal
[CONSTRAINTS] Likely limited diagnostic equipment (e.g., imaging), typical of Sub-county Hospitals in Kenya
[QUESTIONS]
1. An analgesic first?
2. Send for labs?
3. Do a scan first?
→ Decision:

--- END EXAMPLE 1 ---

--- EXAMPLE 2 ---

Text:
I am a nurse with 12 years of experience in Primary care working in a National Referral Hospitals in Uasin Gishu county in Kenya. ER, aged 92 years, female was brought in with inability to walk, abdominal pain, generalized body malaise and history of fecal impaction. On assessment she is sick with GCS of 4/15. BP-131/82mmHg, MAP-96 , HR-92, RR-17b/min, Temp-36.3 , SP02-68%. Questions: What emergency care should patient ER receive? What investigation should be done? What is the diagnosis?

Structured Output:
[CLINICIAN] Nurse, 12 years' experience in Primary Care, National Referral Hospital, Uasin Gishu county, Kenya
[PATIENT] 92F, inability to walk, abdominal pain, general malaise, history of fecal impaction; very low GCS (4/15), hypoxia (SpO2 68%)
[CONSTRAINTS] None explicitly mentioned; National Referral Hospital likely has access to advanced resources
[QUESTIONS]
1. What emergency care should patient ER receive?
2. What investigation should be done?
3. What is the diagnosis?
→ Decision:

--- END EXAMPLE 2 ---


Now extract the same structure from the following text:

Text:
{input_text}

Structured Output:"""


# ===============================
# Fonctions
# ===============================
def clean_response(response: str) -> str:
    if response.startswith("```"):
        response = response.strip("`")
        response = response.replace("python", "").strip()
    if response.endswith("```"):
        response = response.rstrip("`").strip()
    return response

def process_annotation(llm, input_text, template, max_attempts=5):
    prompt = template.format(input_text=input_text)

    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt).content.strip()
            response = clean_response(response)
            return response

        except Exception as e:
            tqdm.write(f"[{input_text[:20]}...] Échec tentative {attempt + 1}: {e}")

    return None

# ===============================
# Traitement principal
# ===============================
llm = ChatOllama(model="phi4", temperature=0.3)

df = pd.read_csv("test_raw.csv")

responses = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    input_text = row["Prompt"]

    response = process_annotation(llm, input_text, TEMPLATE)

    responses.append(response)

df['llm_response'] = responses

df.to_csv(df, "out/test_with_responses.csv")

successful_responses = sum(1 for r in responses if r is not None)
failed_responses = len(responses) - successful_responses

tqdm.write(f"\n✅ Total succeed answers : {successful_responses}")
tqdm.write(f"❌ failed answers : {failed_responses}")
tqdm.write(f"📄 DataFrame saved on : out/test_with_responses.csv")