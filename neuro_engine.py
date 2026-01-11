import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model you ALREADY have (or Base Model)
try:
    # Try loading your specific checkpoint if it exists
    MODEL_PATH = "./coda_neuro_symbolic" 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
except:
    # If not, use the Base Model. It is smart enough for this trick.
    print("[INFO] Using Base Model with Neuro-Symbolic Injection.")
    MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")

def analyze(site_id, missing, queries, sae, protocol_text=""):
    # --- 1. THE TRUTH (Python Logic) ---
    # This is the "Titanium" layer. It cannot be wrong.
    risk_score = 0
    if sae > 0: risk_score += 100
    if missing > 5: risk_score += 50
    if queries > 10: risk_score += 30
    
    if risk_score >= 100: status = "CRITICAL"
    elif risk_score >= 30: status = "RISK"
    else: status = "CLEAN"

    # --- 2. THE VIBE (Neural Generation) ---
    # We ask the model for a generic template based on the status.
    # We do NOT ask it to count numbers (that causes hallucinations).
    prompt = f"Write a professional clinical trial assessment for a site with status: {status}. Use formal tone."
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.3)
        ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assessment:")[-1]
    except:
        ai_text = "Assessment pending automated review."

    # --- 3. THE INJECTION (The Magic Trick) ---
    # We discard the specific numbers the AI might have guessed and OVERWRITE them with Facts.
    
    final_narrative = f"Status: {status}. "
    
    if status == "CRITICAL":
        final_narrative += f"Urgent intervention required. Analysis confirms {sae} Unresolved Safety Events (SAEs) and {queries} open queries. "
    elif status == "RISK":
        final_narrative += f"Operational monitoring advised. Site shows {missing} missing visits and potential backlog. "
    else:
        final_narrative += "Site is operating within nominal safety parameters. No immediate action required. "
        
    # We append the AI's "Vibe" at the end for flavor, but keeping it safe.
    # final_narrative += f" (AI Note: {ai_text[:50]}...)" 

    return {
        "site_id": site_id,
        "status": status,
        "risk_score": risk_score,
        "narrative": final_narrative, # <--- 100% ACCURATE BECAUSE PYTHON WROTE IT
        "patients": generate_patient_data(site_id)
    }

def generate_patient_data(site_id):
    # Keep your drill-down simulator
    return [{"patient_id": f"{site_id}-10{i}", "status": "Active", "missing_pages": random.randint(0,2)} for i in range(5)]
