import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# The first Sovereign Kirundi Model by Nyehashi AI
model_id = "Nyehashai/Ntahokaja-0.31-Base"

print(f"Loading {model_id}...")

# Load Tokenizer (trust_remote_code runs our custom morphological logic)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load Model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            do_sample=True, 
            temperature=0.7,
            repetition_penalty=1.2
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_prompt = "Perezida w'u Burundi yavuze ko "
    print(f"\nPrompt: {test_prompt}")
    print(f"Response: {generate_text(test_prompt)}")