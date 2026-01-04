import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from deep_translator import GoogleTranslator
import torch
import os

# Create offload directory if it doesn't exist, needed by device_map="auto"
os.makedirs("offload", exist_ok=True)

# --- Model Loading ---
# Path to your fine-tuned model. Make sure this directory contains
# your saved model (e.g., adapter_model.bin) and tokenizer files.
# If deploying to Hugging Face Spaces, you would typically upload this directory
# as part of your repository, or specify a Hugging Face model ID.
FINE_TUNED_MODEL_PATH = "./fine_tuned_model"

# Configure BitsAndBytes for 4-bit quantization if your saved model is quantized.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    llm_int8_enable_fp32_cpu_offload=True # Needed if parts are offloaded to CPU
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    FINE_TUNED_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto", # Automatically distributes model layers across available devices
    offload_folder="offload", # Directory for CPU offloading
    low_cpu_mem_usage=True # Reduces CPU RAM usage during loading
)
model.eval() # Set model to evaluation mode
if torch.cuda.is_available():
    model.to("cuda") # Ensure model is on GPU if available

# Set generation configuration for consistent outputs
model.generation_config = GenerationConfig(
    max_new_tokens=180,
    temperature=0.7,
    top_p=0.85,
    do_sample=True,
    repetition_penalty=1.25,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# --- Translation and Crisis Functions ---
translator = GoogleTranslator(source='auto', target='en')

malayalam_chars = set("അആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ")
def is_malayalam(text):
    return any(ch in malayalam_chars for ch in text)

def emergency_response_ml():
    return (
        "നിങ്ങൾ പങ്കുവെച്ചത് വളരെ പ്രധാനപ്പെട്ടതാണ്. ഞാൻ ഇവിടെ കേൾക്കാൻ ആണ്, "
        "എന്നാൽ അടിയന്തിര സാഹചര്യങ്ങളിൽ നേരിട്ട് സഹായം നൽകാൻ എനിക്ക് കഴിയില്ല.\n\n"
        "📍 **ഇന്ത്യയിലെ അടിയന്തിര സഹായ നമ്പറുകൾ:**\n"
        "• AASRA (24/7): 022 2754 6669\n"
        "• സർക്കാർ KIRAN ഹെൽപ്ലൈൻ: 1800-599-0019\n"
        "• iCALL (WhatsApp/ചാറ്റ്): +91 9152987821\n\n"
        ". Tele Manas: 14416\n"
        "📌 ദയവായി ഒരു മനുഷ്യനുമായി ഇപ്പോൾ തന്നെ ബന്ധപ്പെടുക.\n"
        "📌 നിങ്ങൾ ഒറ്റക്കല്ല.\n"
    )

def emergency_response_en():
    return (
        "Thank you for saying that. I’m here to listen, but I can’t provide emergency help.\n\n"
        "📍 **India-specific crisis support:**\n"
        "• AASRA (24/7 Suicide Prevention): 022 2754 6669\n"
        "• Govt. KIRAN Mental Health Helpline: 1800-599-0019\n"
        "• iCALL WhatsApp Support: +91 9152987821\n\n"
        ". Tele Manas: 14416\n"
        "If you're in immediate danger, please contact a human right now.\n"
        "You are not alone.\n"
    )

def check_crisis_status(user_input):
    crisis_terms = [
        "suicide", "kill myself", "end my life", "self harm",
        "can't go on", "want to die", "hopeless", "worthless",
        "no reason to live", "take my life", "harm myself", "give up",
        "feel like dying", "don't want to live", "better off dead",
        "suicidal thoughts", "ending it all"
    ]

    if any(w in user_input.lower() for w in crisis_terms):
        return emergency_response_en()
    return None

def naturalize_malayalam(text):
    replacements = {
        "നിങ്ങൾ ഒരാളല്ല": "നിങ്ങൾ ഒറ്റക്കല്ല",
        "ഒരു തീരുമാനം എടുക്കണം": "ഒരു തീരുമാനത്തിലേക്ക് എത്തണം",
        "ഞാൻ ഇവിടെ നിങ്ങളോടൊപ്പം": "ഞാൻ നിങ്ങളോടൊപ്പം ഉണ്ടെന്ന് കരുതുക",
        "നിങ്ങൾക്ക് അത് ബുദ്ധിമുട്ടാകാം": "അത് വെറും എളുപ്പമല്ലെന്ന് തോന്നുന്നു",
        "കഠിനമാണ്": "ബുദ്ധിമുട്ടായി തോന്നുന്നു",
        "ആശയക്കുഴപ്പം": "കുഴപ്പം",
        "സഹായം തേടുക": "ആളുമായി സംസാരിക്കുക"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def withyourmind_chat(user_input, history=""):
    prompt = f"""
You are WithYourMind, a non-clinical emotional support companion.\nRespond with empathy, reflection, and grounding. Do not diagnose or claim professional authority.\n\nConversation so far:\n{history}\n\nUser: {user_input}\nWithYourMind:\n""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=model.generation_config.temperature,
        top_p=model.generation_config.top_p,
        repetition_penalty=model.generation_config.repetition_penalty
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("WithYourMind:")[-1].strip()

def withyourmind_ml(user_input, history=""):
    crisis_terms_ml = ["ആത്മഹത്യ","ആത്മഹത്യാപരമായ","ജീവിതം അവസാനിപ്പിക്കുക","പോകണം","എനിക്കു ജീവിക്കാനില്ല"]
    if any(term in user_input for term in crisis_terms_ml):
        return emergency_response_ml()

    if is_malayalam(user_input):
        en = translator.translate(user_input)
        reply_en = withyourmind_chat(en, history)
        reply_ml = GoogleTranslator(source='en', target='ml').translate(reply_en)
        return naturalize_malayalam(reply_ml)
    return withyourmind_chat(user_input, history)

# --- Gradio Interface Function ---
def gradio_chat(user_input, history):
    if history is None:
        history = []

    english_crisis_response = check_crisis_status(user_input)
    if english_crisis_response:
        history.append((user_input, english_crisis_response))
        return history

    if is_malayalam(user_input):
        reply = withyourmind_ml(user_input, history)
        history.append((user_input, reply))
        return history

    reply = withyourmind_chat(user_input, history)
    history.append((user_input, reply))
    return history

# --- Gradio Blocks Setup ---
if __name__ == "__main__":
    with gr.Blocks(theme="soft") as iface:
        gr.Markdown("# 🤝 WithYourMind — Emotional Support Chatbot")
        gr.Markdown("Non-clinical emotional support • Empathetic responses • Crisis-safe boundaries")

        chatbot = gr.Chatbot(height=450)
        msg = gr.Textbox(label="How are you feeling today?", placeholder="Type here...")

        msg.submit(gradio_chat, [msg, chatbot], chatbot)
        msg.submit(lambda x: "", msg)

    iface.launch(share=True)
