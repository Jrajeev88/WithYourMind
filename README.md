# WithYourMind - Emotional Support Chatbot

This is an emotional support chatbot built using a fine-tuned Mistral-7B-Instruct model, capable of providing empathetic responses in both English and Malayalam. It also includes robust crisis detection and provides region-specific helpline information for India.

## Features:
- Empathetic conversational AI
- Supports English and Malayalam languages
- Crisis detection with immediate helpline referrals (India-specific)
- Built with Hugging Face Transformers, PEFT, and Gradio

## How to use:
1. Clone this repository.
2. Ensure you have the `fine_tuned_model` directory (containing the saved model and tokenizer) in the same location as `app.py`.
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Gradio application: `python app.py`

## Crisis Information:
If you or someone you know is in crisis, please seek immediate help:
-   **India-specific crisis support:**
    -   AASRA (24/7 Suicide Prevention): 022 2754 6669
    -   Govt. KIRAN Mental Health Helpline: 1800-599-0019
    -   iCALL WhatsApp Support: +91 9152987821
    -   Tele Manas: 14416
-   **Global:** Visit https://www.iasp.info/resources/Crisis_Centres/ for a list of international helplines.

## Disclaimer:
This chatbot is a non-clinical emotional support companion. It does not diagnose, prescribe, or provide professional medical or psychological advice. It is not a substitute for professional help. In emergencies, please contact qualified professionals or emergency services.
