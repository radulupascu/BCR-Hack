import openai
import json

# Înlocuiește cu cheia ta API de la OpenAI
openai.api_key = ''

# Calea către fișierul JSON cu datele tale
file_path = 'finetune.json'

# Încărcarea datelor de antrenament
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Crearea unui fișier de antrenament pe OpenAI
response = openai.File.create(
    file=open(file_path),
    purpose='fine-tune'
)

file_id = response['id']

# Rularea fine-tuning-ului
fine_tune_response = openai.FineTune.create(
    training_file=file_id,
    model="gpt-4",  # Sau orice alt model pe care vrei să îl ajustezi
    n_epochs=4,  # Numărul de epoci de antrenament
    # Adaugă aici orice alte opțiuni necesare
)

print(f"Fine-tuning job ID: {fine_tune_response['id']}")