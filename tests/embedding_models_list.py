from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Listar TODOS los modelos disponibles
models = genai.list_models()
for model in models:
    print(f"Modelo: {model.name}")
    if hasattr(model, 'display_name') and model.display_name:
        print(f"  Nombre: {model.display_name}")
    print(f"  Métodos soportados: {model.supported_generation_methods}")
    print("---")