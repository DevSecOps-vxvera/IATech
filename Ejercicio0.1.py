from transformers import pipeline
import torch

# Fijar semilla aleatoria para obtener resultados consistentes
torch.manual_seed(42)  # Establecer semilla de PyTorch para consistencia

# Cargar el modelo GPT-2
llm = pipeline("text-generation", model="gpt2", device=0)  # Cambiar a 0 si tienes GPU, sino usa -1 para CPU

# Función para generar una respuesta simple en inglés (sin razonamiento)
def generate_simple_response(prompt):
    # Generar respuesta simple, concisa y directa en inglés
    return llm(prompt, max_length=50, truncation=True, pad_token_id=50256)[0]["generated_text"]

# Pregunta en inglés
prompt = "What is an LLM?"

# Generación de respuesta simple en inglés
simple_response = generate_simple_response(prompt)

# Mostrar la respuesta simple en inglés
print("Simple response in English:")
print(simple_response)
