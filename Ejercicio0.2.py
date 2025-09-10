from transformers import pipeline
import torch

# Fijar semilla aleatoria para consistencia
torch.manual_seed(42)

# Cargar el modelo GPT-2
llm = pipeline("text-generation", model="gpt2", device=0)  # Cambiar a 0 si tienes GPU, sino usa -1 para CPU

# Función para "thinking" (con razonamiento)
def generate_with_thinking(prompt):
    # Crear una cadena de razonamiento, añadiendo explicaciones intermedias.
    chain_of_thought = f"Vamos a pensar sobre lo que es un embedding: \n\n{prompt} \n\nPrimero, un embedding es una representación matemática de una palabra, frase, o incluso un documento, en un espacio vectorial. El objetivo de un embedding es transformar las palabras en números que pueden ser procesados por una máquina. Ahora, para entender cómo se logra esto, debemos considerar que las palabras se representan en un espacio multidimensional, donde palabras con significados similares están más cerca unas de otras en este espacio."
    
    # Generar la respuesta con el razonamiento
    response = llm(chain_of_thought, max_length=200, truncation=True, pad_token_id=50256)
    return response[0]["generated_text"]

# Pregunta
prompt = "What is an embedding in the context of large language models?"

# Generación de respuesta con razonamiento
thinking_response = generate_with_thinking(prompt)

# Mostrar la respuesta
print("Response with Thinking (Chain of Thought):")
print(thinking_response)
