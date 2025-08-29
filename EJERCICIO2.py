from transformers import pipeline
import torch, random, numpy as np

# ---- util: semilla global (compatible con tu pipeline)
def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # (Opcional) mayor determinismo en CUDA; puede bajar performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1) Carga pipeline
pipe = pipeline("text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="auto")

# 2) Prompt en formato chat
prompt = pipe.tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Eres un docente."},
        {"role": "user", "content": "Define 'embedding' en una sola línea."}
    ],
    tokenize=False,
    add_generation_prompt=True
)

EOS = pipe.model.config.eos_token_id
PAD = pipe.model.config.eos_token_id

# 3) Línea base determinista (greedy)
out = pipe(prompt, max_new_tokens=30, do_sample=False,
           return_full_text=False, eos_token_id=EOS, pad_token_id=PAD)
print("Greedy (sin sampling) ->", out[0]["generated_text"].strip())

# 4) Comparación de temperatures (reproducible con semilla global)
for t, seed in [(0.2, 42), (0.7, 42), (1.2, 42)]:
    set_seed_all(seed)  # misma "suerte" para cada temperatura
    out = pipe(prompt,
               max_new_tokens=30,
               do_sample=True,
               temperature=t,
               top_p=0.9,
               return_full_text=False,
               eos_token_id=EOS, pad_token_id=PAD)
    print(f"Temp {t} ->", out[0]["generated_text"].strip())
