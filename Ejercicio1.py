from transformers import pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
prompt = pipe.tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Eres un docente."},
        {"role": "user", "content": "Define 'embedding' en una sola línea."}
    ],
    tokenize=False,
    add_generation_prompt=True
)

out = pipe(
    prompt,
    max_new_tokens=30,
    do_sample=False,             # determinista
    return_full_text=False,
    eos_token_id=pipe.model.config.eos_token_id,
    pad_token_id=pipe.model.config.eos_token_id
)
print(out[0]["generated_text"])