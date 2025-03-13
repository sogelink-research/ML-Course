import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

## v2 models
model_path = "openlm-research/open_llama_3b_v2"
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


stop_words = ["Q:"]
stop_words_ids = [
    tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
    for stop_word in stop_words
]
print(f"{stop_words_ids = }")
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

prompt = "Q: What is the color of the sky?\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]
outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    stopping_criteria=stopping_criteria,
    pad_token_id=tokenizer.eos_token_id,
    return_legacy_cache=True,  # Ensure legacy format
)

# token = outputs[0, input_length:]
token = outputs[0]
print(f"{token = }")
output_str = tokenizer.decode(token, skip_special_tokens=True)
# output_str = output_str.replace("<human>:", "")
print("Answer:\n")
print(output_str)
