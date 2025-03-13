import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

MIN_TRANSFORMERS_VERSION = "4.25.1"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

# init
tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
)
model = AutoModelForCausalLM.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16
)
model = model.to("cuda:0")


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


stop_words = ["<human>:"]
stop_words_ids = [
    tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
    for stop_word in stop_words
]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# infer
prompt = """
<human>: Where is Amsterdam?\n
<bot>: - Country: Netherlands\n
- Province: North Holland\n
- Region: Amsterdam metropolitan area\n
<human>: Where is Rotterdam?\n
<bot>: - Country: Netherlands
- Province: South Holland\n
<human>: Where is London?\n
<bot>"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    return_dict_in_generate=True,
    stopping_criteria=stopping_criteria,
    pad_token_id=tokenizer.eos_token_id,
    return_legacy_cache=True,  # Ensure legacy format
)
token = outputs.sequences[0, input_length:]
output_str = tokenizer.decode(token, skip_special_tokens=True)
output_str = output_str.replace("<human>:", "")
print("Answer:\n")
print(output_str)
