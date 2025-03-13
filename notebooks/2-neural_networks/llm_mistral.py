from pathlib import Path

from huggingface_hub import snapshot_download
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer

if Path.cwd().name == "2-neural_networks":
    Path.cwd().chdir(Path.cwd().parent)

mistral_models_path = Path(
    "2-neural_networks", "data", "mistral_models", "7B-Instruct-v0.3"
)
print(mistral_models_path)
mistral_models_path.mkdir(parents=True, exist_ok=True)

if not any(mistral_models_path.iterdir()):
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model.v3",
        ],
        local_dir=mistral_models_path,
    )

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path, device="cpu")

completion_request = ChatCompletionRequest(
    messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")]
)

tokens = list(
    map(
        lambda t: t.to("cpu"),
        tokenizer.encode_chat_completion(completion_request).tokens,
    )
)

out_tokens, _ = generate(
    [tokens],
    model,
    max_tokens=64,
    temperature=0.0,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
