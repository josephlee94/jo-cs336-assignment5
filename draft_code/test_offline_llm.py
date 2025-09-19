from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Stop generation on a newline
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["\n"]
)

# ✅ Add trust_remote_code=True for Qwen models
llm = LLM(model="Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)

# Print each prompt + generation
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
