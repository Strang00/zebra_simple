# Zebra Simple Test

Implementation of [Zebra Puzzle](https://en.wikipedia.org/wiki/Zebra_Puzzle) for LLM, simple variation of [ZebraLogic](https://huggingface.co/blog/yuchenlin/zebra-logic) bench.

What it does exactly:
- loads [ZebraLogic dataset](https://huggingface.co/datasets/WildEval/ZebraLogic) for local work if none yet
- queries and checks N zebra puzzles of specified SIZE for each MODEL in list
- stores json log for each model and summary, extracts model answer and think

Key differences from [ZeroEval](https://github.com/WildEval/ZeroEval):
- requires only `openai` and `duckdb` pip packages and access to APIs
- works via OpenAI API also with Groq, Google, Anthropic and DeepSeek
- switches thinking mode with model suffix `none/default/low/medium/high` for all APIs
- extracts `<think>` or `<thought>` tag reasoning to separate thought data from results
- DOES NOT implement any other benches except zebra puzzle of specified size


## Installation

```sh
pip install openai
pip install duckdb
```

## Usage

```sh
# Configure keys for your models
export GROQ_API_KEY="set your key for @groq models"
export GOOGLE_API_KEY="set your key for @google models"
export NVIDIA_API_KEY="set your key for @nvidia models"

# Run 10 random zebra puzzles 2*2 for models from models.txt
python -u zebra.py -f models.txt -s 2*2 -r 10

# Run 2 exact zebra puzzles 3*3 #0 and #1 for one Groq model
python -u zebra.py -m llama-3.3-70b-versatile@groq -s 3*3 -i 0,1 -t 4096
```


## Results example

`Finished processing 4 models * 10 tests in 0:04:09.991346`

```json
{
    "duration": "0:04:09.991346",
    "size": "2*2",
    "count": 10,
    "max_tokens": 1024,
    "models_summary": {
        "nvidia/llama-3.1-nemotron-70b-instruct@nvidia": 90,
        "nvidia/llama-3.1-nemotron-ultra-253b-v1@nvidia": 100,
        "gemini-2.5-flash-preview-05-20-none@google": 100,
        "gemini-2.5-flash-preview-05-20-low@google": 100
    }
}
```


## Supported models

**Common rules**

- model should contain OpenAI API compatible model name, optional reasoining_effort suffix and `@source`
- each source addressed via OpenAI API with key from environment variable as `GOOGLE_API_KEY` for `@google`
- `reasoning_effort` controlled by suffix after the model name: `none`, `default`, `low`, `medium`, `high`
- @google and @nvidia have specific way of defining thinking mode to allow catch `<think>` or `<thought>` tags
- full model name example: `gemini-2.5-flash-preview-05-20-low@google`

**Models by source**

- @groq models https://console.groq.com/docs/models
- @google models https://ai.google.dev/gemini-api/docs/models
- @nvidia models https://docs.api.nvidia.com/nim/reference/llm-apis
- @xai models https://docs.x.ai/docs/models?cluster=us-east-1
- @deepseek models https://api-docs.deepseek.com/
- @anthropic models https://docs.anthropic.com/en/docs/about-claude/models/overview#model-aliases
- @openai models https://platform.openai.com/docs/pricing

**Keys by source**

- GROQ_API_KEY https://console.groq.com/keys
- GOOGLE_API_KEY https://makersuite.google.com/app/apikey
- NVIDIA_API_KEY https://build.nvidia.com/settings/api-keys
- XAI_API_KEY https://console.x.ai/
- DEEPSEEK_API_KEY keys https://platform.deepseek.com/api-keys
- ANTHROPIC_API_KEY https://console.anthropic.com/settings/keys
- OPENAI_API_KEY https://platform.openai.com/api-keys
- CUSTOM_API_KEY and CUSTOM_API_URL could point to any, @custom model suffix

**Models examples**

```sh
gpt-4.1-nano-high@openai                            # no free
claude-sonnet-4-0-high@anthropic                    # no free
deepseek-reasoner-high@deepseek                     # no free
grok-3-mini-high@xai                                # no free
nvidia/llama-3.1-nemotron-70b-instruct@nvidia       # free for devs, no reasoning
nvidia/llama-3.1-nemotron-ultra-253b-v1-high@nvidia # free for devs, have <think>
gemini-2.5-pro-exp-03-25-none@google                # pro is not free
gemini-2.5-flash-preview-05-20-default@google       # have <thought>
gemini-2.0-flash@google                             # no reasoning
qwen/qwen3-32b-default@groq                         # reasoning none and default, have <think>
qwen-qwq-32b@groq                                   # no reasoning control, but have <think>
deepseek-r1-distill-llama-70b@groq                  # no reasoning control, but have <think>
llama-3.3-70b-versatile@groq                        # no reasoning
llama3-8b-8192@groq                                 # no reasoning
gemma2-9b-it@groq                                   # no reasoning  
allam-2-7b@groq                                     # no reasoning, weak and good to test size 2*2
```


## Acknowledgements

1. First thanks to Creative Workshop team inspired to do this simple test
2. Authors of ZebraLogic and [paper](https://arxiv.org/abs/2502.01100) created good and reusable results, nice work!
3. Big Thanks for WildEval published useful [dataset](https://huggingface.co/datasets/WildEval/ZebraLogic) with solutions 
4. And warmest thanks to author of [Sherlock game](https://www.kaser.com/sherwin.html) implementing puzzle.
   Many happy hours I spent with this game, now it available as [Watson desktop](https://github.com/koro-xx/Watson) and [Sherlok mobile](https://apps.apple.com/ru/app/sherlock-free/id631434866)

## History

- 2025-06-12 Initial version with dataset load (ow, original ZebraLogic by allenai/ZeroEval have no solution)
- 2025-06-13 Extracted and modified prompt from [ZeroEval](https://github.com/WildEval/ZeroEval), completion request and results comparison logic
- 2025-06-14 Created results saving logic, tested models from different sources, improved completion request
- 2025-06-15 Research for `reasoning_effort` and similars to control thinking, extract `<think>` or `<thought>`
- 2025-06-16 Polished results for easy usage by Creative Workshop team, prepared for publishing repos
- 2025-06-16 Added support command-line arguments and usage help, added direct models in args and items list


## Tasks

- [x] Add and test zebra.py
- [x] Add saving parquet if not exists, do not store in our repos, read license
- [x] Add config.py and results to .gitignore
- [x] Add delay option between tests (bypassing RPM/TPM free limits)
- [x] Add and fill LICENSE, check used parts license and reference here
- [x] Collect links to models and keys retrival, models examples with comments
- [x] Implement command-line args retrieval and usage help output
