import re
import os
import datetime
import time
from typing import List
from openai import OpenAI

#OPENAI_RATE_LIMIT_ERROR = openai.RateLimitError
#OPENAI_API_ERROR = openai.APIError

import json
import random
from zebra_template import ZEBRA_GRID


def get_zebra_grid_prompt(item):
    """
    Generate prompt from dataset item, as is from ZeroEval apply_lgp_grid_template()
    input: object with 'puzzle' and 'solution' fields
    output: generated prompt for model
    """
    prompt_str = ZEBRA_GRID[:]
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str


def get_zebra_grid_solution(item):
    """
    Generate solution object from dataset item, comparable with model answer, writen by Strang
    input: object with 'solution' field like this: { "solution": {"header":["House","Name","Height"],"rows":[["1","Arnold","short"],["2","Eric","very short"]]} }
    output: object requested by prompt like this: {'House 1': {'Name': 'Arnold', 'Height': 'short'}, 'House 2': {'Name': 'Eric', 'Height': 'very short'}}
    """
    solution_comparable = dict()
    for row in item['solution']['rows']:
        key = item['solution']['header'][0] + ' ' + row[0]
        obj = dict()
        ix = 1
        for colname in item['solution']['header'][1:]:
            obj[colname] = row[ix]
            ix += 1
        solution_comparable[key] = obj
    return solution_comparable


# Based on ZeroEval, modified (removed dataset loading and optimized random pick)
def zebra_test(dataset):
    #dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    #dataset = list(data)
    #dataset = data.to_dict(orient='records')
    item = random.choice(dataset)
    print(get_zebra_grid_prompt(item))
    print("-"*100)
    print(json.dumps(item["solution"], indent=2))
    print(get_zebra_grid_solution(item))


def download_zebra_dataset():
    import requests

    r = requests.get("https://datasets-server.huggingface.co/parquet?dataset=WildEval/ZebraLogic")
    j = r.json()
    urls = [f['url'] for f in j['parquet_files'] if f['split'] == 'test']

    r = requests.get(urls[0])
    with open('datasets/zebra_grid_mode.parquet', 'wb') as f:
        f.write(r.content)

    r = requests.get(urls[1])
    with open('datasets/zebra_mc_mode.parquet', 'wb') as f:
        f.write(r.content)

    
def get_zebra_dataset(size = "2*2", random = False):
    """
    Return zebra grid_mode dataset as records list filtered by size, used local parquet file
    Currently dataset contains 40 items for each size from '2*2' to '6*6', total 1000 items
    Items ordered by id if random = False, shuffled if random = True
    """
    import duckdb
    import random
    
    if not os.path.exists('datasets/zebra_grid_mode.parquet'):
        download_zebra_dataset()

    con = duckdb.connect()
    data = con.sql(f'''SELECT id, puzzle, solution from 'datasets/zebra_grid_mode.parquet' where size='{size}' order by created_at, id''').df()
    res = data.to_dict(orient='records')
    if (random): random.shuffle(res)
    
    return res


# Based on ZeroEval, modified:
# - changed defaults and params
# - added reasonong_effort directly (was via model name with replacement after)
# - added support for @groq, @google, @custom
def openai_chat_request(
    model: str = None,
    prompt: str = None,
    messages: List[dict] = None,
    reasoning_effort: str = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 512,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    stop: List[str] = None,
    json_mode: bool = False,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        model (str): The model to use.
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages (use instead of prompt, not both).
        reasoning_effort (str, optional): high/medium/low/default/none, Defaults to None.
        temperature (float, optional): The temperature. Defaults to 0.6.
        top_p (float, optional): The top p. Defaults to 0.95.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 512.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
        json_mode (boolean, optional): TO use structured output, Defaults to False.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert (
        prompt is not None or messages is not None
    ), "Either prompt or messages should be provided."
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

    source = None

    # for version > 1.0
    if model.endswith("@groq"):
        assert (
            os.environ.get("GROQ_API_KEY") is not None
        ), "Please set GROQ_API_KEY in the environment variables."
        client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        model = model.replace("@groq", "")
        source = "groq"
    elif model.endswith("@google") or model.startswith('gemini-'):
        assert (
            os.environ.get("GOOGLE_API_KEY") is not None
        ), "Please set GOOGLE_API_KEY in the environment variables."
        client = OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = model.replace("@google", "")
        source = "google"
    elif model.endswith("@custom"):
        assert (
            os.environ.get("CUSTOM_API_KEY") is not None
        ), "Please set CUSTOM_API_KEY in the environment variables."
        assert (
            os.environ.get("CUSTOM_API_URL") is not None
        ), "Please set CUSTOM_API_URL in the environment variables."
        client = OpenAI(
            api_key=os.environ.get("CUSTOM_API_KEY"),
            base_url=os.environ.get("CUSTOM_API_URL"),
        )
        model = model.replace("@custom", "")
        source = "custom"
    elif "deepseek" in model:
        assert (
            os.environ.get("DEEPSEEK_API_KEY") is not None
        ), "Please set DEEPSEEK_API_KEY in the environment variables."
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
        source = "deepseek"
    elif model.endswith("@nvidia"):
        assert (
            os.environ.get("NVIDIA_API_KEY") is not None
        ), "Please set NVIDIA_API_KEY in the environment variables."
        client = OpenAI(
            api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
        )
        model = model.replace("@nvidia", "")
        source = "nvidia"
    elif model.endswith("@xai") or model.startswith("grok-"):
        assert (
            os.environ.get("XAI_API_KEY") is not None
        ), "Please set XAI_API_KEY in the environment variables."

        client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        model = model.replace("@xai", "")
        source = "xai"
    elif model.endswith("@anthropic") or model.startswith("claude-"):
        assert (
            os.environ.get("ANTHROPIC_API_KEY") is not None
        ), "Please set ANTHROPIC_API_KEY in the environment variables."

        client = OpenAI(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1/",
        )
        model = model.replace("@anthropic", "")
        source = "anthropic"
        # See about thinking https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if model.find('/') > 0:
            model = model.split("/")[-1]
        model = model.replace("@openai", "")
        source = "openai"

    thinking_budget = None # according to https://ai.google.dev/gemini-api/docs/openai#thinking
    if reasoning_effort is None:
        if model.endswith("-high"):
            reasoning_effort = "high"
            if source == "google": thinking_budget = 32768 if "-pro" in model else 24576
        elif model.endswith("-medium"):
            reasoning_effort = "medium"
            if source == "google": thinking_budget = 8192
        elif model.endswith("-low"):
            reasoning_effort = "low"
            if source == "google": thinking_budget = 512 if "-flash" in model else 128
        elif model.endswith("-none"):
            reasoning_effort = "none"
            if source == "google": thinking_budget = 0
        elif model.endswith("-default"):
            reasoning_effort = "default"
        if reasoning_effort is not None:
            model = model[:-len(reasoning_effort)-1]

    if source == "nvidia":
        # print(f"Requesting chat completion from OpenAI API with model {model}")
        # remove system message
        if messages[0]["role"] == "system":
            messages = messages[1:]
            # "detailed thinking on"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.001 if temperature == 0 else temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )
    elif source == "xai":
        if model.startswith("grok-3-mini") and reasoning_effort not in ['high', 'low']:
            raise ValueError("You should specify the reasoning effort for grok-3-mini 'high' or 'low'")
        assert model in ["grok-3-mini-beta", "grok-3-mini-fast-beta", "grok-2-1212", "grok-3-beta", "grok-3-fast-beta"], f"Model {model} is not supported"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    elif model.startswith("o1-") or model.startswith("o3-"):
        # print(f"Requesting chat completion from OpenAI API with model {model}")
        if messages[0]["role"] == "system":
            messages = messages[1:]
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"} if json_mode else None,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,  # new version
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            stop=stop,
            **kwargs,
        )
        #hidden_reasoning_tokens = (
        #    response.usage.completion_tokens_details.reasoning_tokens
        #)
    elif source == "google":
        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            #"thinking_budget": thinking_budget,
                            "include_thoughts": (reasoning_effort != "none")
                        }
                    }
                }
            }
            if thinking_budget is not None: # for automatic assignment
                kwargs["extra_body"]["extra_body"]["google"]["thinking_config"]["thinking_budget"] = thinking_budget
        if frequency_penalty != 0: # not supported by google
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0: # not supported by google
            kwargs["presence_penalty"] = presence_penalty
        if stop != None: # None not supported by google
            kwargs["stop"] = stop
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"} if json_mode else None,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,  # new version
            **kwargs,
        )
    else:
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if frequency_penalty != 0: # not supported by google
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0: # not supported by google
            kwargs["presence_penalty"] = presence_penalty
        if stop != None: # None not supported by google
            kwargs["stop"] = stop
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"} if json_mode else None,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,  # new version
            **kwargs,
        )

    print(f"Received response from OpenAI API with model {model}")
    print(response)
    contents = []
    thoughts = []
    for choice in response.choices:
        #print(choice)
        # Check if the response is valid
        if choice.finish_reason not in ["stop", "length"]:
            if "content_filter" in choice.finish_reason:
                raise ValueError(
                    f"OpenAI API response rejected due to content_filter policy."
                )
            else:
                raise ValueError(
                    f"OpenAI API response finish_reason: {choice.finish_reason}"
                )
        elif choice.message.content is None:
            raise ValueError(
                f"OpenAI API response have no content: {choice.finish_reason}"
            )

        thought, text = extract_think(choice.message.content)
        contents.append(text)

        if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content is not None:
            thoughts.append(thought + choice.message.reasoning_content)
        elif thought:
            thoughts.append(thought)

    #if o1_mode:
    #    return contents, hidden_reasoning_tokens, response.usage
    return contents, thoughts, response.usage.to_dict()


def extract_think(text):
    """
    Split text onto think|thought and remaining part
    """
    m = re.search(r'<(think|thought)>(.*)</(think|thought)>', text, re.DOTALL)
    if m:
        return m[0], text[:m.start()].strip() + text[m.end():].strip()
    return '', text.strip()


def extract_json(text):
    """
    Split text onto json and remaining part
    """
    m = re.search(r'\{(.*(?:.*\{.*\}.*)*)\}', text, re.DOTALL)
    if m:
        return m[0], text[:m.start()].strip() + text[m.end():].strip()
    return '', text.strip()


def generate_result_item(item, prompt, answer, success, thought, usage):
    """
    Generate result item as dictionary, used for one prompt iteration
    """
    return { 
        "id": item['id'],
        "success": success,
        "finished": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "solution": (get_zebra_grid_solution(item)),
        "answer": answer,
        "thought": thought,
        "usage": usage,
    }


# Based on https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
def normalize_filename(fn):
    """
    Sanitize filename, invalid chars replaced by '-', allowed only alpha-numeric and '@-_.()[] '
    """
    validchars = "@-_.()[] "
    out = ""
    for c in fn:
      if str.isalpha(c) or str.isdigit(c) or (c in validchars):
        out += c
      else:
        out += "-"
    return out    


def zebra_run(models, size, count, max_tokens):
    """
    Main zebra test function, doing all work:
    - load dataset of passed ~size~ and pick ~count~ random items
    - for each model in ~models~ query same picked items with ~max_tokens~ limit
    - for each model in ~models~ write details to json file
    - print final results as list 'model: success_rate'
    """
    lastmodel = None
    lastrun = None
    started = datetime.datetime.now()
    run_prefix = started.strftime('%Y%m%d_%H%M%S')
    dataset = get_zebra_dataset(size, random=True)[:count]
    models_summary = {}

    for model in models:
        print(f'Testing {count} items {size} on {model}')

        if lastmodel is not None and model.split('@')[-1] == lastmodel.split('@')[-1]:
            if lastrun is not None and (datetime.datetime.now() - lastrun) < datetime.timedelta(seconds=60):
                wait_seconds = 60 - (datetime.datetime.now() - lastrun).seconds
                print(f"Waiting {wait_seconds} seconds due to minimum 1 minute per model from one source...")
                time.sleep(wait_seconds)
            
        successed = 0
        dt = datetime.datetime.now()
        lastmodel = model
        lastrun = dt
        res_items = []
        total_tokens = 0

        for i in range(0,count):
            item = dataset[i]
            prompt = get_zebra_grid_prompt(item)
            solution_expected = get_zebra_grid_solution(item)
            thought = None
            usage = None

            print(item['id'])

            try:
                answers, thoughts, usage = openai_chat_request(model, prompt=prompt, max_tokens=max_tokens, json_mode=False)
                if usage is not None: total_tokens += usage['total_tokens']
                jsonstr, text = extract_json(answers[0])
                answer = json.loads(jsonstr)
                thought = thoughts[0] if len(thoughts) > 0 else None

                solution_answered = answer['solution']
                print('Expected:', solution_expected)
                print('Answered:', solution_answered)
                success = solution_expected == solution_answered
                if success: successed += 1
            except Exception as ex:
                print(ex)
                success = False
                if hasattr(ex, 'body'):
                    answer = ex.body
                else:
                    answer = str(ex)
            print(('FAILURE', 'SUCCESS')[success])

            res_items.append(generate_result_item(item, prompt, answer, success, thought, usage))

        results = {
            "model": model,
            "size": size,
            "max_tokens": max_tokens,
            "total_tokens": total_tokens,
            "started": dt.isoformat(),
            "duration": str(datetime.datetime.now()-dt),
            "count": count,
            "successed": successed,
            "rate": int(100 * successed / count),
            "items": res_items,
        }
        models_summary[model] = int(100 * successed / count)
        print(f'{model} passed {int(100 * successed / count)}% ({successed} of {count}) in {datetime.datetime.now()-dt}')

        try:
            # Write collected info as json
            filename = f"results/{run_prefix}_{normalize_filename(model)}_{size.replace('*','x')}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=4)
            print(f'Saved {filename}')
        except Exception as ex:
            print(f'FAILED to save {filename}')
            print(ex)

    print(f'Finished processing {len(models)} models * {count} tests in {datetime.datetime.now()-started}')

    summary = { 
        "duration": str(datetime.datetime.now()-started), 
        "size": size, 
        "count": count, 
        "max_tokens": max_tokens, 
        "models_summary": models_summary
    }
    filename = f"results/{run_prefix}_!results_{size.replace('*','x')}.json"
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
    print(f'Saved {filename}')

    return summary