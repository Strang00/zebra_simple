import zebra_utils
import sys
import config
config.update_environ()

filename = None
size = None
items = None
count = None
max_tokens = 1024
models = []

for i in range(1, len(sys.argv), 1):
    if (sys.argv[i] == '-f' and i+1 < len(sys.argv)):
        filename = sys.argv[i+1]        
        with open(filename, 'r') as f:
            models = [x.strip() for x in f.readlines() if x.strip()]
    elif (sys.argv[i] == '-m' and i+1 < len(sys.argv)):
        models = [x.strip() for x in sys.argv[i+1].split(',') if x.strip()]
        filename = None
    elif (sys.argv[i] == '-s' and i+1 < len(sys.argv)):
        size = sys.argv[i+1]
    elif (sys.argv[i] == '-i' and i+1 < len(sys.argv)):
        items = [int(x) for x in sys.argv[i+1].split(',') if int(x) in range(0,40)]
        count = len(items)
    elif (sys.argv[i] == '-r' and i+1 < len(sys.argv)):
        count = int(sys.argv[i+1])
        items = None
    elif (sys.argv[i] == '-t' and i+1 < len(sys.argv)):
        max_tokens = int(sys.argv[i+1])

if len(sys.argv) == 1 or size is None or count is None:
    print(f'Zebra Simple is simple tool to lest LLM models on WildEval/ZebraLogic dataset')
    print(f'Usage: python -u {sys.argv[0]} [OPTIONS]')
    print( 'Options:')
    print( '       -f FILENAME   - file with models list, one model per line')
    print( '       -m MODELS     - comma-separated list of models, incompatible with -f')
    print( '       -s SIZE       - ZebraLogic puzzle size from "2*2" to "6*6"')
    print( '       -i ITEMS      - comma-separated list of item numbers 0-39 in dataset')
    print( '       -r COUNT      - count of random puzzles to test, incompatible with -i')
    print( '       -t MAX_TOKENS - limit of used tokens per each completion, default 1024')
    print( 'Models:')
    print( '       Use format NAME[-REASONING_EFFORT][@SOURCE]')
    print( '       NAME is model name from selected SOURCE, links and examples in README')
    print( '       REASONING_EFFORT [opt] is none, default, low, medium, high')
    print( '       SOURCE [opt] is google, groq, nvidia, openai, xai, deepseek, anthropic')
    print( 'Examples:')
    print(f'       python -u {sys.argv[0]} -f models.txt -s 2*2 -r 10')
    print(f'       python -u {sys.argv[0]} -m llama-3.3-70b-versatile@groq -s 3*3 -i 0 -t 4096')
    print("")
    sys.exit(0)

if len(models) == 0:
   print("No models passed")
   sys.exit(1)

if count is None or count <= 0:
   print("Invalid count or items passed")
   sys.exit(1)

if items is None:
    print(f'Testing ZebraLogic {size} {count} items with max_tokens={max_tokens} using models:', models)
    dataset = zebra_utils.get_zebra_dataset(size, randomize=True)[:count]
else:
    print(f'Testing ZebraLogic {size} items {items} with max_tokens={max_tokens} using models:', models)
    dataset = zebra_utils.get_zebra_dataset(size, randomize=False)
    dataset = [dataset[i] for i in items]

zebra_utils.zebra_run(dataset, models, size, max_tokens)
