import zebra_utils

# To define API keys in code fix config.py and uncomment following 2 lines
#import config
#config.update_environ()

# TODO: print usage help on no arguments
# TODO: extract params from command line

filename = 'models.txt'
size = '2*2'
count = 10
max_tokens = 1024

with open(filename, 'r') as f:
    models = [x.strip() for x in f.readlines() if x.strip()]

print(models)

results = zebra_utils.zebra_run(models, size, count, max_tokens)
print(results)
