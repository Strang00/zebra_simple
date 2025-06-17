import os

GROQ_API_KEY = ""
GOOGLE_API_KEY = ""
NVIDIA_API_KEY = ""
DEEPSEEK_API_KEY = ""
XAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
OPENAI_API_KEY = ""
CUSTOM_API_KEY = GROQ_API_KEY
CUSTOM_API_URL = "https://api.groq.com/openai/v1"

def update_environ():
    if os.environ.get("GROQ_API_KEY") == None: os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    if os.environ.get("GOOGLE_API_KEY") == None: os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    if os.environ.get("NVIDIA_API_KEY") == None: os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    if os.environ.get("DEEPSEEK_API_KEY") == None: os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    if os.environ.get("XAI_API_KEY") == None: os.environ["XAI_API_KEY"] = XAI_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY") == None: os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    if os.environ.get("OPENAI_API_KEY") == None: os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if os.environ.get("CUSTOM_API_KEY") == None: os.environ["CUSTOM_API_KEY"] = CUSTOM_API_KEY
    if os.environ.get("CUSTOM_API_URL") == None: os.environ["CUSTOM_API_URL"] = CUSTOM_API_URL
    #os.environ["PYTHONIOENCODING"] = "utf-8"
    #os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
    #os.environ["PYTHONUTF8"] = "1"
    #os.system('chcp 65001')