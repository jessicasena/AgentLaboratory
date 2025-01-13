import os, json
import tiktoken
import time
from openai import AzureOpenAI

TOKENS_IN = dict()
TOKENS_OUT = dict()
encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, tries=5, timeout=5.0, temp=None, print_cost=True, openai_api_key=None):
    """
    Query the Azure OpenAI model with given prompt and system message
    """
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not azure_api_key or not azure_endpoint:
        raise Exception("Azure OpenAI credentials not found in environment variables")

    for _ in range(tries):
        try:
            client = AzureOpenAI(
            api_key="09c11676e8dc489abcadeed70ea91e4d",  
            api_version="2024-02-01",
            azure_endpoint = "https://clinicalnotes-embed-test-001.openai.azure.com/"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            deployment_name='ehrsummary'

            completion = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.0
            )

            answer = completion.choices[0].message.content

            if print_cost:
                encoding = tiktoken.encoding_for_model("gpt-4")
                tokens_in = len(encoding.encode(system_prompt + prompt))
                tokens_out = len(encoding.encode(answer))
                
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += tokens_in
                TOKENS_OUT[model_str] += tokens_out

                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            
            return answer

        except Exception as e:
            print("Inference Exception:", e)
            if _ == tries - 1:
                raise e
            time.sleep(timeout)
    
    raise Exception("Max retries: timeout")