import os

import requests
import json
api_key = os.environ.get("MOONSHOT_API_KEY")

def moonshot_count_tokens(messages,model="moonshot-v1-8k"):
    url = 'https://api.moonshot.cn/v1/tokenizers/estimate-token-count'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": messages
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    true_response = response.json()
    if "error" not in true_response.keys():
        return true_response["data"]['total_tokens']

    return 0