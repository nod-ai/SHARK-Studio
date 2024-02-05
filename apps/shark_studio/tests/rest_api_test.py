import requests
from PIL import Image
import base64
from io import BytesIO
import json


def llm_chat_test(verbose=False):
    # Define values here
    prompt = "What is the significance of the number 42?"

    url = "http://127.0.0.1:8080/v1/chat/completions"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "model": "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        "messages": [
            {
                "role": "",
                "content": prompt,
            }
        ],
        "device": "vulkan://0",
        "max_tokens": 4096,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)
    res_dict = json.loads(res.content.decode("utf-8"))
    print(f"[chat] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res_dict['choices'][0]['message']['content']}\n")


if __name__ == "__main__":

    # "Exercises the Stable Diffusion REST API of Shark. Make sure "
    # "Shark is running in API mode on 127.0.0.1:8080 before running"
    # "this script."

    llm_chat_test(verbose=True)
