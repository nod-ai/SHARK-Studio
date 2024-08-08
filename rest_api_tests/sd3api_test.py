import requests
from pydantic import BaseModel, Field
import json


def view_json_file(file_path):
    content = ""
    with open(file_path, "r") as fopen:
        content = fopen.read()
    return content


# Define the URL of the REST API endpoint
api_url = "http://127.0.0.1:8080/sdapi/v1/txt2img/"  # Replace with your actual API URL


class GenerationInputData(BaseModel):
    prompt: list = [""]
    negative_prompt: list = [""]
    hf_model_id: str | None = None
    height: int = Field(default=512, ge=128, le=1024, multiple_of=8)
    width: int = Field(default=512, ge=128, le=1024, multiple_of=8)
    sampler_name: str = "EulerDiscrete"
    cfg_scale: float = Field(default=7.5, ge=1)
    steps: int = Field(default=20, ge=1, le=100)
    seed: int = Field(default=-1)
    n_iter: int = Field(default=1)
    config: dict = None


# Create an instance of GenerationInputData with example arguments
data = GenerationInputData(
    prompt=[
        "A phoenix made of diamond, black background, dream sequence, rising from coals"
    ],
    negative_prompt=[
        "cropped, cartoon, lowres, low quality, black and white, bad scan, pixelated"
    ],
    hf_model_id="shark_sd3.py",
    height=512,
    width=512,
    sampler_name="EulerDiscrete",
    cfg_scale=7.5,
    steps=20,
    seed=-1,
    n_iter=1,
    config=json.loads(view_json_file("../configs/sd3_phoenix_npu.json")),
)

# Convert the data to a dictionary
data_dict = data.dict()

# Optional: Define headers if needed (e.g., for authentication)
headers = {
    "User-Agent": "PythonTest",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
}


def test_post_request(url, data, headers=None):
    try:
        # Send a POST request to the API endpoint
        response = requests.post(url, json=data, headers=headers)

        # Print the status code and response content
        print(f"Status Code: {response.status_code}")
        print("Response Content:")
        # print(response.json())  # Print the JSON response

    except requests.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")


# Run the test
test_post_request(api_url, data_dict, headers)
