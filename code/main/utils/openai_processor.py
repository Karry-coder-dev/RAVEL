import base64
import datetime
import os
import sys
import uuid
from typing import Dict, Any

import openai
import httpx
import torch
import re
import json
from easydict import EasyDict
import time

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def format_request(custom_id, system_prompt: str, user_prompt: str, image_path=None,
                   end_point='/v1/chat/completions',
                   temperature=None,
                   top_p=None,
                   max_tokens=2048,
                   model="gpt-4o-mini"):
    user_content = []
    if image_path is not None:
        if isinstance(image_path, list):
            base64_images = [encode_image(x) for x in image_path]
            image_content = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
                "modalities": "multi-images",
            } for base64_image in base64_images]
            user_content.extend(image_content)
        else:
            base64_image = encode_image(image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
            user_content.append(image_content)
    user_content.append({
        "type": "text",
        "text": user_prompt
    })
    payload = {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": end_point,
        "body": {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                # {
                #     "role": "system",
                #     "content": system_prompt,
                # },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    }
    if temperature is None and top_p is None:
        raise ValueError("Please provide temperature or top_p.")
    if temperature is not None:
        payload["body"]["temperature"] = temperature
    if top_p is not None:
        payload["body"]["top_p"] = top_p
    # print(payload)
    return payload


def chat_completion_to_dict(chat_completion: ChatCompletion) -> Dict[str, Any]:
    return {
        "id": chat_completion.id,
        "choices": {
            "finish_reason": chat_completion.choices[0].finish_reason,
            "index": chat_completion.choices[0].index,
            "logprobs": chat_completion.choices[0].logprobs,
            "message": {
                "content": chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role
            },
            "matched_stop": chat_completion.choices[0].matched_stop
        },
        "created": chat_completion.created,
        "model": chat_completion.model,
        "object": chat_completion.object,
        "usage": {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens
        }
    }


class OpenAIBatchProcessor:
    def __init__(self, base_url="http://127.0.0.1:30000/v1", api_key="EMPTY", log_path='./'):
        if base_url is None:
            self.client = OpenAI(api_key=api_key)
        else:
            # SGLang runs on loopback.  The server's proxy must not intercept
            # those local requests.
            local_http = httpx.Client(trust_env=False)
            self.client = openai.Client(base_url=base_url, api_key=api_key,
                                      http_client=local_http)
        self.log_path = os.path.join(log_path, 'SGLang_storage')
        os.makedirs(self.log_path, exist_ok=True)

    def process_batch(self, input_file_path, endpoint, interval: int = 10,
                      n_retries: int = 2,
                      verbose=False):
        for attempt_idx in range(n_retries):
            try:
                # Upload the input file
                with open(input_file_path, "rb") as file:
                    if sum(1 for line in file) == 1:
                        file.seek(0)
                        result = [self.process_single(file.readlines())]
                        return result
                    file.seek(0)
                    uploaded_file = self.client.files.create(file=file, purpose="batch")

                # Create the batch job
                batch_job = self.client.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint=endpoint,
                    completion_window="24h",
                )

                # Monitor the batch job status
                start_time = time.time()
                try_time = 0
                while batch_job.status not in ["completed", "failed", "cancelled"]:
                    try_time += 1
                    time.sleep(interval)  # Wait for few seconds before checking the status again
                    if verbose and try_time % 1 == 0:
                        print(f"Batch job status: {batch_job.status}...trying again every {interval} seconds...")
                    batch_job = self.client.batches.retrieve(batch_job.id)

                # Check the batch job status and errors
                if batch_job.status == "failed":
                    print(f"Batch job failed with status: {batch_job.status}")
                    print(f"Batch job errors: {batch_job.errors}")
                    return None

                # If the batch job is completed, process the results
                if batch_job.status == "completed":
                    cur_time = int(time.time() - start_time)
                    # print result of batch job
                    if verbose:
                        print("batch finished using {} sec:".format(cur_time), batch_job.request_counts)

                    result_file_id = batch_job.output_file_id
                    # Retrieve the file content from the server
                    file_response = self.client.files.content(result_file_id)
                    result_content = file_response.read()  # Read the content of the file

                    # Save the content to a local file
                    result_file_name = os.path.join(self.log_path, f"batch_job_chat_results_{uuid.uuid4()}.jsonl")
                    with open(result_file_name, "wb") as file:
                        file.write(result_content)  # Write the binary content to the file
                    # Load data from the saved JSONL file
                    results = []
                    with open(result_file_name, "r", encoding="utf-8") as file:
                        for line in file:
                            json_object = json.loads(
                                line.strip()
                            )  # Parse each line as a JSON object
                            results.append(json_object)

                    return results
                else:
                    print(f"Batch job failed with status: {batch_job.status}")
                    return None
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch batch result. Exception:", e)
                time.sleep(2)

    def process_single(self, request):
        request = json.loads(request[0].decode('utf-8').strip("[]\n"))
        custom_id = request["custom_id"]
        response = self.client.chat.completions.create(**request["body"])
        response = chat_completion_to_dict(response)
        response = {
            "id": "0",
            "custom_id": custom_id,
            "response": {
                "body": response
            }
        }
        return response

    def submit_batch(self, input_file_path, endpoint):
        for attempt_idx in range(3):
            try:
                with open(input_file_path, "rb") as file:
                    uploaded_file = self.client.files.create(file=file, purpose="batch")

                # Create the batch job
                batch_job = self.client.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint=endpoint,
                    completion_window="24h",
                )
                return batch_job
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to submit the batch. Exception:", e)
                time.sleep(3)

    def retrieve_batch(self, batch_job, interval: int = 5, verbose=False):
        for attempt_idx in range(3):
            try:
                while batch_job.status not in ["completed", "failed", "cancelled", "in_progress", "finalizing"]:
                    batch_job = self.client.batches.retrieve(batch_job.id)
                    time.sleep(interval)
                if verbose:
                    for i in tqdm(range(batch_job.request_counts.total), desc="Processing batch", file=sys.stdout,
                                  ncols=100):
                        while batch_job.request_counts.completed <= i:
                            if batch_job.status in ["completed", "failed", "cancelled"]:
                                break
                            batch_job = self.client.batches.retrieve(batch_job.id)
                            # bar(batch_job.request_counts.completed - bar.current)
                            time.sleep(interval if batch_job.request_counts.completed != 0 else interval * 5)
                            # Wait for few seconds before checking the status again
                        time.sleep(0.1)
                    print("Batch finalizing...")
                    while batch_job.status not in ["completed", "failed", "cancelled"]:
                        batch_job = self.client.batches.retrieve(batch_job.id)
                        time.sleep(interval)
                else:
                    while batch_job.status not in ["completed", "failed", "cancelled"]:
                        # progress = f"completed/total: {batch_job.request_counts.completed}/{batch_job.request_counts.total}"
                        # print(f"Batch job {batch_job.status} {progress}...trying again after {interval} seconds...")
                        batch_job = self.client.batches.retrieve(batch_job.id)
                        time.sleep(interval)  # Wait for few seconds before checking the status again

                # Check the batch job status and errors
                if batch_job.status == "failed":
                    print(f"Batch job failed with status: {batch_job.status}")
                    print(f"Batch job errors: {batch_job.errors}")
                    return None

                # If the batch job is completed, process the results
                if batch_job.status == "completed":
                    # print result of batch job
                    if verbose:
                        print(f"Batch {batch_job.id[-5:]} finished {batch_job.request_counts}. Downloading...")

                    result_file_id = batch_job.output_file_id
                    # Retrieve the file content from the server
                    file_response = self.client.files.content(result_file_id)
                    result_content = file_response.read()  # Read the content of the file

                    # Save the content to a local file
                    result_file_name = os.path.join(self.log_path, f"batch_job_chat_results_{uuid.uuid4()}.jsonl")
                    with open(result_file_name, "wb") as file:
                        file.write(result_content)  # Write the binary content to the file
                    # Load data from the saved JSONL file
                    results = []
                    with open(result_file_name, "r", encoding="utf-8") as file:
                        for line in file:
                            json_object = json.loads(
                                line.strip()
                            )  # Parse each line as a JSON object
                            results.append(json_object)
                    if verbose:
                        print(f"Batch {batch_job.id[-5:]} retrieved.")
                    return results
                else:
                    print(f"Batch job failed with status: {batch_job.status}")
                    return None
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"Failed to fetch batch result. Exception:", e)
                time.sleep(3)

            exit(0)


if __name__ == "__main__":
    client = OpenAIBatchProcessor("192.168.49.59:10000/v1")
