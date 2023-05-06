#!/usr/bin/env python3

import argparse

import openai

MODELS = ["gpt-4", "gpt-4-32K", "gpt-3.5-turbo"]

parser = argparse.ArgumentParser(
    description="Select a model and load user prompt from a text file."
)
parser.add_argument(
    "prompt_file", type=str, help="Path to the text file containing the user prompt"
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-3.5-turbo",
    help=f"Choose the model ({MODELS}). Default: gpt-3.5-turbo",
)

args = parser.parse_args()
assert args.model in MODELS

with open(".openai.secret", "r") as f:
    contents = f.read().strip()
api_key = contents.split("\n")[0].split("=")[1].strip()
org_id = contents.split("\n")[1].split("=")[1].strip()

openai.organization = org_id
openai.api_key = api_key

with open(args.prompt_file, "r") as f:
    user_content = f.read().strip()

msg = [
    {
        "role": "system",
        "content": "You are Chris Lattner, an expert C++ programmer and compiler engineer who is deeply familiar with LLVM.",
    },
    {
        "role": "user",
        "content": user_content,
    },
]

response = openai.ChatCompletion.create(model=args.model, messages=msg, n=1)
print(response.choices[0].message.content)
