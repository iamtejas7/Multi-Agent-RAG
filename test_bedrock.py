"""Quick test to verify AWS Bedrock access for both LLM and Embedding models."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

import boto3

REGION = os.getenv("AWS_REGION", "us-east-2")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

session_kwargs = {"region_name": REGION}
if ACCESS_KEY and SECRET_KEY:
    session_kwargs["aws_access_key_id"] = ACCESS_KEY
    session_kwargs["aws_secret_access_key"] = SECRET_KEY

session = boto3.Session(**session_kwargs)
client = session.client("bedrock-runtime")

print(f"Region: {REGION}")
print(f"Identity: {session.client('sts').get_caller_identity()['Arn']}")
print()

# Test 1: Titan Embedding
print(f"[1] Testing Titan Embedding ({EMBEDDING_MODEL})...")
try:
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": "test embedding"}),
    )
    result = json.loads(response["body"].read())
    dim = len(result["embedding"])
    print(f"    SUCCESS - Embedding dimension: {dim}")
except Exception as e:
    print(f"    FAILED - {e}")

print()

# Test 2: Claude LLM
print(f"[2] Testing LLM ({LLM_MODEL})...")
try:
    response = client.invoke_model(
        modelId=LLM_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        }),
    )
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]
    print(f"    SUCCESS - Response: {text}")
except Exception as e:
    print(f"    FAILED - {e}")
