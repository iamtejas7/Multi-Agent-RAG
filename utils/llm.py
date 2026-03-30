import boto3
from langchain_aws import ChatBedrock
from config import AWS_REGION, BEDROCK_MODEL_ID


def get_llm(temperature: float = 0.1) -> ChatBedrock:
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        client=client,
        model_kwargs={"temperature": temperature, "max_tokens": 2048},
    )
