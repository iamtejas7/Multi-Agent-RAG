import boto3
from langchain_aws import ChatBedrock
from config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, BEDROCK_MODEL_ID


def _get_boto3_session() -> boto3.Session:
    kwargs = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return boto3.Session(**kwargs)


def get_llm(temperature: float = 0.1) -> ChatBedrock:
    session = _get_boto3_session()
    client = session.client("bedrock-runtime")
    return ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        client=client,
        model_kwargs={"temperature": temperature, "max_tokens": 2048},
    )
