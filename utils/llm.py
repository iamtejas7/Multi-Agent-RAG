from langchain_aws import ChatBedrock
from config import AWS_REGION, BEDROCK_MODEL_ID


def get_llm(temperature: float = 0.1) -> ChatBedrock:
    return ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": temperature, "max_tokens": 2048},
    )
