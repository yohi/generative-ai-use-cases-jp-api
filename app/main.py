import boto3
import warrant
import json
import os
import logging
from pprint import pprint

from fastapi import FastAPI, Request, status
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from litellm import completion

USERNAME = os.environ['USERNAME']
PASSWORD = os.environ['PASSWORD']
REGION = os.environ['REGION']
USER_POOL_ID = os.environ['USER_POOL_ID']
CLIENT_ID = os.environ['CLIENT_ID']
IDENTITY_POOL_ID = os.environ['IDENTITY_POOL_ID']
LAMBDA_FUNCTION_NAME = os.environ['LAMBDA_FUNCTION_NAME']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Chat(BaseModel):
    content: str


class OpenAI(BaseModel):
    messages: list
    model: str
    frequency_penelty: float | None = 0.0
    logit_bias: dict | None = None
    top_logrobs: int | None = None
    max_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list | None = None
    stream: bool | None = False
    top_p: float | None = 1.0
    tools: list | None = None
    tools_choice: str | None = None


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def get_id_token():
    """
    """
    cognito_idp = boto3.client('cognito-idp', region_name=REGION)
    srp = warrant.aws_srp.AWSSRP(
        username=USERNAME,
        password=PASSWORD,
        pool_id=USER_POOL_ID,
        client_id=CLIENT_ID,
        client=cognito_idp,
    )
    srp_a = srp.get_auth_params()['SRP_A']
    response = cognito_idp.initiate_auth(
        AuthFlow='USER_SRP_AUTH',
        AuthParameters={
            'USERNAME': USERNAME,
            'SRP_A': srp_a
        },
        ClientId=CLIENT_ID,
    )
    assert response['ChallengeName'] == 'PASSWORD_VERIFIER'
    challenge_response = srp.process_challenge(response['ChallengeParameters'])
    response = cognito_idp.respond_to_auth_challenge(
        ClientId=CLIENT_ID,
        ChallengeName='PASSWORD_VERIFIER',
        ChallengeResponses=challenge_response,
    )

    return response['AuthenticationResult']['IdToken']


def get_credentials():
    """
    """
    id_token = get_id_token()
    cognito_identity = boto3.client(
        'cognito-identity',
        region_name=REGION,
    )
    response = cognito_identity.get_id(
        IdentityPoolId=IDENTITY_POOL_ID,
        Logins={
            f'cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}': id_token,
        }
    )
    identity_id = response['IdentityId']
    response = cognito_identity.get_credentials_for_identity(
        IdentityId=identity_id,
        Logins={
            f'cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}': id_token,
        }
    )
    return response['Credentials']


def call_lambda(openai):
    """
    """
    print('call_lambda')
    credentials = get_credentials()
    print(credentials)

    lambda_client = boto3.client(
        'lambda',
        region_name=REGION,
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretKey'],
        aws_session_token=credentials['SessionToken'],
    )
    print(lambda_client)

    response = lambda_client.invoke_with_response_stream(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "model": {
                "modelId": "anthropic.claude-v2:1",
                "type": "bedrock"
            },
            # "messages": [
            #     {
            #         "role": "system",
            #         "content": "あなたはチャットでユーザを支援するAIアシスタントです。"
            #     },
            #     {
            #         "role": "user",
            #         "content": openai.content,
            #     }
            # ]
            "messages": openai.messages,
        }),
    )
    print(response)

    for event in response['EventStream']:
        if 'PayloadChunk' in event:
            yield event['PayloadChunk']['Payload'].decode('utf-8')
        elif 'InvokeComplete' in event:
            pass


@app.post("/chat")
async def chat(chat: Chat):
    return StreamingResponse(call_lambda(chat))


# @app.post('/v1/chat/completions')
# def openai(openai: OpenAI):
#     if True:
#         os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_AI_API_KEY']
#         response = completion(
#             model='gemini/gemini-pro',
#             messages=openai.messages,
#         )
# 
#         print(response.json())
#         return JSONResponse(content=response.json())
#     else:
# 
#         credentials = get_credentials()
#         # bedrock = boto3.client(
#         #             service_name="bedrock-runtime",
#         #             region_name="us-east-1",
#         #             aws_access_key_id=credentials['AccessKeyId'],
#         #             aws_secret_access_key=credentials['SecretKey'],
#         #             aws_session_token=credentials['SessionToken'],
#         # )
# 
#         # response = completion(
#         #     model="bedrock/anthropic.claude-instant-v1",
#         #     messages=[{ "content": "Hello, how are you?","role": "user"}],
#         #     aws_bedrock_client=bedrock,
#         # )
#         # print(response)
# 
#         # return {'HELLO': 'WORLD'}
# 
#         credentials = get_credentials()
#         print(credentials)
# 
#         lambda_client = boto3.client(
#             'lambda',
#             region_name=REGION,
#             aws_access_key_id=credentials['AccessKeyId'],
#             aws_secret_access_key=credentials['SecretKey'],
#             aws_session_token=credentials['SessionToken'],
#         )
#         print(lambda_client)
# 
#         response = lambda_client.invoke(
#             FunctionName=LAMBDA_FUNCTION_NAME,
#             InvocationType='RequestResponse',
#             Payload=json.dumps({
#                 "model": {
#                     "modelId": "anthropic.claude-v2:1",
#                     "type": "bedrock"
#                 },
#                 # "messages": [
#                 #     {
#                 #         "role": "system",
#                 #         "content": "あなたはチャットでユーザを支援するAIアシスタントです。"
#                 #     },
#                 #     {
#                 #         "role": "user",
#                 #         "content": openai.content,
#                 #     }
#                 # ]
#                 "messages": openai.messages,
#             }),
#         )
#         print(response)
#         # {'ResponseMetadata': {'RequestId': '07ee080c-928f-4ba8-9092-332952d98178', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Mon, 04 Mar 2024 07:35:59 GMT', 'content-type': 'application/octet-stream', 'content-length': '548', 'connection': 'keep-alive', 'x-amzn-requestid': '07ee080c-928f-4ba8-9092-332952d98178', 'x-amzn-remapped-content-length': '0', 'x-amz-executed-version': '$LATEST', 'x-amzn-trace-id': 'root=1-65e579d9-605692c83ab029dc5324f871;parent=6d88e0cbba1c93af;sampled=0;lineage=d456cf1d:0'}, 'RetryAttempts': 0}, 'StatusCode': 200, 'ExecutedVersion': '$LATEST', 'Payload': <botocore.response.StreamingBody object at 0x7a60c1f75a20>}
# 
# 
#         for event in response['EventStream']:
#             if 'PayloadChunk' in event:
#                 yield event['PayloadChunk']['Payload'].decode('utf-8')
#             elif 'InvokeComplete' in event:
#                 pass
# 
#         return {'HELLO': 'WORLD'}
# 
# 
@app.post('/v1/chat/completions')
async def openai(openai: OpenAI):
    print('request:')
    pprint(json.loads(openai.model_dump_json()))
    if True:
        os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_AI_API_KEY']
        response = completion(
            model='gemini/gemini-pro',
            messages=openai.messages,
        )
        """
        {
            'id': 'chatcmpl-2e2e8752-1808-49de-9c18-836a3d2ad5a7',
            'choices': [
                {
                    'finish_reason': 'stop',
                    'index': 1,
                    'message': {
                        'content': 'Are you a supporter of Hokkaido Consadole Sapporo?',
                        'role': 'assistant'
                    }
                }
            ],
            'created': 1709538223,
            'model': 'gemini/gemini-pro',
            'object': 'chat.completion',
            'system_fingerprint': None,
            'usage': {
                'prompt_tokens': 29,
                'completion_tokens': 16,
                'total_tokens': 45
            }
        }
        """
        ret = response.json()
    else:
        credentials = get_credentials()

        lambda_client = boto3.client(
            'lambda',
            region_name=REGION,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretKey'],
            aws_session_token=credentials['SessionToken'],
        )

        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                "model": {
                    "modelId": "anthropic.claude-v2:1",
                    "type": "bedrock"
                },
                # "messages": [
                #     {
                #         "role": "system",
                #         "content": "あなたはチャットでユーザを支援するAIアシスタントです。"
                #     },
                #     {
                #         "role": "user",
                #         "content": openai.content,
                #     }
                # ]
                "messages": openai.messages,
            }),
        )
        ret = {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'index': 1,
                    'message': {
                        'content': response['Payload'].read().decode('utf-8'),
                        'role': 'assistant'
                    }
                }
            ],
            # 'created': 1709544282,
            # 'id': response['ResponseMetadata']['RequestId'],
            # 'model': 'anthropic.claude-v2:1',
            # 'object': 'chat.completion',
            # 'system_fingerprint': None,
            # 'usage': {
            #     'completion_tokens': 8,
            #     'prompt_tokens': 13,
            #     'total_tokens': 21
            # }
        }
        """
        {
            'ResponseMetadata': {
                'RequestId': '07ee080c-928f-4ba8-9092-332952d98178',
                'HTTPStatusCode': 200,
                'HTTPHeaders': {
                    'date': 'Mon, 04 Mar 2024 07:35:59 GMT',
                    'content-type': 'application/octet-stream',
                    'content-length': '548',
                    'connection': 'keep-alive',
                    'x-amzn-requestid': '07ee080c-928f-4ba8-9092-332952d98178',
                    'x-amzn-remapped-content-length': '0',
                    'x-amz-executed-version': '$LATEST',
                    'x-amzn-trace-id': 'root=1-65e579d9-605692c83ab029dc5324f871;parent=6d88e0cbba1c93af;sampled=0;lineage=d456cf1d:0'
                    },
                'RetryAttempts': 0
            },
            'StatusCode': 200,
            'ExecutedVersion': '$LATEST',
            'Payload': <botocore.response.StreamingBody object at 0x7a60c1f75a20>
        }
        """
        # return JSONResponse(content={'hello': 'world'})
    print('response:')
    pprint(ret)
    return JSONResponse(content=ret)
