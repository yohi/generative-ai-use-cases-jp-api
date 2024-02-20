import boto3
import warrant
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

USERNAME = os.environ['USERNAME']
PASSWORD = os.environ['PASSWORD']
REGION = os.environ['REGION']
USER_POOL_ID = os.environ['USER_POOL_ID']
CLIENT_ID = os.environ['CLIENT_ID']
IDENTITY_POOL_ID = os.environ['IDENTITY_POOL_ID']
LAMBDA_FUNCTION_NAME = os.environ['LAMBDA_FUNCTION_NAME']


class Chat(BaseModel):
    content: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def call_lambda(chat):
    client = boto3.client('cognito-idp', region_name=REGION)
    srp = warrant.aws_srp.AWSSRP(
        username=USERNAME,
        password=PASSWORD,
        pool_id=USER_POOL_ID,
        client_id=CLIENT_ID,
        client=client
    )

    srp_a = srp.get_auth_params()['SRP_A']

    response = client.initiate_auth(
        AuthFlow='USER_SRP_AUTH',
        AuthParameters={
            'USERNAME': USERNAME,
            'SRP_A': srp_a
        },
        ClientId=CLIENT_ID,
    )
    # assert(response['ChallengeName'] == 'PASSWORD_VERIFIER')
    challenge_response = srp.process_challenge(response['ChallengeParameters'])
    response = client.respond_to_auth_challenge(
        ClientId=CLIENT_ID,
        ChallengeName='PASSWORD_VERIFIER',
        ChallengeResponses=challenge_response,
    )

    id_token = response['AuthenticationResult']['IdToken']
    cognito_identity_client = boto3.client(
        'cognito-identity',
        region_name=REGION,
    )
    response = cognito_identity_client.get_id(
        IdentityPoolId=IDENTITY_POOL_ID,
        Logins={
            'cognito-idp.us-east-1.amazonaws.com/' + USER_POOL_ID: id_token,
        }
    )

    identity_id = response['IdentityId']
    response = cognito_identity_client.get_credentials_for_identity(
        IdentityId=identity_id,
        Logins={
            'cognito-idp.us-east-1.amazonaws.com/' + USER_POOL_ID: id_token,
        }
    )

    lambda_client = boto3.client(
        'lambda',
        region_name=REGION,
        aws_access_key_id=response['Credentials']['AccessKeyId'],
        aws_secret_access_key=response['Credentials']['SecretKey'],
        aws_session_token=response['Credentials']['SessionToken'],
    )

    response = lambda_client.invoke_with_response_stream(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "model": {
                "modelId": "anthropic.claude-v2:1",
                "type": "bedrock"
            },
            "messages": [
                {
                    "role": "system",
                    "content": "あなたはチャットでユーザを支援するAIアシスタントです。"
                },
                {
                    "role": "user",
                    "content": chat.content,
                }
            ]
        }),
    )

    for event in response['EventStream']:
        if 'PayloadChunk' in event:
            yield event['PayloadChunk']['Payload'].decode('utf-8')
        elif 'InvokeComplete' in event:
            pass


@app.post("/chat")
async def chat(chat: Chat):
    return StreamingResponse(call_lambda(chat))
