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
    system_message: str
    content: str


app = FastAPI()


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


def call_lambda(chat):
    """
    """
    credentials = get_credentials()
    lambda_client = boto3.client(
        'lambda',
        region_name=REGION,
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretKey'],
        aws_session_token=credentials['SessionToken'],
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
                    "content": chat.system_message,
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
