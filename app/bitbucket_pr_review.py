"""
https://github.com/aws-samples/generative-ai-use-cases-jp/actions/runs/7004938177/workflow
https://zenn.dev/t_dai
https://zenn.dev/tadyjp/scraps/a7510f838edf8c
"""

from atlassian.bitbucket import Cloud
import requests

BITBUCKET_URL = ''
BITBUCKET_USERNAME = ''
BITBUCKET_TOKEN = ''
BITBUCKET_REPO_SLUG = ''
BITBUCKET_PR_ID = 0

cloud = Cloud(
    username=BITBUCKET_USERNAME,
    # password=BITBUCKET_PASSWORD,
    token=BITBUCKET_TOKEN,
)

repository = cloud.repositories.get('xxx', 'yyy')
pr = repository.pullrequests.get(BITBUCKET_PR_ID)

pr_description = pr.description
pr_diff = pr.diff()


prompt = f"""
Please review the following pull request:

Description:
{pr_description}

Diff:
{pr_diff}

Provide a detailed technical review focusing on best practices, potential bugs or issues, security concerns, etc.
"""

payload = {
    'content': prompt,
}

print(payload)

response = requests.post('http://localhost:56721/chat', json=payload)

print(response)

print(response.content.decode('utf-8'))
