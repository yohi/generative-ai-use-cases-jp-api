"""

cf.)
https://github.com/aws-samples/generative-ai-use-cases-jp/actions/runs/7004938177/workflow
https://zenn.dev/t_dai
https://zenn.dev/tadyjp/scraps/a7510f838edf8c
"""
import argparse
from atlassian.bitbucket import Cloud
import os
import logging
from dotenv import load_dotenv
from litellm import completion
from prompt import CodeRabbitPrompt
from pprint import pprint


dir_path = os.path.dirname(os.path.abspath("__file__"))
dotenv_path = os.path.join(dir_path, '.env')
load_dotenv(dotenv_path, verbose=True)

AI_MODEL = 'gemini/gemini-pro'

prompt = CodeRabbitPrompt()

level = logging.ERROR

# --------------------------------
# 1.loggerの設定
# --------------------------------
# loggerオブジェクトの宣言
logger = logging.getLogger(__name__)

# loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
logger.setLevel(level)

# --------------------------------
# 2.handlerの設定
# --------------------------------
# handlerの生成
stream_handler = logging.StreamHandler()

# handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
stream_handler.setLevel(level)

# ログ出力フォーマット設定
handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # noqa: E501
stream_handler.setFormatter(handler_format)

# --------------------------------
# 3.loggerにhandlerをセット
# --------------------------------
logger.addHandler(stream_handler)

# --------------------------------
# ログ出力テスト
# --------------------------------


def main(args):
    bitbucket_url = args.bitbucket_url
    if not bitbucket_url:
        raise Exception('BITBUCKET_URL is required')

    bitbucket_username = args.bitbucket_username
    if not bitbucket_username:
        raise Exception('BITBUCKET_USERNAME is required')

    bitbucket_token = args.bitbucket_token
    if not bitbucket_token:
        raise Exception('BITBUCKET_TOKEN is required')

    bitbucket_workspace = args.bitbucket_workspace
    if not bitbucket_workspace:
        raise Exception('BITBUCKET_WORKSPACE is required')

    bitbucket_repository = args.bitbucket_repository
    if not bitbucket_repository:
        raise Exception('BITBUCKET_REPOSITORY is required')

    bitbucket_pr_id = args.bitbucket_pr_id

    cloud = Cloud(
        username=bitbucket_username,
        token=bitbucket_token,
    )

    repository = cloud.repositories.get(bitbucket_workspace, bitbucket_repository)  # noqa: E501
    pr = repository.pullrequests.get(bitbucket_pr_id)

    # logger.debug(vars(pr))
    logger.info(f'{pr.title=}')
    logger.info(f'{pr.description=}')
    logger.info(f'{pr.diff()=}')
    logger.info(f'{pr.diffstat()=}')

    responses = []

    messages = [
        {
            "role": "system",
            "content": prompt._SYSTEM_MESSAGE['default']
        },
        {
            'role': "user",
            'content': '',
        }
    ]

    prompt.title = pr.title
    prompt.description = pr.description
    prompt.diff = pr.diff()

    file_diffs = []
    for diff in pr.diff().split('diff --git'):
        if diff.strip():
            file_diffs.append(f'diff --git{diff}')

    print()
    print()
    print()
    print()
    for index, diffstat in enumerate(pr.diffstat()):
        print('8' * 100)
        # print('diffstat')
        # print(diffstat)
        # pprint(diffstat.new.__dir__())
        # print()
        # print()
        prompt.filename = diffstat.new.escaped_path
        link = diffstat.new.get_link('self')
        file_diff = file_diffs[index]
        # print(f'{index=}')
        # print(f'{prompt.filename=}')
        # print(f'{link=}')
        # print('file_diff')
        # pprint(file_diff)
        # print()
        prompt.file_diff = file_diff
        # print('WE ARE SAPPORO!')
        messages[1]['content'] = f'{prompt.summarize_file_diff}'
        print('************* messages **********')
        print(messages[1]['content'])
        # logger.info(f'{messages=}')

        response = completion(model=AI_MODEL, messages=messages)
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        print('************* response **********')
        print(response)
        prompt.short_summary = content
        messages[1]['content'] = f'{prompt.review_file_diff}\nin Japanese'
        print('************* messages **********')
        print(messages[1]['content'])
        # logger.info(f'{messages=}')
        response = completion(model=AI_MODEL, messages=messages)
        print('************* response **********')
        print(response)
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        responses.append(content)


    print()
    print()
    print()
    print(responses)
    print()
    for res in responses:
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--bitbucket-url',
        default=os.environ.get('BITBUCKET_URL'),
    )
    parser.add_argument(
        '--bitbucket-username',
        default=os.environ.get('BITBUCKET_USERNAME'),
    )
    parser.add_argument(
        '--bitbucket-token',
        default=os.environ.get('BITBUCKET_TOKEN'),
    )
    parser.add_argument(
        '--bitbucket-workspace',
        default=os.environ.get('BITBUCKET_WORKSPACE'),
    )
    parser.add_argument(
        '--bitbucket-repository',
        default=os.environ.get('BITBUCKET_REPOSITORY'),
    )
    parser.add_argument(
        '--bitbucket-pr-id',
        type=int,
        required=True,
    )

    args = parser.parse_args()
    main(args)
