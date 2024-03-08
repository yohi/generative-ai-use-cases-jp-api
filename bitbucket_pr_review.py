"""

cf.)
https://github.com/aws-samples/generative-ai-use-cases-jp/actions/runs/7004938177/workflow
https://zenn.dev/t_dai
https://zenn.dev/tadyjp/scraps/a7510f838edf8c
"""
import argparse
from copy import deepcopy
from atlassian.bitbucket import Cloud
import os
import logging
from dotenv import load_dotenv
from litellm import completion
from pprint import pprint
import prompt as prompt_module


dir_path = os.path.dirname(os.path.abspath("__file__"))
dotenv_path = os.path.join(dir_path, '.env')
load_dotenv(dotenv_path, verbose=True)

AI_MODEL = 'gemini/gemini-pro'
# AI_MODEL = 'claude-3-sonnet-20240229'

prompt = prompt_module.MyPrompt()

# level = logging.ERROR

# # --------------------------------
# # 1.loggerの設定
# # --------------------------------
# # loggerオブジェクトの宣言
# logger = logging.getLogger(__name__)
# 
# # loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
# logger.setLevel(level)
# 
# # --------------------------------
# # 2.handlerの設定
# # --------------------------------
# # handlerの生成
# stream_handler = logging.StreamHandler()
# 
# # handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
# stream_handler.setLevel(level)
# 
# # ログ出力フォーマット設定
# handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # noqa: E501
# stream_handler.setFormatter(handler_format)
# 
# # --------------------------------
# # 3.loggerにhandlerをセット
# # --------------------------------
# logger.addHandler(stream_handler)
# 
# # --------------------------------
# # ログ出力テスト
# # --------------------------------

class Color:
    BLACK          = '\033[30m'  # (文字)黒
    RED            = '\033[31m'  # (文字)赤
    GREEN          = '\033[32m'  # (文字)緑
    YELLOW         = '\033[33m'  # (文字)黄
    BLUE           = '\033[34m'  # (文字)青
    MAGENTA        = '\033[35m'  # (文字)マゼンタ
    CYAN           = '\033[36m'  # (文字)シアン
    WHITE          = '\033[37m'  # (文字)白
    COLOR_DEFAULT  = '\033[39m'  # 文字色をデフォルトに戻す
    BOLD           = '\033[1m'   # 太字
    UNDERLINE      = '\033[4m'   # 下線
    INVISIBLE      = '\033[08m'  # 不可視
    REVERCE        = '\033[07m'  # 文字色と背景色を反転
    BG_BLACK       = '\033[40m'  # (背景)黒
    BG_RED         = '\033[41m'  # (背景)赤
    BG_GREEN       = '\033[42m'  # (背景)緑
    BG_YELLOW      = '\033[43m'  # (背景)黄
    BG_BLUE        = '\033[44m'  # (背景)青
    BG_MAGENTA     = '\033[45m'  # (背景)マゼンタ
    BG_CYAN        = '\033[46m'  # (背景)シアン
    BG_WHITE       = '\033[47m'  # (背景)白
    BG_DEFAULT     = '\033[49m'  # 背景色をデフォルトに戻す
    RESET          = '\033[0m'   # 全てリセット


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
    # logger.info(f'{pr.title=}')
    # logger.info(f'{pr.description=}')
    # logger.info(f'{pr.diff()=}')
    # logger.info(f'{pr.diffstat()=}')

    responses = []

    default_messages = [
        {
            "role": "system",
            "content": prompt._SYSTEM_MESSAGE['default']
        },
    ]

    prompt.title = pr.title
    prompt.description = pr.description
    prompt.diff = pr.diff()

    file_diffs = []
    for diff in pr.diff().split('diff --git'):
        if diff.strip():
            file_diffs.append(f'diff --git{diff}')

    for index, diffstat in enumerate(pr.diffstat()):
        messages = deepcopy(default_messages)
        print('8' * 100)
        # print('diffstat')
        # print(diffstat)
        # pprint(diffstat.new.__dir__())
        print(messages)
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
        messages.append(
            {
                "role": "user",
                "content": prompt.summarize_file_diff,
            }
        )
        # print('************* messages **********')
        # print(messages[1]['content'])
        # logger.info(f'{messages=}')

        response = completion(model=AI_MODEL, messages=messages)
        print(response)
        content = response.get('choices', [{}])[-1].get('message', {}).get('content')
        # print('************* response **********')
        # print(content)
        prompt.short_summary = content.split('## Triage:')[0]
        language = 'Please your answer description in Japanese.'
        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f'{language}\n{prompt.review_file_diff}',
            }
        )
        print('************* messages **********')
        print(f'{Color.RED}{messages[-1]["content"]}{Color.RESET}')
        # logger.info(f'{messages=}')
        response = completion(model=AI_MODEL, messages=messages)
        print(response)
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        print('************* response **********')
        print(response.get('choices', [{}]))
        print(f'{Color.GREEN}{content}{Color.RESET}')
        responses.append(content)

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
