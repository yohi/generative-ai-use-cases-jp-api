"""

cf.)
https://github.com/aws-samples/generative-ai-use-cases-jp/actions/runs/7004938177/workflow
https://zenn.dev/t_dai
https://zenn.dev/tadyjp/scraps/a7510f838edf8c
"""
import re
import argparse
from copy import deepcopy
from types import new_class
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

# AI_MODEL = 'gemini/gemini-pro'
# AI_MODEL = 'claude-3-sonnet-20240229'
AI_MODEL = 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0'

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
    BLACK = '\033[30m'  # (文字)黒
    RED = '\033[31m'  # (文字)赤
    GREEN = '\033[32m'  # (文字)緑
    YELLOW = '\033[33m'  # (文字)黄
    BLUE = '\033[34m'  # (文字)青
    MAGENTA = '\033[35m'  # (文字)マゼンタ
    CYAN = '\033[36m'  # (文字)シアン
    WHITE = '\033[37m'  # (文字)白
    COLOR_DEFAULT = '\033[39m'  # 文字色をデフォルトに戻す
    BOLD = '\033[1m'   # 太字
    UNDERLINE = '\033[4m'   # 下線
    INVISIBLE = '\033[08m'  # 不可視
    REVERCE = '\033[07m'  # 文字色と背景色を反転
    BG_BLACK = '\033[40m'  # (背景)黒
    BG_RED = '\033[41m'  # (背景)赤
    BG_GREEN = '\033[42m'  # (背景)緑
    BG_YELLOW = '\033[43m'  # (背景)黄
    BG_BLUE = '\033[44m'  # (背景)青
    BG_MAGENTA = '\033[45m'  # (背景)マゼンタ
    BG_CYAN = '\033[46m'  # (背景)シアン
    BG_WHITE = '\033[47m'  # (背景)白
    BG_DEFAULT = '\033[49m'  # 背景色をデフォルトに戻す
    RESET = '\033[0m'   # 全てリセット


def cleaned_patch(diff_patch, pr):
    """

    [
        {
            "filename": "src/main/java/com/example/demo/HelloController.java",
            "link": "https://bitbucket.org/.../src/main/java/com/example/demo/HelloController.java",
            "filediff": "",
            "patches": [
                {
                    "patch": "",
                    "hunks": 
                    "old_hunk": {
                        "start_line": 1,
                        "end_line": 1
                    },
                    "new_hunk": {
                        "start_line": 1,
                        "end_line": 1
                    }
                }
            ]
        }

    ]
    """
    file_split_pattern = ' '.join(['diff', '--git', 'a/'])
    patch_split_regex = r"(^@@ -(\d+),(\d+) \+(\d+),(\d+) @@)"

    # print(diff_patch)
    # patchを"diff --git "（ファイルごと）で分割
    diff_patches = diff_patch.split(file_split_pattern)[1:]

    files = []
    for index, diffstat in enumerate(pr.diffstat()):
        filediff = diff_patches[index]
        filename = diffstat.new.escaped_path
        link = diffstat.new.get_link('self')

        patches = []
        iter = reversed(list(re.finditer(patch_split_regex, filediff, re.MULTILINE)))
        _dict = {
            'filename': filename,
            'link': link,
            'filediff': filediff,
        }
        for m in iter:
            match, old_begin, old_diff, new_begin, new_diff = m.groups()
            # match（"@@ -W,X +Y,Z @@"）が最後に現れるところでで分割
            index = filediff.rindex(match)
            patch = filediff[index + len(match):]
            filediff = filediff[:index]
            patch = {
                'patch': patch,
                'old_hunk_line': {
                    'start_line': int(old_begin),
                    'end_line': int(old_begin) + int(old_diff) - 1,
                },
                'new_hunk_line': {
                    'start_line': int(new_begin),
                    'end_line': int(new_begin) + int(new_diff) - 1,
                }
            }
            patches.append(parse_patch(patch))
        _dict['patches'] = list(reversed(patches))
        files.append(_dict)

    return files


def parse_patch(p):
    # TODO
    old_hunk_lines = [
    ]
    new_hunk_lines = [
    ]

    new_line = p['new_hunk_line']['start_line']
    lines = p['patch'].split('\n')

    if lines[-1] == '':
        # 最終行が空行の場合は削除
        lines.pop()

    # 前後の注釈をスキップする行数
    skip_start = 3
    skip_end = 3

    remove_only = not [line for line in lines if line.startswith('+')]

    for current_line, line in enumerate(lines):
        if line.startswith('-'):
            old_hunk_lines.append(line[1:])
        elif line.startswith('+'):
            new_hunk_lines.append(f'{new_line}: {line[1:]}')
            new_line += 1
        else:
            old_hunk_lines.append(line)
            if remove_only or (current_line > skip_start and current_line <= len(lines) - skip_end):
                new_hunk_lines.append(f'{new_line}: {line}')
            else:
                new_hunk_lines.append(f'{line}')
            new_line += 1

    return p | {
        'old_hunk': '\n'.join(old_hunk_lines),
        'new_hunk': '\n'.join(new_hunk_lines)
    }

# def cleaned_patch(patch):
#     regex = r"(^@@ -(\d+),(\d+) \+(\d+),(\d+) @@)"
# 
#     pattern = ' '.join(['diff', '--git', 'a/'])
#     diffs = []
#     for diff in patch.split(pattern):
#         # patchを"diff --git "（ファイルごと）で分割
#         if not diff.split():
#             continue
#         patches = []
#         iter = reversed(list(re.finditer(regex, diff, re.MULTILINE)))
#         _dict = {
#             'diff': diff
#         }
#         for p in iter:
#             match, old_begin, old_diff, new_begin, new_diff = p.groups()
#             # match（"@@ -W,X +Y,Z @@"）が最後に現れるところでで分割
#             index = diff.rindex(match)
#             patch = diff[index + len(match):]
#             diff = diff[:index]
#             patches.append({
#                 'patch': patch,
#                 'old_hunk': {
#                     'start_line': int(old_begin),
#                     'end_line': int(old_begin) + int(old_diff) - 1,
#                 },
#                 'new_hunk': {
#                     'start_line': int(new_begin),
#                     'end_line': int(new_begin) + int(new_diff) - 1,
#                 }
#             })
#         _dict['patches'] = reversed(patches)
#         diffs.append(_dict)
# 
#     ret = []
#     for d in diffs:
#         hunks = []
#         for p in d['patches']:
#             hunks.append(parse_patch(p))
#         ret.append({
#             'diff': d['diff'],
#             'hunks': hunks
#         })
#     return ret


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

    responses = []

    system_messages = {
        "role": "system",
        "content": prompt._SYSTEM_MESSAGE['default']
    }

    # PRのタイトル
    prompt.title = pr.title
    # PRの説明
    prompt.description = pr.description
    # PRのdiffpatch
    diff_patch = pr.diff()

    files = cleaned_patch(diff_patch, pr)
    for file in files:
        prompt.filename = file['filename']
        prompt.link = file['link']
        # prompt.filediff = file['filediff']
        prompt.diff = file['filediff']
        hunks = []
        for patch in file['patches']:
            hunks.append(f'''
---new_hunk---
```
{patch["new_hunk"]}
```

---old_hunk---
```
{patch["old_hunk"]}
```
        ''')
        prompt.patches = '\n'.join(hunks)
        messages = [
            system_messages
        ]
        messages.append(
            {
                "role": "user",
                "content": prompt.summarize_file_diff,
            }
        )
        response = completion(model=AI_MODEL, messages=messages)
        content = response.get('choices', [{}])[-1].get('message', {}).get('content')
        print('************* response **********')
        print(f'{Color.GREEN}{content}{Color.RESET}')
        responses.append(content)

        triage_regex = r"\[TRIAGE\]:\s*(NEEDS_REVIEW|APPROVED)"
        triage_match = re.search(triage_regex, content, re.MULTILINE)
        needs_review = True
        if triage_match:
            triage = triage_match.group(1)
            needs_review = triage == 'NEEDS_REVIEW'

        if needs_review:
            summary = re.sub(r"^.*triage.*$", '', content, 0, re.MULTILINE | re.IGNORECASE).strip()

            prompt.short_summary = summary
            language = ''
            messages = [
                system_messages
            ]
            messages.append(
                {
                    "role": "user",
                    "content": f'{language}\n{prompt.review_file_diff}',
                }
            )

            # print('************* messages **********')
            # print(f'{Color.RED}{messages[-1]["content"]}{Color.RESET}')

            response = completion(model=AI_MODEL, messages=messages)
            content = response.get('choices', [{}])[0].get('message', {}).get('content')
            print('************* response **********')
            print(f'{Color.BLUE}{content}{Color.RESET}')

#             # sanitize_response
#             content = sanitize_code_brock(content,  'suggestion')
#             content = sanitize_code_brock(content,  'diff')
# 
#             lines = content.split('\n')
#             line_number_range_regex = r"(?:^|\s)(\d+)-(\d+):\s*$"
#             comment_separator = '---'
# 
#             for line in lines:
#                 line_number_range_match = re.match(line_number_range_regex, line)
#                 print(line_number_range_match)
#                 if line_number_range_match:
#                     current_start_line = int(line_number_range_match.group(1))
#                     current_end_line = int(line_number_range_match.group(2))
#                     current_comment = ''
#                     store_review(file['patches'], current_start_line, current_end_line, current_comment)
#             print('************* response **********')
#             print(f'{Color.YELLOW}{content}{Color.RESET}')
#             responses.append(content)

def store_review(patches, start_line, end_line, comment):
    within_patch = False
    best_batch_start_line = -1
    best_batch_end_line = -1
    max_intersection = 0

    print('okamura daihachi')
    print(patches)
    for patch in patches[start_line:end_line]:
        intersection = max(0, min(patch_end_line, end_line) - max(patch_start_line, start_line))
        if intersection > max_intersection:
            within_patch = True
            max_intersection = intersection
            best_batch_start_line = patch_start_line
            best_batch_end_line = patch_end_line



def sanitize_code_brock(comment, code_block_label):
    code_block_start = f'```{code_block_label}'
    line_number_regex = r"^ *(\d+):"
    code_block_end = '```'
    code_block_start_index = comment.find(code_block_start)
    while code_block_start_index != -1:
        code_block_end_index = comment.find(code_block_end, code_block_start_index)
        code_block = comment[code_block_start_index:code_block_end_index]
        code_block = code_block.replace(code_block_start, '').replace(code_block_end, '').strip()

        sanitized_block = re.sub(line_number_regex, '', code_block)

        comment = comment[:code_block_start_index] + sanitized_block + comment[code_block_end_index:]

        code_block_start_index = comment.find(code_block_start, code_block_start_index) + len(code_block_start + sanitized_block + code_block_end)

    return comment



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
