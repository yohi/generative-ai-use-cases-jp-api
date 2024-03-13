"""

cf.)
https://github.com/aws-samples/generative-ai-use-cases-jp/actions/runs/7004938177/workflow
https://zenn.dev/t_dai
https://zenn.dev/tadyjp/scraps/a7510f838edf8c
"""
import argparse
import os
import re
from copy import deepcopy

import sentry_sdk
from atlassian.bitbucket import Cloud
from dotenv import load_dotenv
from litellm import BadRequestError, completion

import prompt as prompt_module

sentry_sdk.init(
    dsn=os.environ.get('SENTRY_DSN'),
    enable_tracing=True
)

# AIモデルの定義
GEMINI = 'gemini/gemini-pro'
CLAUDE_3 = 'claude-3-sonnet-20240229'
BEDROCK_CLAUDE_2_1 = 'bedrock/anthropic.claude-v2:1'
BEDROCK_CLAUDE_3 = 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0'

# デフォルトで使用するAIモデル
AI_MODEL = BEDROCK_CLAUDE_3

# dotenv周りの設定
dir_path = os.path.dirname(os.path.abspath("__file__"))
dotenv_path = os.path.join(dir_path, '.env')
load_dotenv(dotenv_path, verbose=True)


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
    # (文字)黒
    BLACK = '\033[30m'
    # (文字)赤
    RED = '\033[31m'
    # (文字)緑
    GREEN = '\033[32m'
    # (文字)黄
    YELLOW = '\033[33m'
    # (文字)青
    BLUE = '\033[34m'
    # (文字)マゼンタ
    MAGENTA = '\033[35m'
    # (文字)シアン
    CYAN = '\033[36m'
    # (文字)白
    WHITE = '\033[37m'
    # 文字色をデフォルトに戻す
    COLOR_DEFAULT = '\033[39m'

    # 太字
    BOLD = '\033[1m'
    # 下線
    UNDERLINE = '\033[4m'
    # 不可視
    INVISIBLE = '\033[08m'
    # 文字色と背景色を反転
    REVERCE = '\033[07m'

    # (背景)黒
    BG_BLACK = '\033[40m'
    # (背景)赤
    BG_RED = '\033[41m'
    # (背景)緑
    BG_GREEN = '\033[42m'
    # (背景)黄
    BG_YELLOW = '\033[43m'
    # (背景)青
    BG_BLUE = '\033[44m'
    # (背景)マゼンタ
    BG_MAGENTA = '\033[45m'
    # (背景)シアン
    BG_CYAN = '\033[46m'
    # (背景)白
    BG_WHITE = '\033[47m'
    # 背景色をデフォルトに戻す
    BG_DEFAULT = '\033[49m'

    # 全てリセット
    RESET = '\033[0m'


# def cleaned_patch(diff_patch, pr):
#     """
#     """
#     file_split_pattern = ' '.join(['diff', '--git', 'a/'])
#     patch_split_regex = r"(^@@ -(\d+),(\d+) \+(\d+),(\d+) @@)"
# 
#     # patchを"diff --git "（ファイルごと）で分割
#     diff_patches = diff_patch.split(file_split_pattern)[1:]
# 
#     files = []
#     for index, diffstat in enumerate(pr.diffstat()):
#         filediff = diff_patches[index]
#         filename = diffstat.new.escaped_path
#         link = diffstat.new.get_link('self')
# 
#         patches = []
#         iter = reversed(list(re.finditer(patch_split_regex, filediff, re.MULTILINE)))
#         _dict = {
#             'filename': filename,
#             'link': link,
#             'filediff': filediff,
#         }
#         for m in iter:
#             match, old_begin, old_diff, new_begin, new_diff = m.groups()
#             # match（"@@ -W,X +Y,Z @@"）が最後に現れるところでで分割
#             index = filediff.rindex(match)
#             patch = filediff[index + len(match):]
#             filediff = filediff[:index]
#             patch = {
#                 'patch': patch,
#                 'old_hunk_line': {
#                     'start_line': int(old_begin),
#                     'end_line': int(old_begin) + int(old_diff) - 1,
#                 },
#                 'new_hunk_line': {
#                     'start_line': int(new_begin),
#                     'end_line': int(new_begin) + int(new_diff) - 1,
#                 }
#             }
#             patches.append(parse_patch(patch))
#         _dict['patches'] = list(reversed(patches))
#         files.append(_dict)
# 
#     return files


def parse_hunk(p):
    """
    hunkをパース（解析）して、old_hunkとnew_hunkを追加して返す
    """
    # 古いhunkの行を格納するリスト
    old_hunk_lines = []
    # 新しいhunkの行を格納するリスト
    new_hunk_lines = []
    # 新しいhunkの開始行
    new_line = p['new_hunk_line']['start_line']

    # パッチを行レベルに分割
    lines = p['hunk'].split('\n')

    if lines[0] == '':
        # 先頭行が空行の場合は削除
        lines.pop(0)

    if lines[-1] == '':
        # 最終行が空行の場合は削除
        lines.pop()

    # 前後の注釈をスキップする行数
    skip_start = 3
    skip_end = 3

    # 新規追加行が存在しないか否か
    remove_only = not [line for line in lines if line.startswith('+')]

    for current_line, line in enumerate(lines, 1):
        # 1行ずつ解析
        if line.startswith('-'):
            # 行頭が"-"の場合は古いhunkの行として追加
            old_hunk_lines.append(line[1:])
        elif line.startswith('+'):
            # 行頭が"+"の場合は新しいhunkの行として追加
            new_hunk_lines.append(f'{new_line}: {line[1:]}')
            new_line += 1
        else:
            old_hunk_lines.append(line)
            if remove_only or (skip_start < current_line <= len(lines) - skip_end):  # noqa: E501
                # 新規追加行がない場合、または前後の注釈をスキップする行数の範囲内の場合は新しいhunkに行数を記述  # noqa: E501
                new_hunk_lines.append(f'{new_line}: {line}')
            else:
                new_hunk_lines.append(f'{line}')
            new_line += 1

    return p | {
        'old_hunk': '\n'.join(old_hunk_lines),
        'new_hunk': '\n'.join(new_hunk_lines)
    }


def chat(user_message: str, debug: int, jp: bool = False) -> str:
    """
    生成AIにメッセージを送信して応答を受け取る

    Args:
        user_message: 送信するメッセージ
        debug: デバッグレベル
        jp: 日本語モードか否か

    Returns:
        生成AIからの応答
    """
    ai_model = AI_MODEL

    # システムロールのメッセージ
    system_message = {
        "role": "system",
        "content": prompt._SYSTEM_MESSAGE['default']
    }
    if jp:
        # 日本語モードの場合はシステムメッセージに日本語で応答するよう指示を追加
        system_message['content'] += '\nIMPORTANT: Entire response must be in the language with ISO code: ja-JP'  # noqa: E501

    # 送信メッセージを格納するリスト
    messages = [
        system_message,
    ]
    messages.append(
        {
            "role": "user",
            "content": user_message,
        },
    )

    # 初回メッセージのコピーを保持
    first_messages = deepcopy(messages)

    if debug > 2:
        print('************* request *************')
        print(f'{Color.RED}{messages[0]["content"]}{Color.RESET}')
        print(f'{Color.RED}{messages[-1]["content"]}{Color.RESET}')

    finish_reason = None
    content = ''
    while finish_reason in (None, 'max_tokens', 'length', 'bad_request'):
        # 終了理由がNone, max_tokens, length, bad_requestの場合は継続
        print('xxxxxx')
        for m in messages:
            print(f'{m["role"]=}')
            print(f'{len(m["content"])=}')
        print('xxxxxx')
        try:
            # 生成AIにメッセージを送信
            response = completion(model=ai_model, messages=messages)
            # 終了理由の取得
            finish_reason = response.finish_reason
            # レスポンスメッセージの取得
            message = response.get('choices', [{}])[-1].get('message', {})
            # 送信メッセージにレスポンスを追加
            messages.append(dict(message))
            # レスポンスメッセージを連結
            content += message.get('content')
        except BadRequestError:
            # BadRequestErrorが発生した場合（2024/03/13時点でBedrockのClaude3だと3回目以降のやり取りができない）
            finish_reason = 'bad_request'
            # AIモデルをBedrockのClaude2.1に変更し最初からやり直す
            ai_model = BEDROCK_CLAUDE_2_1
            messages = first_messages
            content = ''
        except Exception:
            raise
        print(finish_reason)

    if debug > 1:
        print('************* response **********')
        print(f'{Color.GREEN}{content}{Color.RESET}')

    return content

def parse_patch(pr):
    """
    """
    # ファイル単位で分割するためのパターン TODO
    file_split_pattern = ' '.join(['diff', '--git', 'a/'])
    # patch単位で分割するための正規表現
    patch_split_regex = r"(^@@ -(\d+),(\d+) \+(\d+),(\d+) @@)"

    # patchを"diff --git "（ファイルごと）で分割
    patches = pr.diff().split(file_split_pattern)[1:]

    ret = []
    for index, diffstat in enumerate(pr.diffstat()):
        patch = patches[index]
        filename = diffstat.new.escaped_path
        link = diffstat.new.get_link('self')

        iter = reversed(list(re.finditer(patch_split_regex, patch, re.MULTILINE)))

        hunks = []
        d = {
            'filename': filename,
            'link': link,
            'patch': patch,
        }
        for m in iter:
            match, old_begin, old_diff, new_begin, new_diff = m.groups()
            # match（"@@ -W,X +Y,Z @@"）が最後に現れるところでで分割
            index = patch.rindex(match)
            hunk = patch[index + len(match):]
            patch = patch[:index]
            hunk = {
                'hunk': hunk,
                'old_hunk_line': {
                    'start_line': int(old_begin),
                    'end_line': int(old_begin) + int(old_diff) - 1,
                },
                'new_hunk_line': {
                    'start_line': int(new_begin),
                    'end_line': int(new_begin) + int(new_diff) - 1,
                }
            }
            hunks.append(parse_hunk(hunk))
        d['hunks'] = list(reversed(hunks))
        ret.append(d)

    return ret


def parse_pull_request(pr):
    title = pr.title
    description = pr.description
    patches = parse_patch(pr)
    return {
        'title': title,
        'description': description,
        'patches': patches,
    }



def main(args):
    """
    メイン処理

    Args:
        args: コマンドライン引数
    """
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

    debug = args.debug

    # Atlassian Cloudeオブジェクトの作成
    cloud = Cloud(
        username=bitbucket_username,
        token=bitbucket_token,
    )

    # リポジトリの取得
    repository = cloud.repositories.get(bitbucket_workspace, bitbucket_repository)  # noqa: E501
    # PRの取得
    pr = repository.pullrequests.get(bitbucket_pr_id)

    parsed_pr = parse_pull_request(pr)

    # PRのタイトル
    prompt.title = parsed_pr['title']
    # PRの説明
    prompt.description = parsed_pr['description']

    patches = parsed_pr['patches']

    for p in patches:
        filename = p['filename']
        link = p['link']
        patch = p['patch']
        hunks = p['hunks']

        prompt.filename = filename
        prompt.link = link
        prompt.patch = patch
        prompt.diff = patch  # todo

        prompt.hunk = '\n'.join([f'''
---new_hunk---
```
{hunk["new_hunk"]}
```

---old_hunk---
```
{hunk["old_hunk"]}
```
''' for hunk in hunks])


        messages = [
            {
                "role": "system",
                "content": prompt._SYSTEM_MESSAGE['default'],
            },
        ]
        messages.append(
            {
                "role": "user",
                "content": prompt.summarize_file_diff,
            }
        )

        if debug > 0:
            print(f'************* filename={prompt.filename} **********')

        if debug > 2:
            print('************* request *************')
            print(f'{Color.RED}{messages[-1]["content"]}{Color.RESET}')

        content = chat(prompt.summarize_file_diff, debug)

        triage_regex = r"\[TRIAGE\]:\s*(NEEDS_REVIEW|APPROVED)"
        triage_match = re.search(triage_regex, content, re.MULTILINE)
        needs_review = True
        if triage_match:
            triage = triage_match.group(1)
            needs_review = triage == 'NEEDS_REVIEW'

        if needs_review:
            summary = re.sub(r"^.*triage.*$", '', content, 0, re.MULTILINE | re.IGNORECASE).strip()  # noqa: E501

            prompt.short_summary = summary

            content = chat(prompt.review_file_diff, debug, jp=True)

            reviews = parse_review(content, hunks)
            lgtm_count = 0
            review_count = 0
            print(f'{len(reviews)=}')
            for review in reviews:
                if review['comment'].find('LGTM') > -1 or review['comment'].find('looks good to me') > -1:  # noqa: E501
                    lgtm_count += 1
                    continue
                review_count += 1
                # review_comment(filename, link, review)
                if debug > 0:
                    print(f'''{Color.YELLOW}
                    {filename=}
                    {link=}
                    {review['start_line']=}
                    {review['end_line']=}
                    {review['comment']=}
                    {Color.RESET}''')

                data = {
                    "content": {
                        "raw": review['comment'],
                    },
                    "inline": {
                        "path": filename,
                        "to": review['start_line'],
                    },
                }
                pr.post('comments', data=data)

            print(f'{lgtm_count=}')
            print(f'{review_count=}')


def parse_review(content, patches):
    reviews = []
    content = saniteze_response(content.strip())

    lines = content.split('\n')
    line_number_range_regex = r"(?:^|\s)(\d+)-(\d+):\s*$"
    comment_separator = '---'

    def store_review(current_start_line, current_end_line, current_comment):
        if current_start_line is not None and current_end_line is not None:
            review = {
                'start_line': current_start_line,
                'end_line': current_end_line,
                'comment': current_comment,
            }

            within_patch = False
            best_patch_start_line = -1
            best_patch_end_line = -1
            max_intersection = 0

            for p in patches:
                start_line = p['new_hunk_line']['start_line']
                end_line = p['new_hunk_line']['end_line']

                intersection_start = max(review['start_line'], start_line)
                intersection_end = min(review['end_line'], end_line)
                intersection_length = max(0, intersection_end - intersection_start + 1)  # noqa: E501

                if intersection_length > max_intersection:
                    max_intersection = intersection_length
                    best_patch_start_line = start_line
                    best_patch_end_line = end_line
                    within_patch = (intersection_length == review['end_line'] - review['start_line'] + 1)  # noqa: E501

                if within_patch:
                    break

            if not within_patch:
                if best_patch_start_line != -1 and best_patch_end_line != -1:
                    review = {
                        'comment':  f'> Note: This review was outside of the patch, so it was mapped to the patch with the greatest overlap. Original lines [{review["start_line"]}-{review["end_line"]}]\n\n{review["comment"]}',  # noqa: E501
                        'start_line': best_patch_start_line,
                        'end_line': best_patch_end_line,
                    }
                else:
                    review = {
                        'comment': f'> Note: This review was outside of the patch, but no patch was found that overlapped with it. Original lines [{review["start_line"]}-{review["end_line"]}]\n\n{review["comment"]}',  # noqa: E501'
                        'start_line': patches[0]['new_hunk_line']['start_line'],  # noqa: E501
                        'end_line': patches[0]['new_hunk_line']['end_line'],
                    }

            reviews.append(review)

    current_start_line = None
    current_end_line = None
    current_comment = ''

    for line in lines:
        line_number_range_match = re.match(line_number_range_regex, line)

        if line_number_range_match:
            store_review(current_start_line, current_end_line, current_comment)
            current_start_line = int(line_number_range_match.group(1))
            current_end_line = int(line_number_range_match.group(2))
            current_comment = ''
            continue

        if line.strip() == comment_separator:
            store_review(current_start_line, current_end_line, current_comment)
            current_start_line = None
            current_end_line = None
            current_comment = ''
            continue

        if current_start_line is not None and current_end_line is not None:
            current_comment += f'{line}\n'

    store_review(current_start_line, current_end_line, current_comment)

    return reviews


def saniteze_response(content):
    content = sanitize_code_brock(content,  'suggestion')
    content = sanitize_code_brock(content,  'diff')
    return content


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

        code_block_start_index = comment.find(code_block_start, code_block_start_index + len(code_block_start + sanitized_block + code_block_end))

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

    parser.add_argument(
        '--debug',
        type=int,
        default=0,
    )

    args = parser.parse_args()
    main(args)
