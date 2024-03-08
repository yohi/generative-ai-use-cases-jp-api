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


PATCH = '''
diff --git a/django/project/apps/notification/migrations/0001_initial.py b/django/project/apps/notification/migrations/0001_initial.py
index 6a1895d7c..d1d57d65a 100644
--- a/django/project/apps/notification/migrations/0001_initial.py
+++ b/django/project/apps/notification/migrations/0001_initial.py
@@ -125,7 +125,7 @@ class Migration(migrations.Migration):
             True: CREATE TABLE が必要
             False: CREATE TABLE が不要（この操作はスキップされます。）
         """
-        print(f'operation.name={operation.name}  exists_table_names={exists_table_names}')
+        # print(f'operation.name={operation.name}  exists_table_names={exists_table_names}')
         if operation.name == 'News':
             return 'news' not in exists_table_names
 
diff --git a/django/project/config/settings/project.py b/django/project/config/settings/project.py
index 3875766db..359cde8f5 100644
--- a/django/project/config/settings/project.py
+++ b/django/project/config/settings/project.py
@@ -315,3 +315,6 @@ TEST_RUNNER = 'django_utility.tests.runner.stacked_context_test_runner.StackedCo
 StackedContextTestRunner.set_contexts({
     'n_plus_one': {},
 })
+
+# 引当可能数計算のストアドファンクションで1回に処理するSKU数の上限値
+BULK_STORED_FUNCTION_MAX_SKU_LIMIT = env.int('BULK_STORED_FUNCTION_MAX_SKU_LIMIT', default=1000)
diff --git a/django/project/services/allocable_quantity_calculation.py b/django/project/services/allocable_quantity_calculation.py
index d6bd79673..d7a9bb730 100644
--- a/django/project/services/allocable_quantity_calculation.py
+++ b/django/project/services/allocable_quantity_calculation.py
@@ -6,6 +6,7 @@ Todo:
 """
 import datetime
 import logging
+import math
 import typing
 import uuid
 from collections.abc import Sequence
@@ -14,6 +15,7 @@ from functools import cache
 
 import jpholiday
 import polars as pl
+from django.conf import settings
 from django.db import connections
 from django.db.models import QuerySet
 from django.utils import timezone
@@ -1293,16 +1295,34 @@ class StoredAllocableQuantityCalculationService(BaseAllocableQuantityCalculation
         # CalculateTypeで宣言されているキー要素を取得
         keys = CalculateType.__annotations__.keys()
 
-        results = cls.__bulk_calculate_via_stored(
-            shop_id,
-            product_sku_ids,
-            stock_types,
-            rounding_unit,
-            round_by_product_sku,
-            mapping_rule,
-        )
+        # 1ストアドファンクションに対して実行すSKUの上限値
+        bulk_calculate_limit = settings.BULK_STORED_FUNCTION_MAX_SKU_LIMIT
+
+        # product_sku_idsの個数とBULK_CALCULATE_LIMITからrange_stopを算出（小数点切り上げ）
+        range_stop: int = math.ceil(len(product_sku_ids) / bulk_calculate_limit)
+
+        results = []
+
+        for i in range(range_stop):
+            # BULK_CALCULATE_LIMIT文だけ引当可能数を算出する
+            start = i * bulk_calculate_limit
+            stop = start + bulk_calculate_limit
+            product_sku_ids_chunk: list[int] = product_sku_ids[start:stop]
+            stock_types_chunk: list[int] = stock_types[start:stop]
+
+            results += cls.__bulk_calculate_via_stored(
+                shop_id,
+                product_sku_ids_chunk,
+                stock_types_chunk,
+                rounding_unit,
+                round_by_product_sku,
+                mapping_rule,
+            )
+
+        # キー結合したあとでkey_codeとlocation_codeでソートする
+        ret = sorted([dict(zip(keys, result)) for result in results], key=lambda x: (x['key_code'], x['location_code']))
 
-        return [dict(zip(keys, result)) for result in results]
+        return ret
 
     @classmethod
     def __calculate_via_stored(
diff --git a/django/project/tests/batch/execute/test_send_stock.py b/django/project/tests/batch/execute/test_send_stock.py
index 03b6ff78e..e584a1f82 100644
--- a/django/project/tests/batch/execute/test_send_stock.py
+++ b/django/project/tests/batch/execute/test_send_stock.py
@@ -14,6 +14,7 @@ from django.conf import settings
 from django.core.management import call_command
 from django.core.management.base import CommandError
 from django.db import DatabaseError
+from django.test import override_settings
 from django.utils import timezone
 from django.utils.timezone import datetime
 
@@ -1516,6 +1517,58 @@ class TestSendStock(AlertAssertionMixin, ScsBatchTenantTestCase):
                 # ローカルでテスト実施する時のみコメント解除する
                 # self.assertBackUp('milk_amzfs_send_stock_20210202200000.csv')
 
+    def test_23_01_limit_5(self) -> None:
+        """
+        ファイル作成処理1
+
+        ディシジョンテーブルNo23_1
+
+        ストアドファンクションが分割されるように上限を5に上書き
+        ストアド実行回数: 1, 最後に実行される対象数が上限より少ない
+        """
+        with override_settings(BULK_STORED_FUNCTION_MAX_SKU_LIMIT=5):
+            # ストアドファンクションが分割されるように上限を5に上書き
+            self.test_23_01()
+
+    def test_23_01_limit_3(self) -> None:
+        """
+        ファイル作成処理1
+
+        ディシジョンテーブルNo23_1
+
+        ストアドファンクションが分割されるように上限を3に上書き
+        ストアド実行回数: 2, 最後に実行される対象数が1
+        """
+        with override_settings(BULK_STORED_FUNCTION_MAX_SKU_LIMIT=3):
+            # ストアドファンクションが分割されるように上限を3に上書き
+            self.test_23_01()
+
+    def test_23_01_limit_2(self) -> None:
+        """
+        ファイル作成処理1
+
+        ディシジョンテーブルNo23_1
+
+        ストアドファンクションが分割されるように上限を2に上書き
+        ストアド実行回数: 2, 最後に実行される対象数が上限と同じ
+        """
+        with override_settings(BULK_STORED_FUNCTION_MAX_SKU_LIMIT=2):
+            # ストアドファンクションが分割されるように上限を2に上書き
+            self.test_23_01()
+
+    def test_23_01_limit_1(self) -> None:
+        """
+        ファイル作成処理1
+
+        ディシジョンテーブルNo23_1
+
+        ストアドファンクションが分割されるように上限を1に上書き
+        ストアド実行回数: 3以上（4）, 最後に実行される対象数が1（上限と同じ）
+        """
+        with override_settings(BULK_STORED_FUNCTION_MAX_SKU_LIMIT=1):
+            # ストアドファンクションが分割されるように上限を1に上書き
+            self.test_23_01()
+
     def test_23_02(self) -> None:
         """
         ファイル作成処理2

'''  # noqa: E501, W293

def parse_patch(p):
    # TODO
    old_hunk_lines = [
    ]
    new_hunk_lines = [
    ]

    new_line = p['new_hunk']['start_line']
    lines = p['patch'].split('\n')

    remove_only = not [line for line in lines if line.startswith('+')]

    for current_line, line in enumerate(lines):
        if line.startswith('-'):
            old_hunk_lines.append(line[1:])
        elif line.startswith('+'):
            new_hunk_lines.append(f'{new_line}: {line[1:]}')
            new_line += 1
        else:
            # old_hunk_lines.append(line)
            # new_hunk_lines.append(line)
            new_hunk_lines.append(f'{new_line}: {line[1:]}')
            new_line += 1
    # print('remove_only')
    # print(remove_only)

    return {
        'old_hunk': '\n'.join(old_hunk_lines),
        'new_hunk': '\n'.join(new_hunk_lines)
    }

def cleaned_patch(patch):
    # patch = PATCH
    regex = r"(^@@ -(\d+),(\d+) \+(\d+),(\d+) @@)"

    pattern = ' '.join(['diff', '--git '])
    diffs = []
    for diff in patch.split(pattern):
        # patchを"diff --git "（ファイルごと）で分割
        if not diff.split():
            continue
        patches = []
        iter = reversed(list(re.finditer(regex, diff, re.MULTILINE)))
        for p in iter:
            match, old_begin, old_diff, new_begin, new_diff = p.groups()
            # match（"@@ -W,X +Y,Z @@"）が最後に現れるところでで分割
            index = diff.rindex(match)
            patch = diff[index + len(match):]
            diff = diff[:index]
            patches.append({
                'patch': patch,
                'old_hunk': {
                    'start_line': int(old_begin),
                    'end_line': int(old_begin) + int(old_diff) - 1,
                },
                'new_hunk': {
                    'start_line': int(new_begin),
                    'end_line': int(new_begin) + int(new_diff) - 1,
                }
            })
        diffs.append(reversed(patches))

    ret = []
    for d in diffs:
        for p in d:
            ret.append(parse_patch(p))

    return ret


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

    default_messages = [
        {
            "role": "system",
            "content": prompt._SYSTEM_MESSAGE['default']
        },
    ]

    prompt.title = pr.title
    prompt.description = pr.description
    patch = pr.diff()
    prompt.diff = patch

    file_diffs = cleaned_patch(patch)

    hunk_str = ''

    for file_diff in file_diffs:

        hunk_str += f'''
---new_hunk---
```
{file_diff["new_hunk"]}
```

---old_hunk---
```
{file_diff["old_hunk"]}
```
'''
    prompt.file_diff = patch
    prompt.patches = hunk_str

    file_diffs = []
    print(f'{Color.RED}{prompt.summarize_file_diff}{Color.RESET}')
    prompt.short_summary = 'consadole'
    prompt.filename = 'sapporo'
    print(f'{Color.YELLOW}{prompt.review_file_diff}{Color.RESET}')
    return 0


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
        print('************* messages **********')
        print(f'{Color.RED}{messages[-1]["content"]}{Color.RESET}')

        response = completion(model=AI_MODEL, messages=messages)
        content = response.get('choices', [{}])[-1].get('message', {}).get('content')

        print('************* response **********')
        print(f'{Color.GREEN}{content}{Color.RESET}')

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

        response = completion(model=AI_MODEL, messages=messages)
        content = response.get('choices', [{}])[0].get('message', {}).get('content')

        print('************* response **********')
        print(f'{Color.BLUE}{content}{Color.RESET}')

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
