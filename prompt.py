
class BasePrompt:
    _SYSTEM_MESSAGE = {}
    _SUMMARIZE = {}
    _SUMMARIZE_RELEASE_NOTES = {}
    _SUMMARIZE_FILE_DIFF = ''
    _TRIAGE_FILE_DIFF = ''
    _SUMMARIZE_CHANGESETS = ''
    _SUMMARIZE_PREFIX = ''
    _SUMMARIZE_SHORT = ''
    _REVIEW_FILE_DIFF = ''
    _COMMENT = ''

    def __format(self, template):
        replaces = {key: value for key, value in self.__dict__.items()}
        return template.format(**replaces)
        # return '\n'.join([eval(f"f'{row}'") for row in template.split('\n')])

    @property
    def summarize_file_diff(self):
        return self.__format(self._SUMMARIZE_FILE_DIFF + self._TRIAGE_FILE_DIFF)  # noqa: E501

    @property
    def summarize_changesets(self):
        return self.__format(self._SUMMARIZE_CHANGESETS)

    @property
    def summarize(self):
        return self.__format(self._SUMMARIZE_PREFIX + self._SUMMARIZE)

    @property
    def summarize_short(self):
        return self.__format(self._SUMMARIZE_PREFIX + self._SUMMARIZE_SHORT)

    @property
    def summarize_release_notes(self):
        return self.__format(self._SUMMARIZE_PREFIX + self._SUMMARIZE_RELEASE_NOTES)  # noqa: E501

    @property
    def review_file_diff(self):
        return self.__format(self._REVIEW_FILE_DIFF)

    @property
    def comment(self):
        return self.__format(self._COMMENT)


class CodeRabbitPrompt(BasePrompt):
    # PRタイトル
    title = ''
    # PR説明
    description = ''
    # ファイルの差分
    file_diff = ''
    # ファイル変更の要約
    raw_summary = ''
    # 変更の要約
    short_summary = ''
    patches = ''
    diff = ''
    comment_chain = ''
    comment = ''
    filename = ''

    # プロンプトの説明
    _SYSTEM_MESSAGE = {
        'description': 'System message to be sent to OpenAI',
        'default': '''
You are `@coderabbitai` (aka `github-actions[bot]`), a language model
trained by OpenAI. Your purpose is to act as a highly experienced
software engineer and provide a thorough review of the code hunks
and suggest code snippets to improve key areas such as:
  - Logic
  - Security
  - Performance
  - Data races
  - Consistency
  - Error handling
  - Maintainability
  - Modularity
  - Complexity
  - Optimization
  - Best practices: DRY, SOLID, KISS

Do not comment on minor code style issues, missing
comments/documentation. Identify and resolve significant
concerns to improve overall code quality while deliberately
disregarding minor issues.
        '''  # noqa[E501],
    }

    # 全体の要約生成プロンプト
    _SUMMARIZE_PROMOPT_PROVIDE = {
        'description': 'The prompt for final summarization response',
        'default': '''
Provide your final response in markdown with the following content:

- **Walkthrough**: A high-level summary of the overall change instead of
  specific files within 80 words.
- **Changes**: A markdown table of files and their summaries. Group files
  with similar changes together into a single row to save space.
- **Poem**: Below the changes, include a whimsical, short poem written by
  a rabbit to celebrate the changes. Format the poem as a quote using
  the ">" symbol and feel free to use emojis where relevant.

Avoid additional commentary as this summary will be added as a comment on the
GitHub pull request. Use the titles "Walkthrough" and "Changes" and they must be H2.
        ''',  # noqa[E501]
    }

    # リリースノートの要約生成プロンプト
    _SUMMARIZE_RELEASE_NOTES_PROVIDE = {
        'description': 'The prompt for generating release notes in the same chat as summarize stage',  # noqa[E501]
        'default': '''
Craft concise release notes for the pull request.
Focus on the purpose and user impact, categorizing changes as "New Feature", "Bug Fix",
"Documentation", "Refactor", "Style", "Test", "Chore", or "Revert". Provide a bullet-point list,
e.g., "- New Feature: Added search functionality to the UI". Limit your response to 50-100 words
and emphasize features visible to the end-user while omitting code-level details.
        '''  # noqa[E501],
    }

    _SUMMARIZE_CHANGESETS = '''
Provided below are changesets in this pull request. Changesets are in chronlogical order and new changesets are appended to the end of the list. The format consists of filename(s) and the summary of changes for those files. There is a separator between each changeset.
Your task is to deduplicate and group together files with related/similar changes into a single changeset. Respond with the updated changesets using the same format as the input.

{raw_summary}
''' # noqa[E501]

    _SUMMARIZE_SHORT = '''
Your task is to provide a concise summary of the changes. This summary will be used as a prompt while reviewing each file and must be very clear for the AI bot to understand.

Instructions:

- Focus on summarizing only the changes in the PR and stick to the facts.
- Do not provide any instructions to the bot on how to perform the review.
- Do not mention that files need a through review or caution about potential issues.
- Do not mention that these changes affect the logic or functionality of the code.
- The summary should not exceed 500 words.
''' # noqa[E501]

    # 差分の要約生成プロンプト
    _SUMMARIZE_FILE_DIFF = '''
## GitHub PR Title

{title}

## Description

{description}

## Diff

```diff
{file_diff}
```

## Instructions

I would like you to succinctly summarize the diff within 100 words.
If applicable, your summary should include a note about alterations
to the signatures of exported functions, global data structures and
variables, and any changes that might affect the external interface or
behavior of the code.
''' # noqa[E501]

    # ファイルの変更要約生成プロンプト
    _TRIAGE_FILE_DIFF = '''
Below the summary, I would also like you to triage the diff as `NEEDS_REVIEW` or `APPROVED` based on the following criteria:

- If the diff involves any modifications to the logic or functionality, even if they seem minor, triage it as `NEEDS_REVIEW`. This includes changes to control structures, function calls, or variable assignments that might impact the behavior of the code.
- If the diff only contains very minor changes that don't affect the code logic, such as fixing typos, formatting, or renaming variables for clarity, triage it as `APPROVED`.

Please evaluate the diff thoroughly and take into account factors such as the number of lines changed, the potential impact on the overall system, and the likelihood of introducing new bugs or security vulnerabilities.
When in doubt, always err on the side of caution and triage the diff as `NEEDS_REVIEW`.

You must strictly follow the format below for triaging the diff:
[TRIAGE]: <NEEDS_REVIEW or APPROVED>

Important:
- In your summary do not mention that the file needs a through review or caution about
  potential issues.
- Do not provide any reasoning why you triaged the diff as `NEEDS_REVIEW` or `APPROVED`.
- Do not mention that these changes affect the logic or functionality of the code in the summary. You must only use the triage status format above to indicate that.
''' # noqa[E501]

    _SUMMARIZE_PREFIX = '''
Here is the summary of changes you have generated for files:
```
{raw_summary}
```
    '''

    # ファイルの変更のレビュー要求
    _REVIEW_FILE_DIFF = '''
## GitHub PR Title

`{title}`

## Description

```
{description}
```

## Summary of changes

```
{short_summary}
```

## IMPORTANT Instructions

Input: New hunks annotated with line numbers and old hunks (replaced code). Hunks represent incomplete code fragments.
Additional Context: PR title, description, summaries and comment chains.
Task: Review new hunks for substantive issues using provided context and respond with comments if necessary.
Output: Review comments in markdown with exact line number ranges in new hunks. Start and end line numbers must be within the same hunk. For single-line comments, start=end line number. Must use example response format below.
Use fenced code blocks using the relevant language identifier where applicable.
Don't annotate code snippets with line numbers. Format and indent code correctly.
Do not use `suggestion` code blocks.
For fixes, use `diff` code blocks, marking changes with `+` or `-`. The line number range for comments with fix snippets must exactly match the range to replace in the new hunk.

- Do NOT provide general feedback, summaries, explanations of changes, or praises
  for making good additions.
- Focus solely on offering specific, objective insights based on the
  given context and refrain from making broad comments about potential impacts on
  the system or question intentions behind the changes.

If there are no issues found on a line range, you MUST respond with the
text `LGTM!` for that line range in the review section.

## Example

### Example changes

---new_hunk---
```
  z = x / y
    return z

20: def add(x, y):
21:     z = x + y
22:     retrn z
23:
24: def multiply(x, y):
25:     return x * y

def subtract(x, y):
  z = x - y
```

---old_hunk---
```
  z = x / y
    return z

def add(x, y):
    return x + y

def subtract(x, y):
    z = x - y
```

---comment_chains---
```
Please review this change.
```

---end_change_section---

### Example response

22-22:
There's a syntax error in the add function.
```diff
-    retrn z
+    return z
```
---
24-25:
LGTM!
---

## Changes made to `{filename}` for your review

{file_diff}
`
''' # noqa[E501]

    _COMMENT = '''
A comment was made on a GitHub PR review for a
diff hunk on a file - `{filename}`. I would like you to follow
the instructions in that comment.

## GitHub PR Title

`{title}`

## Description

```
{description}
```

## Summary generated by the AI bot

```
{short_summary}
```

## Entire diff

```diff
{file_diff}
```

## Diff being commented on

```diff
{diff}
```

## Instructions

Please reply directly to the new comment (instead of suggesting a reply) and your reply will be posted as-is.

If the comment contains instructions/requests for you, please comply.
For example, if the comment is asking you to generate documentation comments on the code, in your reply please generate the required code.

In your reply, please make sure to begin the reply by tagging the user with "@user".

## Comment format

`user: comment`

## Comment chain (including the new comment)

```
{comment_chain}
```

## The comment/request that you need to directly reply to

```
{comment}
```
''' # noqa[E501]



class PRAgentPrompt(BasePrompt):
    pass

