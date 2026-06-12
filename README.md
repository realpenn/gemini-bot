# Gemini Telegram Bot

一个只在指定 Telegram 群聊和指定用户私聊中工作的 bot：

- 在允许的群聊里，回复 `@bot` 或回复 bot 的消息会触发回答。
- 如果配置了 `QUESTION_PREFIX`，允许的群聊里以该前缀开头的消息会被当作提问。
- 群隐私模式下推荐使用 `/ask@bot 问题`。
- 如果配置了 `ALLOWED_USER_ID`，该用户允许私聊 bot；该用户在群里的普通消息不会自动触发。
- 其他私聊、群聊或直接访问会回复 `NO_PERMISSION_TEXT`。
- 普通群聊消息如果不是直接找 bot，bot 会保持沉默，避免刷屏。
- 模型回复会转换为 Telegram MarkdownV2 格式发送；如果解析失败，会自动退回纯文本。
- 使用 OpenAI-compatible 接口调用配置的模型。
- 支持通过 Tavily API 调用 `web_search` tool 获取实时网络信息。

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 配置

```bash
cp .env.example .env
```

然后在 `.env` 中填入：

- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY`
- `GEMINI_API_BASE`
- `GEMINI_MODEL`
- `TAVILY_API_KEY`（可选，不填则禁用 web 搜索工具）
- `ALLOWED_CHAT_ID`（可选，不填则不允许群聊自动响应）
- `ALLOWED_USER_ID`（可选，不填则不允许私聊）
- `QUESTION_PREFIX`（可选，不填则禁用前缀触发）
- `NO_PERMISSION_TEXT`（可选）

不要提交 `.env`；公开仓库只提交 `.env.example`。

`TAVILY_API_KEY` 不填时，bot 仍可运行，但不会向模型暴露 web 搜索工具。
模型遇到新闻、最新状态、价格、版本、政策、赛事、产品规格等可能变化的问题时，会自动调用 Tavily 搜索并基于结果回答。

Telegram Bot API 的普通消息更新里不会带群邀请链接，不能直接用 `https://t.me/+...` 判定群。获取 `ALLOWED_CHAT_ID` 的方式：

1. 把 bot 加入目标群。
2. 在目标群发送 `/chatid`。
3. 把 bot 回复的数字填入 `.env` 的 `ALLOWED_CHAT_ID`。

如果希望 bot 自动看到群里的普通文本触发词，需要在 BotFather 里关闭该 bot 的 group privacy mode。否则 Telegram 只会把命令、@bot、以及回复 bot 的消息发给 bot。

## 运行

```bash
. .venv/bin/activate
python bot.py
```
