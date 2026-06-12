from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from tavily import AsyncTavilyClient
from telegram import BotCommand, Message, MessageEntity, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


LOGGER = logging.getLogger(__name__)
MAX_TELEGRAM_MESSAGE_LENGTH = 4096
ENV_FILE = Path(__file__).resolve().with_name(".env")
MAX_TOOL_ROUNDS = 3
MAX_WEB_SEARCH_RESULTS = 8
MAX_SEARCH_RESULT_CONTENT_CHARS = 1200
MAX_TOOL_RESULT_CHARS = 12000

SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}
SEARCH_TOPICS = {"general", "news", "finance"}
TIME_RANGES = {"day", "week", "month", "year"}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web with Tavily for current or external information. "
            "Use this for news, recent events, changing facts, prices, product details, "
            "or when the user asks you to look something up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query to run.",
                },
                "topic": {
                    "type": "string",
                    "enum": sorted(SEARCH_TOPICS),
                    "description": "Search category. Use news for current events.",
                },
                "search_depth": {
                    "type": "string",
                    "enum": sorted(SEARCH_DEPTHS),
                    "description": "Latency versus relevance tradeoff.",
                },
                "time_range": {
                    "type": "string",
                    "enum": sorted(TIME_RANGES),
                    "description": "Optional recency filter such as day, week, month, or year.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WEB_SEARCH_RESULTS,
                    "description": "Maximum number of search results to return.",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of domains to restrict results to.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of domains to exclude.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

WEB_TOOL_SYSTEM_PROMPT = (
    "你可以使用 web_search 工具获取实时或外部网络信息。"
    "当用户问题涉及新闻、最新状态、价格、版本、政策、赛事、产品规格等可能变化的信息，"
    "或用户明确要求查询网络时，应先调用该工具。使用工具结果回答时，请尽量给出来源标题和 URL。"
)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_api_base: str
    gemini_model: str
    tavily_api_key: str | None
    allowed_chat_id: int | None
    allowed_user_id: int | None
    no_permission_text: str
    question_prefix: str | None
    system_prompt: str
    request_timeout_seconds: float


@dataclass(frozen=True)
class BotIdentity:
    id: int
    username: str


def load_settings() -> Settings:
    load_dotenv(dotenv_path=ENV_FILE, override=False)

    telegram_bot_token = require_env("TELEGRAM_BOT_TOKEN")
    gemini_api_key = require_env("GEMINI_API_KEY")
    gemini_api_base = require_env("GEMINI_API_BASE")
    gemini_model = require_env("GEMINI_MODEL")
    allowed_chat_id = parse_optional_int(os.getenv("ALLOWED_CHAT_ID"), "ALLOWED_CHAT_ID")
    allowed_user_id = parse_optional_int(os.getenv("ALLOWED_USER_ID"), "ALLOWED_USER_ID")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        gemini_api_key=gemini_api_key,
        gemini_api_base=normalize_openai_base_url(gemini_api_base),
        gemini_model=gemini_model,
        tavily_api_key=optional_nonempty_env("TAVILY_API_KEY"),
        allowed_chat_id=allowed_chat_id,
        allowed_user_id=allowed_user_id,
        no_permission_text=env_or_default("NO_PERMISSION_TEXT", "无权限。"),
        question_prefix=optional_nonempty_env("QUESTION_PREFIX"),
        system_prompt=env_or_default(
            "SYSTEM_PROMPT",
            "你是一个在 Telegram 群聊中回答问题的中文助手。回答应清晰、准确、简洁。",
        ),
        request_timeout_seconds=float(env_or_default("REQUEST_TIMEOUT_SECONDS", "60")),
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def optional_nonempty_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def env_or_default(name: str, default_value: str) -> str:
    return optional_nonempty_env(name) or default_value


def parse_optional_int(value: str | None, name: str) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def message_text(message: Message) -> str:
    return message.text or message.caption or ""


def mentions_bot(message: Message, bot_username: str) -> bool:
    username = bot_username.removeprefix("@").lower()
    target = f"@{username}"
    text = message_text(message).lower()

    entities = message.entities or message.caption_entities or ()
    for entity in entities:
        if entity.type not in {MessageEntity.MENTION, MessageEntity.BOT_COMMAND}:
            continue
        token = message.parse_entity(entity).lower()
        if token == target or token.endswith(target):
            return True

    return target in text


def is_reply_to_bot(message: Message, bot_id: int) -> bool:
    reply = message.reply_to_message
    return bool(reply and reply.from_user and reply.from_user.id == bot_id)


def is_private_chat(message: Message) -> bool:
    return message.chat.type == ChatType.PRIVATE


def is_group_chat(message: Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


def is_allowed_group(message: Message, settings: Settings) -> bool:
    return (
        is_group_chat(message)
        and settings.allowed_chat_id is not None
        and message.chat.id == settings.allowed_chat_id
    )


def is_allowed_user_message(message: Message, settings: Settings) -> bool:
    return bool(
        settings.allowed_user_id is not None
        and message.from_user
        and message.from_user.id == settings.allowed_user_id
    )


def is_allowed_private_chat(message: Message, settings: Settings) -> bool:
    return is_private_chat(message) and is_allowed_user_message(message, settings)


def is_direct_bot_interaction(
    message: Message, settings: Settings, identity: BotIdentity
) -> bool:
    return (
        mentions_bot(message, identity.username)
        or is_reply_to_bot(message, identity.id)
        or has_question_prefix(message, settings)
    )


def has_question_prefix(message: Message, settings: Settings) -> bool:
    if not settings.question_prefix:
        return False
    return message_text(message).lstrip().startswith(settings.question_prefix)


def should_answer(message: Message, settings: Settings, identity: BotIdentity) -> bool:
    if is_allowed_private_chat(message, settings):
        return True
    if is_allowed_group(message, settings):
        return is_direct_bot_interaction(message, settings, identity)
    return False


def should_deny(message: Message, settings: Settings, identity: BotIdentity) -> bool:
    if should_answer(message, settings, identity):
        return False
    if is_private_chat(message):
        return True
    return is_direct_bot_interaction(message, settings, identity)


def strip_bot_mention(text: str, bot_username: str) -> str:
    username = re.escape(bot_username.removeprefix("@"))
    text = re.sub(rf"^/[A-Za-z0-9_]+@{username}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"@{username}\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_question_prefix(text: str, settings: Settings) -> str:
    text = text.lstrip()
    if settings.question_prefix and text.startswith(settings.question_prefix):
        text = text[len(settings.question_prefix) :]
    return text.strip()


def strip_leading_command(text: str) -> str:
    return re.sub(r"^/[A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)?\s*", "", text).strip()


def chunks(text: str, size: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> Iterable[str]:
    for start in range(0, len(text), size):
        yield text[start : start + size]


def model_system_prompt(settings: Settings, tools_enabled: bool) -> str:
    if not tools_enabled:
        return settings.system_prompt
    return f"{settings.system_prompt}\n\n{WEB_TOOL_SYSTEM_PROMPT}"


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


def normalized_choice(
    value: Any, allowed_values: set[str], default_value: str
) -> str:
    if isinstance(value, str) and value in allowed_values:
        return value
    return default_value


def normalized_max_results(value: Any) -> int:
    try:
        max_results = int(value)
    except (TypeError, ValueError):
        max_results = 5
    return max(1, min(max_results, MAX_WEB_SEARCH_RESULTS))


def normalized_domain_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    domains = []
    for item in value:
        if not isinstance(item, str):
            continue
        domain = item.strip().lower()
        if domain:
            domains.append(domain)
    return domains[:20]


def truncate_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def compact_tavily_response(response: dict[str, Any]) -> dict[str, Any]:
    compact_results = []
    for result in response.get("results", [])[:MAX_WEB_SEARCH_RESULTS]:
        if not isinstance(result, dict):
            continue
        compact_results.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": truncate_text(
                    result.get("content"), MAX_SEARCH_RESULT_CONTENT_CHARS
                ),
                "published_date": result.get("published_date"),
                "score": result.get("score"),
            }
        )

    return {
        "query": response.get("query"),
        "answer": response.get("answer"),
        "results": compact_results,
        "response_time": response.get("response_time"),
        "request_id": response.get("request_id"),
        "usage": response.get("usage"),
    }


def tool_result_json(payload: dict[str, Any]) -> str:
    content = json.dumps(payload, ensure_ascii=False)
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    return f"{content[:MAX_TOOL_RESULT_CHARS]}..."


async def run_web_search_tool(
    tavily_client: AsyncTavilyClient | None,
    settings: Settings,
    arguments: dict[str, Any],
) -> str:
    if tavily_client is None:
        return tool_result_json(
            {"error": "Tavily is not configured. Set TAVILY_API_KEY to enable web search."}
        )

    query = str(arguments.get("query", "")).strip()
    if not query:
        return tool_result_json({"error": "Missing required search query."})

    search_depth = normalized_choice(
        arguments.get("search_depth"), SEARCH_DEPTHS, "basic"
    )
    topic = normalized_choice(arguments.get("topic"), SEARCH_TOPICS, "general")
    max_results = normalized_max_results(arguments.get("max_results"))
    include_domains = normalized_domain_list(arguments.get("include_domains"))
    exclude_domains = normalized_domain_list(arguments.get("exclude_domains"))

    search_kwargs: dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_answer": "basic",
        "include_favicon": True,
        "include_usage": True,
        "timeout": settings.request_timeout_seconds,
    }

    time_range = arguments.get("time_range")
    if isinstance(time_range, str) and time_range in TIME_RANGES:
        search_kwargs["time_range"] = time_range
    if include_domains:
        search_kwargs["include_domains"] = include_domains
    if exclude_domains:
        search_kwargs["exclude_domains"] = exclude_domains

    try:
        response = await asyncio.wait_for(
            tavily_client.search(**search_kwargs),
            timeout=settings.request_timeout_seconds,
        )
    except Exception as exc:
        LOGGER.exception("Tavily search failed: %s", exc)
        return tool_result_json({"error": f"Tavily search failed: {exc}"})

    return tool_result_json(compact_tavily_response(response))


async def run_tool_call(
    tool_call: Any,
    tavily_client: AsyncTavilyClient | None,
    settings: Settings,
) -> str:
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", "")
    arguments = parse_tool_arguments(getattr(function, "arguments", None))

    if name == "web_search":
        return await run_web_search_tool(tavily_client, settings, arguments)
    return tool_result_json({"error": f"Unknown tool: {name}"})


def log_message_state(
    message: Message, settings: Settings, identity: BotIdentity, source: str
) -> None:
    text = message_text(message).replace("\n", " ")
    preview = text[:120]
    LOGGER.info(
        "%s chat_id=%s chat_type=%s user_id=%s allowed_group=%s direct=%s "
        "prefix=%s allowed_user=%s allowed_private=%s should_answer=%s text=%r",
        source,
        message.chat.id,
        message.chat.type,
        message.from_user.id if message.from_user else None,
        is_allowed_group(message, settings),
        is_direct_bot_interaction(message, settings, identity),
        has_question_prefix(message, settings),
        is_allowed_user_message(message, settings),
        is_allowed_private_chat(message, settings),
        should_answer(message, settings, identity),
        preview,
    )


async def ask_gemini(
    client: AsyncOpenAI,
    settings: Settings,
    question: str,
    tavily_client: AsyncTavilyClient | None,
) -> str:
    tools_enabled = tavily_client is not None
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": model_system_prompt(settings, tools_enabled)},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS + 1):
        create_kwargs: dict[str, Any] = {
            "model": settings.gemini_model,
            "messages": messages,
        }
        if tools_enabled:
            create_kwargs["tools"] = [WEB_SEARCH_TOOL]
            create_kwargs["tool_choice"] = "auto"

        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=settings.request_timeout_seconds,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            answer = message.content
            return answer.strip() if answer else "我暂时没有生成有效回复。"

        messages.append(message.model_dump(exclude_none=True))
        for tool_call in tool_calls:
            LOGGER.info(
                "running_tool name=%s id=%s",
                getattr(getattr(tool_call, "function", None), "name", ""),
                getattr(tool_call, "id", ""),
            )
            tool_content = await run_tool_call(tool_call, tavily_client, settings)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )

    return "我已经完成了网络查询，但暂时没能整理出可靠回复，请稍后再试。"



async def reply_text(message: Message, text: str) -> None:
    first = True
    for chunk in chunks(text):
        try:
            if first:
                await message.reply_text(
                    chunk,
                    do_quote=True,
                    parse_mode=ParseMode.MARKDOWN,
                )
                first = False
            else:
                await message.get_bot().send_message(
                    message.chat_id,
                    chunk,
                    parse_mode=ParseMode.MARKDOWN,
                )
        except BadRequest as exc:
            LOGGER.warning("Markdown reply failed; falling back to plain text: %s", exc)
            if first:
                await message.reply_text(chunk, do_quote=True)
                first = False
            else:
                await message.get_bot().send_message(message.chat_id, chunk)


async def chat_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    await message.reply_text(
        f"chat_id: {message.chat.id}\nchat_type: {message.chat.type}",
        do_quote=True,
    )


async def start_or_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    identity: BotIdentity = context.application.bot_data["identity"]
    if should_deny(message, settings, identity):
        await message.reply_text(settings.no_permission_text, do_quote=True)
        return

    if should_answer(message, settings, identity):
        await message.reply_text("请直接在群里 @我 或回复我的消息提问。", do_quote=True)


async def answer_question(
    message: Message, context: ContextTypes.DEFAULT_TYPE, question: str
) -> None:
    settings: Settings = context.application.bot_data["settings"]
    client: AsyncOpenAI = context.application.bot_data["openai_client"]
    tavily_client: AsyncTavilyClient | None = context.application.bot_data.get(
        "tavily_client"
    )

    if not question:
        await message.reply_text("请发送文本问题。", do_quote=True)
        return

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    LOGGER.info("asking_model chat_id=%s message_id=%s", message.chat_id, message.message_id)

    try:
        answer = await ask_gemini(client, settings, question, tavily_client)
    except (asyncio.TimeoutError, OpenAIError) as exc:
        LOGGER.exception("Gemini request failed: %s", exc)
        await message.reply_text("模型暂时不可用，请稍后再试。", do_quote=True)
        return

    LOGGER.info(
        "model_answered chat_id=%s message_id=%s answer_length=%s",
        message.chat_id,
        message.message_id,
        len(answer),
    )
    try:
        await reply_text(message, answer)
    except TelegramError as exc:
        LOGGER.exception("Telegram reply failed: %s", exc)
        return
    LOGGER.info("reply_sent chat_id=%s message_id=%s", message.chat_id, message.message_id)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled update error: %s", context.error)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    identity: BotIdentity = context.application.bot_data["identity"]
    log_message_state(message, settings, identity, "ask_command")
    if not (is_allowed_group(message, settings) or is_allowed_private_chat(message, settings)):
        await message.reply_text(settings.no_permission_text, do_quote=True)
        return

    await answer_question(message, context, strip_leading_command(message_text(message)))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    settings: Settings = context.application.bot_data["settings"]
    identity: BotIdentity = context.application.bot_data["identity"]
    log_message_state(message, settings, identity, "message")

    if should_deny(message, settings, identity):
        await message.reply_text(settings.no_permission_text, do_quote=True)
        return

    if not should_answer(message, settings, identity):
        return

    question = message_text(message)
    if has_question_prefix(message, settings):
        question = strip_question_prefix(question, settings)
    else:
        question = strip_bot_mention(question, identity.username)
    await answer_question(message, context, question)


async def post_init(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    bot = await application.bot.get_me()
    if not bot.username:
        raise RuntimeError("Telegram bot does not have a username")

    application.bot_data["identity"] = BotIdentity(id=bot.id, username=bot.username)
    application.bot_data["openai_client"] = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_api_base,
    )
    application.bot_data["tavily_client"] = (
        AsyncTavilyClient(api_key=settings.tavily_api_key)
        if settings.tavily_api_key
        else None
    )
    await application.bot.set_my_commands(
        [
            BotCommand("chatid", "显示当前聊天 ID"),
            BotCommand("ask", "向 bot 提问"),
            BotCommand("start", "查看使用提示"),
            BotCommand("help", "查看使用提示"),
        ]
    )

    LOGGER.info("Bot started as @%s", bot.username)
    if not settings.tavily_api_key:
        LOGGER.warning("TAVILY_API_KEY is not configured. Web search tools are disabled.")
    if settings.allowed_chat_id is None:
        LOGGER.warning(
            "ALLOWED_CHAT_ID is not configured. Use /chatid in the target group, "
            "then add that numeric id to .env."
        )


def build_application(settings: Settings) -> Application:
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data["settings"] = settings

    application.add_handler(CommandHandler("chatid", chat_id_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler(["start", "help"], start_or_help_command))
    application.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, handle_message))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = load_settings()
    application = build_application(settings)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
