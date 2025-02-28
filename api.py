import secrets
import time
import uuid
import hashlib
import json
import re
import httpx
import asyncio
from typing import AsyncGenerator, List, Dict, Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from collections import defaultdict
import logging

app = FastAPI()

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False


class ConversationManager:
    def __init__(self):
        self.conversations = defaultdict(lambda: {
            'id': None,
            'last_active': 0,
            'messages': []
        })
        self.expire_time = 3600  # 会话过期时间（秒）

    def get_conversation(self, user_id):
        conv = self.conversations[user_id]
        current_time = time.time()
        
        # 如果会话过期，创建新会话
        if current_time - conv['last_active'] > self.expire_time:
            conv['id'] = str(uuid.uuid4())
            conv['messages'] = []
        
        conv['last_active'] = current_time
        return conv

    def update_conversation(self, user_id, message):
        conv = self.get_conversation(user_id)
        conv['messages'].append(message)
        return conv['id']

# 创建会话管理器实例
conversation_manager = ConversationManager()

class Pipe:
    API_DOMAIN = "https://ai-api.dangbei.net"
    data_prefix = "data:"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

    async def get_sts(self, device_id):
        """获取 STS 令牌"""
        url = f"{self.API_DOMAIN}/ai-search/fileApi/v1/sts"
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        payload = {}
        sign = self._generate_sign(timestamp, payload, nonce)
        
        headers = self._get_headers(device_id, timestamp, nonce, sign)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            return response.json()

    async def create_conversation(self, device_id):
        """创建新会话"""
        url = f"{self.API_DOMAIN}/ai-search/conversationApi/v1/create"
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        
        payload = {
            "botCode": "AI_SEARCH",
            "metaData": {
                "writeCode": ""
            }
        }
        
        sign = self._generate_sign(timestamp, payload, nonce)
        headers = self._get_headers(device_id, timestamp, nonce, sign)

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            data = response.json()
            if data.get("success"):
                return data["data"]["conversationId"]
            return None

    async def get_conversation_list(self, device_id):
        """获取会话列表"""
        url = f"{self.API_DOMAIN}/ai-search/conversationApi/v1/list"
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        payload = {}
        sign = self._generate_sign(timestamp, payload, nonce)
        
        headers = self._get_headers(device_id, timestamp, nonce, sign)

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            return response.json()

    def _get_headers(self, device_id, timestamp, nonce, sign):
        """生成通用请求头"""
        return {
            "Origin": "https://ai.dangbei.com",
            "Referer": "https://ai.dangbei.com/",
            "User-Agent": self.user_agent,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": sign,
            "timestamp": timestamp,
            "Content-Type": "application/json"
        }

    async def pipe(self, body: dict) -> AsyncGenerator[Dict, None]:
        try:
            device_id = self._generate_device_id()
            user_id = body.get('user_id', 'default_user')
            
            logger.info(f"收到用户请求 - user_id: {user_id}")
            logger.info(f"请求内容: {body['messages'][-1]['content']}")
            
            # 获取 STS 令牌
            sts_result = await self.get_sts(device_id)
            logger.info(f"STS 令牌获取结果: {sts_result}")
            
            # 创建新会话
            conversation_id = await self.create_conversation(device_id)
            if not conversation_id:
                logger.error("创建会话失败")
                yield {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "抱歉，创建会话失败，请稍后再试。"
                        },
                        "finish_reason": "error"
                    }]
                }
                return

            logger.info(f"会话ID: {conversation_id}")
            user_message = body["messages"][-1]["content"]
            
            # 发送聊天请求
            payload = {
                "stream": True,
                "botCode": "AI_SEARCH",
                "question": user_message,
                "userAction": "deep",
                "model": "deepseek",
                "chatOption": None,
                "files": [],
                "reference": [],
                "conversationId": conversation_id
            }

            timestamp = str(int(time.time()))
            nonce = self._nanoid(21)
            sign = self._generate_sign(timestamp, payload, nonce)
            
            headers = self._get_headers(device_id, timestamp, nonce, sign)
            headers["Accept"] = "text/event-stream"

            api_url = f"{self.API_DOMAIN}/ai-search/chatApi/v1/chat"
            logger.info(f"发送请求到 API - headers: {headers}")
            logger.info(f"请求 payload: {payload}")

            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream("POST", api_url, 
                                           json=payload, 
                                           headers=headers, 
                                           timeout=1200) as response:
                        if response.status_code != 200:
                            error_content = await response.aread()
                            error_text = error_content.decode('utf-8')
                            logger.error(f"API 请求失败: HTTP {response.status_code}")
                            logger.error(f"错误响应: {error_text}")
                            yield {
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "抱歉，我现在遇到了一些问题，请稍后再试。"
                                    },
                                    "finish_reason": "error"
                                }]
                            }
                            return

                        async for line in response.aiter_lines():
                            if not line.startswith(self.data_prefix):
                                continue

                            json_str = line[len(self.data_prefix):]
                            logger.debug(f"收到响应: {json_str}")

                            try:
                                data = json.loads(json_str)
                                if data.get("type") == "answer":
                                    content = data.get("content")
                                    if content:
                                        logger.debug(f"处理响应内容: {content}")
                                        yield {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {str(e)}")
                                continue

                except httpx.RequestError as e:
                    logger.error(f"请求错误: {str(e)}")
                    yield {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "抱歉，网络连接出现问题，请稍后再试。"
                            },
                            "finish_reason": "error"
                        }]
                    }

        except Exception as e:
            logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
            yield {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "抱歉，系统出现了问题，请稍后再试。"
                    },
                    "finish_reason": "error"
                }]
            }

    def _format_error(self, status_code: int, error: bytes) -> str:
        error_str = error.decode(errors="ignore") if isinstance(error, bytes) else error
        return json.dumps({"error": f"HTTP {status_code}: {error_str}"}, ensure_ascii=False)

    def _format_exception(self, e: Exception) -> str:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False)

    def _nanoid(self, size=21) -> str:
        url_alphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"
        random_bytes = secrets.token_bytes(size)
        return "".join([url_alphabet[b & 63] for b in reversed(random_bytes)])

    def _generate_device_id(self) -> str:
        return f"{uuid.uuid4().hex}_{self._nanoid(20)}"

    def _generate_sign(self, timestamp: str, payload: dict, nonce: str) -> str:
        payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        sign_str = f"{timestamp}{payload_str}{nonce}"
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()


pipe = Pipe()


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """
    OpenAI API 兼容的 Chat 端点
    """

    async def response_generator():
        async for chunk in pipe.pipe(request.dict()):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    content = ""
    error_message = None
    
    async for chunk in pipe.pipe(request.model_dump()):
        if "choices" in chunk and chunk["choices"]:
            if "delta" in chunk["choices"][0]:
                content += chunk["choices"][0]["delta"]["content"]
            elif "message" in chunk["choices"][0]:
                error_message = chunk["choices"][0]["message"]["content"]
                break

    if error_message:
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": error_message
                },
                "finish_reason": "error"
            }],
        }
    
    parts = content.split("\n\n\n", 1)
    if len(parts) > 1:
        reasoning_content = parts[0]
        actual_content = parts[1]
    else:
        reasoning_content = ""
        actual_content = content

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": actual_content
            },
            "finish_reason": "stop"
        }],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
