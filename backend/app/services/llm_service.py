"""LLM服务模块 (LangChain 版本)"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
import os
from typing import Optional
from ..config import settings

# 全局 LLM 实例
_llm_instance: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """获取 LangChain LLM 实例（单例模式）"""
    global _llm_instance

    if _llm_instance is None:
        # 从环境变量读取配置
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4")

        # 验证必要的配置
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未配置（LangChain 必需）")

        # 创建 ChatOpenAI 实例（使用 config 中的温度和超时配置）
        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=settings.agent_temperature,
            max_tokens=4096,
            timeout=settings.agent_timeout,
            max_retries=2
        )

        print(f"[SUCCESS] LangChain LLM 初始化成功")
        print(f"   模型: {model}")
        print(f"   Base URL: {base_url}")
        print(f"   温度: {settings.agent_temperature}")
        print(f"   超时: {settings.agent_timeout}秒")

    return _llm_instance


def reset_llm():
    """重置 LLM 实例（用于测试或重新配置）"""
    global _llm_instance
    _llm_instance = None

