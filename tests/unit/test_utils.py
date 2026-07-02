"""
Verify logger and LLM client work correctly in 
mock mode before any agent or config code exists.

Run:
    pytest tests/unit/test_utils.py
"""

import logging
import os
import sys
from unittest import result
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.logger import get_logger
from utils.llm_client import LLMClient, LLMResponse

class TestLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
    
    def test_logger_name_is_set(self):
        logger = get_logger("finadvice.test")
        assert logger.name == "finadvice.test"

    def test_same_name_returns_same_instance(self):
        a = get_logger("shared.module")
        b = get_logger("shared.module")
        assert a is b
    
    def test_different_names_return_different_instances(self):
        a = get_logger("module.one")
        b = get_logger("module.two")
        assert a is not b

    def test_logger_has_exactly_one_handler(self):
        get_logger("handler.test")
        logger = get_logger("handler.test")
        assert len(logger.handlers) == 1

    def test_logger_does_not_propagate(self):
        logger = get_logger("propagate.test")
        assert logger.propagate is False

    def test_log_levels_available(self):
        logger = get_logger("level.test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
class TestLLMClientMock:
    def setup_method(self):
        os.environ.pop("OPENAI_API_KEY", None)
        self.client = LLMClient()

    def test_initialises_in_mock_mode_without_api_key(self):
        assert self.client._mode == "mock"
    
    def test_chat_returns_llm_response_type(self):
        result = self.client.chat(
            system="You are a test agent.",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert isinstance(result, LLMResponse)

    def test_mock_response_has_content(self):
        result = self.client.chat(
            system="You are a financial advisor.",
            messages=[{"role": "user", "content": "What is a bond?"}],
        )
        assert isinstance(result.content, str)
        assert len(result.content) > 0
    
    def test_mock_response_tokens_are_zero(self):
        result = self.client.chat(
            system="System prompt.",
            messages=[{"role": "user", "content": "Query"}],
        )
        assert result.tokens_used == 0

    def test_mock_response_model_is_mock(self):
        result = self.client.chat(
            system="System.",
            messages=[{"role": "user", "content": "Query"}],
        )
        assert result.model == "mock"

    def test_mock_content_contains_mock_label(self):
        result = self.client.chat(
            system="You are a risk profiling agent.",
            messages=[{"role": "user", "content": "Assess my risk"}],
        )
        
    def test_empty_messages_list_does_not_crash(self):
        result = self.client.chat(
            system="System.",
            messages=[],
        )
        assert isinstance(result, LLMResponse)

    def test_multi_turn_messages_do_not_crash(self):
        messages = [
            {"role": "user", "content": "What is my risk profile?"},
            {"role": "assistant", "content": "I need more information."},
            {"role": "user", "content": "I am 35, moderate risk."},
        ]
        result = self.client.chat(
            system="System.",
            messages=messages,
        )
        assert isinstance(result, LLMResponse)

    def test_temprature_override_accepted(self):
        result = self.client.chat(
            system="System.",
            messages=[{"role": "user", "content": "Query"}],
            temprature=0.0,
        )
        assert isinstance(result, LLMResponse)

    def test_model_default_is_set(self):
        assert self.client.model == "gpt-4o-mini"

    def test_model_override_accepted(self):
        client = LLMClient(model="gpt-4o")
        assert client.model == "gpt-4o"

    def test_repr_contains_model_and_mode(self):
        r = repr(self.client)
        assert "mock" in r
        assert "gpt-4o-mini" in r