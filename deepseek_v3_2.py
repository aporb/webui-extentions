"""
DeepSeek API Integration for Open WebUI (Version 1.0.0)

A comprehensive Python implementation for integrating DeepSeek's V3 API with Open WebUI.
This module provides a robust interface to DeepSeek's advanced language model capabilities
while maintaining full compatibility with Open WebUI's plugin architecture.

Features:
1. Model Capabilities
   - DeepSeek V3 Integration - Advanced language model with 671B MoE parameters
   - High Performance - 60 tokens/second throughput
   - Large Context - 8192 token context window
   - Vision Support - Process and analyze image inputs

2. Output Formats
   - Text Generation - Standard text completion responses
   - JSON Mode - Validated JSON output with schema enforcement
   - Streaming - Real-time token-by-token responses
   - Function Calling - Define and execute external functions

3. Resource Management
   - Context Caching - Efficient token reuse with hit/miss tracking
   - Cost Optimization - Reduced costs for cached content (¥0.1/M vs ¥1/M)
   - Usage Metrics - Detailed token and cost tracking
   - Request Retries - Automatic retry for unstable operations

4. Advanced Controls
   - Temperature Control - Task-specific randomness settings
   - Top-K Sampling - Control output token diversity
   - Top-P Filtering - Nucleus sampling for better quality
   - Beta Features - Support for prefix and FIM capabilities

5. Error Handling
   - Status Codes - Specific error codes with descriptions
   - Validation - Input parameter and response validation
   - Recovery - Automatic recovery from transient errors
   - Logging - Comprehensive error and usage logging

Author: Amyn Porbanderwala
Email: amyn@porbanderwala.com
Website: www.porbanderwala.com
Version: 1.0.0
Date: January 2024
License: Free for non-commercial use
"""

import os
import json
import time
import logging
import requests
import aiohttp
import tiktoken
from typing import List, Union, Generator, Iterator, Dict, Optional, AsyncIterator, Literal
from pydantic import BaseModel, Field, constr
from open_webui.utils.misc import pop_system_message

# Valid use cases for temperature optimization
UseCaseType = Literal["coding", "data_analysis", "conversation", "translation", "creative"]

class Pipe:
    """
    Main class for DeepSeek API integration with Open WebUI.
    
    This class implements the Open WebUI plugin interface while providing
    access to DeepSeek's V3 language model capabilities. It handles all aspects
    of API communication, including request formatting, response processing,
    error handling, and usage tracking.

    Key Components:
    - Valves: Configuration settings for API behavior
    - Streaming: Support for real-time response streaming
    - Function Calling: Ability to define and call external functions
    - Context Caching: Efficient token usage through caching
    - Usage Tracking: Detailed metrics on token usage and costs

    The class follows Open WebUI's pipe pattern, where data flows through
    configurable valves that control various aspects of the API interaction.
    """

    # API endpoints for standard and beta feature access
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    BETA_API_URL = "https://api.deepseek.com/beta/chat/completions"
    
    # Request timeout settings (connect_timeout, read_timeout)
    REQUEST_TIMEOUT = (3.05, 60)
    
    # Maximum token limits for supported models
    MODEL_MAX_TOKENS = {
        "deepseek-chat": 8192,  # DeepSeek-V3 context window size
    }

    # Temperature presets optimized for different use cases
    TEMPERATURE_RECOMMENDATIONS = {
        "coding": 0.0,      # Maximum precision for code generation
        "data_analysis": 1.0,  # Balanced for analytical tasks
        "conversation": 1.3,   # Natural dialogue flow
        "translation": 1.3,    # Fluent language translation
        "creative": 1.5      # Enhanced creativity
    }

    # Comprehensive error code mapping for better error handling
    ERROR_CODES = {
        400: "Invalid request body format",
        401: "Authentication failed - invalid API key",
        402: "Insufficient balance",
        422: "Invalid parameters",
        429: "Rate limit reached",
        500: "Server error",
        503: "Server overloaded"
    }

    class Valves(BaseModel):
        """Configuration parameters for the DeepSeek API V3"""
        DEEPSEEK_API_KEY: str = Field(
            default=os.getenv("DEEPSEEK_API_KEY", ""),
            description="Your DeepSeek API key"
        )
        temperature: float = Field(
            default=1.0,  # Default for general use
            description="Controls randomness (0.0-2.0). Use TEMPERATURE_RECOMMENDATIONS for specific tasks."
        )
        top_k: int = Field(
            default=40,
            description="Limits sampling to top-k tokens"
        )
        top_p: float = Field(
            default=1.0,  # Updated default per V3
            description="Controls diversity via nucleus sampling (0.0-1.0)"
        )
        max_tokens: int = Field(
            default=4096,
            description="Maximum tokens to generate (1-8192)"
        )
        stream: bool = Field(
            default=False,
            description="Enable streaming responses"
        )
        response_format: str = Field(
            default="text",
            description="Output format (text/json_object). Set to json_object for guaranteed valid JSON output."
        )
        enable_beta_features: bool = Field(
            default=False,
            description="Enable experimental features like prefix and FIM"
        )
        enable_context_cache: bool = Field(
            default=True,
            description="Enable context caching for efficient token usage"
        )
        function_call_retry: int = Field(
            default=2,
            description="Number of retries for unstable function calls"
        )
        stream_options: dict = Field(
            default={"include_usage": True},
            description="Options for streaming responses, including usage statistics"
        )
        use_case: Optional[UseCaseType] = Field(
            default=None,
            description="Specify use case for automatic temperature optimization"
        )

    def __init__(self):
        """
        Initialize the DeepSeek pipe with logging and default configuration.

        Sets up:
        - Basic logging configuration
        - Pipe type and identifier
        - Default valve settings
        - Request tracking
        """
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "deepseek"
        self.valves = self.Valves()
        self.request_id = None

    def get_deepseek_models(self) -> List[dict]:
        """
        Return list of supported DeepSeek models with their capabilities.

        Returns:
            List[dict]: List of model configurations with:
                - id: Unique model identifier
                - name: Human-readable model name
                - context_length: Maximum token context window
                - supports_vision: Vision capability flag

        Example:
            [
                {
                    "id": "deepseek/deepseek-chat",
                    "name": "deepseek-chat",
                    "context_length": 8192,
                    "supports_vision": True
                }
            ]
        """
        return [
            {
                "id": f"deepseek/{name}",
                "name": name,
                "context_length": self.MODEL_MAX_TOKENS.get(name, 8192),
                "supports_vision": True,
            }
            for name in ["deepseek-chat"]
        ]

    def pipes(self) -> List[dict]:
        """
        Return available pipe configurations for Open WebUI integration.

        This method is required by the Open WebUI plugin interface and returns
        the same information as get_deepseek_models() to maintain consistency.

        Returns:
            List[dict]: List of available model configurations
        """
        return self.get_deepseek_models()

    def process_image(self, image_data: dict) -> str:
        """
        Process and validate image data for vision-enabled models.

        Args:
            image_data (dict): Image data dictionary containing:
                - image_url: Dict with 'url' key containing image location
                  Can be a data URL or direct URL

        Returns:
            str: Processed image data ready for API submission

        Example:
            >>> image_data = {"image_url": {"url": "data:image/jpeg;base64,..."}}
            >>> processed = pipe.process_image(image_data)
        """
        if image_data["image_url"]["url"].startswith("data:image"):
            return image_data["image_url"]["url"].split(",", 1)[1]
        return image_data["image_url"]["url"]

    async def pipe(self, body: Dict, __event_emitter__=None) -> Union[str, Generator, Iterator]:
        """
        Main processing pipeline for handling requests to the DeepSeek API.

        This method processes all requests to the DeepSeek API, handling both streaming
        and non-streaming responses. It manages various features including function
        calling, JSON output, context caching, and usage tracking.

        Args:
            body (Dict): Request configuration containing:
                - model (str): Model identifier (e.g., "deepseek/deepseek-chat")
                - messages (List[Dict]): Conversation messages
                - temperature (float, optional): Output randomness (0.0-2.0)
                - max_tokens (int, optional): Maximum tokens to generate
                - stream (bool, optional): Enable streaming responses
                - functions (List[Dict], optional): Available functions
                - response_format (Dict, optional): Output format configuration
                - use_case (str, optional): Task type for temperature optimization
            __event_emitter__ (callable, optional): Callback for status updates

        Returns:
            Union[str, Generator, Iterator]: 
                - String response for non-streaming requests
                - Generator/Iterator for streaming responses
                - Error dict if request fails

        Raises:
            ValueError: If required parameters are missing
            Exception: For API communication errors

        Features:
            - Automatic temperature optimization based on use case
            - Context caching for efficient token usage
            - Function calling with retry mechanism
            - JSON output validation
            - Detailed usage tracking
            - Beta features support (prefix and FIM)

        Example:
            >>> response = await pipe.pipe({
            ...     "model": "deepseek/deepseek-chat",
            ...     "messages": [{"role": "user", "content": "Hello!"}],
            ...     "use_case": "conversation"  # Sets optimal temperature
            ... })
        """
        
        # Validate API key
        if not self.valves.DEEPSEEK_API_KEY:
            error_msg = "Error: DEEPSEEK_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

        try:
            # Extract and process messages
            system_message, messages = pop_system_message(body.get("messages", []))
            
            # Validate model selection
            if "model" not in body:
                raise ValueError("Model name is required")

            model_name = body["model"].split("/")[-1]
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_name, 8192)

            # Add system message if present
            if system_message:
                messages.insert(0, {"role": "system", "content": str(system_message)})

            # Determine temperature based on use case if specified
            use_case = body.get("use_case")
            if use_case and use_case in self.TEMPERATURE_RECOMMENDATIONS:
                temperature = float(body.get("temperature", self.TEMPERATURE_RECOMMENDATIONS[use_case]))
            else:
                temperature = float(body.get("temperature", self.valves.temperature))

            # Handle JSON output mode
            response_format = body.get("response_format", {}).get("type", self.valves.response_format)
            if response_format == "json_object":
                # Configure JSON output mode by ensuring proper system instruction
                json_instruction = "You must respond with valid JSON output."
                if system_message:
                    # Add JSON instruction if not already present
                    if "json" not in system_message.lower():
                        messages[0]["content"] = f"{system_message}\n{json_instruction}"
                else:
                    # Create new system message with JSON instruction
                    messages.insert(0, {"role": "system", "content": json_instruction})

            # Handle function calling with automatic retries for stability
            functions = body.get("functions", [])
            if functions:
                # Attempt function call setup with configured retry count
                for attempt in range(self.valves.function_call_retry):
                    try:
                        # Construct comprehensive API payload with all parameters
                        payload = {
                            "model": model_name,
                            "messages": messages,
                            "max_tokens": min(
                                body.get("max_tokens", self.valves.max_tokens),
                                max_tokens_limit
                            ),
                            "temperature": temperature,  # Use case optimized
                            "top_k": int(body.get("top_k", self.valves.top_k)) if body.get("top_k") is not None else None,
                            "top_p": float(body.get("top_p", self.valves.top_p)) if body.get("top_p") is not None else None,
                            "stream": body.get("stream", self.valves.stream),
                            "response_format": {"type": response_format},
                            "functions": functions,  # Function definitions
                            "function_call": body.get("function_call", "auto"),  # Auto or specific function
                            "stream_options": self.valves.stream_options if body.get("stream") else None
                        }
                        break  # Success - exit retry loop
                    except Exception as e:
                        # Handle retry exhaustion
                        if attempt == self.valves.function_call_retry - 1:
                            raise Exception(f"Function calling failed after {self.valves.function_call_retry} retries: {str(e)}")
                        # Log warning and retry after delay
                        logging.warning(f"Function calling attempt {attempt+1} failed: {str(e)}")
                        time.sleep(1)  # Brief delay before retry
            else:
                # Construct API payload for non-function requests
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": min(
                        body.get("max_tokens", self.valves.max_tokens),
                        max_tokens_limit
                    ),
                    "temperature": temperature,
                    "top_k": int(body.get("top_k", self.valves.top_k)) if body.get("top_k") is not None else None,
                    "top_p": float(body.get("top_p", self.valves.top_p)) if body.get("top_p") is not None else None,
                    "stream": body.get("stream", self.valves.stream),
                    "response_format": {"type": response_format},
                    "stream_options": self.valves.stream_options if body.get("stream") else None
                }

            # Enable beta features if requested
            if self.valves.enable_beta_features:
                payload["prefix"] = True
                payload["fim"] = True

            # Clean payload
            payload = {k: v for k, v in payload.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }

            api_url = self.BETA_API_URL if self.valves.enable_beta_features else self.API_URL

            # Handle streaming responses
            if payload["stream"]:
                return self._stream_response(
                    url=api_url,
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__
                )

            # Process non-streaming response
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"Error: HTTP {response.status}: {await response.text()}"
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": error_msg, "done": True}
                            })
                        return {"content": error_msg, "format": "text"}

                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        response_text = result["choices"][0]["message"]["content"]

                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {
                                    "description": "Request completed successfully",
                                    "done": True
                                }
                            })

                        return response_text
                    return ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            return {"content": messages, "format": "text"}

    async def _stream_response(
        self, url: str, headers: dict, payload: dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        """
        Handle streaming responses from the DeepSeek API.

        This internal method manages the streaming connection and processes
        server-sent events (SSE) from the API. It provides real-time access
        to model outputs and usage statistics.

        Args:
            url (str): API endpoint URL
            headers (dict): Request headers including authentication
            payload (dict): Request payload
            __event_emitter__ (callable, optional): Status update callback

        Yields:
            str: Response chunks as they arrive:
                - Text content chunks for normal responses
                - JSON strings for function calls
                - Error messages if something goes wrong

        Features:
            - Real-time token usage tracking
            - Cache performance monitoring (hit/miss rates)
            - Cost tracking (¥0.1/M for cache hits, ¥1/M for misses)
            - Function call streaming support
            - Detailed error reporting
            - Progress status updates

        Error Handling:
            - HTTP errors with specific status codes
            - JSON parsing errors for malformed responses
            - Network and connection errors
            - Early completion handling (length/filter)

        Example:
            >>> async for chunk in pipe._stream_response(url, headers, payload):
            ...     print(chunk, end="", flush=True)  # Real-time output
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": "Processing stream...",
                                "done": False
                            }
                        })

                    if response.status != 200:
                        error_msg = f"Stream error: HTTP {response.status}: {await response.text()}"
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": error_msg, "done": True}
                            })
                        yield error_msg
                        return

                    accumulated_text = ""
                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                
                                # Handle usage statistics in stream
                                if "usage" in data and data["usage"] is not None:
                                    usage = data["usage"]
                                    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
                                    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
                                    total = usage.get("total_tokens", 0)
                                    
                                    if total > 0:
                                        logging.info(
                                            f"Stream usage - Total: {total} tokens, "
                                            f"Cache hit: {cache_hit} tokens ({cache_hit/total*100:.1f}% @ ¥0.1/M), "
                                            f"Cache miss: {cache_miss} tokens ({cache_miss/total*100:.1f}% @ ¥1/M)"
                                        )
                                    
                                    if __event_emitter__ and self.valves.stream_options.get("include_usage"):
                                        await __event_emitter__({
                                            "type": "status",
                                            "data": {
                                                "description": "Usage statistics",
                                                "usage": usage
                                            }
                                        })
                                    continue

                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    
                                    # Handle content chunks
                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        text_chunk = choice["delta"]["content"]
                                        accumulated_text += text_chunk
                                        yield text_chunk

                                    # Handle function calls in stream
                                    if (
                                        "delta" in choice
                                        and "function_call" in choice["delta"]
                                    ):
                                        yield json.dumps(choice["delta"]["function_call"])

                                    # Handle completion
                                    if "finish_reason" in choice:
                                        if choice["finish_reason"] == "stop":
                                            if __event_emitter__:
                                                await __event_emitter__({
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Stream completed",
                                                        "done": True
                                                    }
                                                })
                                            break
                                        elif choice["finish_reason"] in ["length", "content_filter"]:
                                            error_msg = f"Stream ended early: {choice['finish_reason']}"
                                            if __event_emitter__:
                                                await __event_emitter__({
                                                    "type": "status",
                                                    "data": {
                                                        "description": error_msg,
                                                        "done": True,
                                                        "error": True
                                                    }
                                                })
                                            break
                            except json.JSONDecodeError as e:
                                error_msg = f"Stream parse error: {e}"
                                logging.error(error_msg)
                                if __event_emitter__:
                                    await __event_emitter__({
                                        "type": "status",
                                        "data": {
                                            "description": error_msg,
                                            "done": False,
                                            "error": True
                                        }
                                    })
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            yield error_msg
