"""
OpenRouter API Client for multi-model AI integration.

This module provides a unified interface for communicating with the OpenRouter API,
supporting multiple AI models (Claude, GPT, Llama) with streaming responses,
rate limiting, and robust error handling.
"""

import json
import time
import logging
from typing import List, Dict, Any, Iterator, Union, Optional
from dataclasses import asdict
import asyncio
from threading import Lock
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .types import (
    AIConfig, Message, ResponseChunk, AIStatistics,
    OpenRouterAPIError, ModelNotAvailableError, RateLimitExceededError,
    InvalidAPIKeyError, ContextWindowExceededError
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for OpenRouter API calls."""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
        self.lock = Lock()
    
    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        with self.lock:
            current_time = time.time()
            
            # Clean old requests
            self.minute_requests = [t for t in self.minute_requests if current_time - t < 60]
            self.hour_requests = [t for t in self.hour_requests if current_time - t < 3600]
            
            # Check limits (allow up to the limit, not just under it)
            return (len(self.minute_requests) < self.requests_per_minute and 
                    len(self.hour_requests) < self.requests_per_hour)
    
    def record_request(self) -> None:
        """Record a request for rate limiting."""
        with self.lock:
            current_time = time.time()
            self.minute_requests.append(current_time)
            self.hour_requests.append(current_time)
    
    def wait_if_needed(self) -> float:
        """Wait if rate limit would be exceeded. Returns wait time."""
        if not self.can_make_request():
            with self.lock:
                current_time = time.time()
                
                # Calculate wait time based on oldest request
                if len(self.minute_requests) >= self.requests_per_minute:
                    wait_time = 60 - (current_time - self.minute_requests[0])
                    if wait_time > 0:
                        logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                        return wait_time
        
        return 0.0


class OpenRouterClient:
    """
    OpenRouter API client for multi-model AI integration.
    
    Provides unified interface for Claude, GPT, and Llama models
    through the OpenRouter service with streaming support and error handling.
    """
    
    def __init__(self, config: AIConfig):
        """
        Initialize OpenRouter client.
        
        Args:
            config (AIConfig): AI configuration with API key and settings
        """
        self.config = config
        self.current_model = config.default_model
        self.rate_limiter = RateLimiter(config.rate_limit_rpm, config.rate_limit_rph)
        self.statistics = AIStatistics()
        self.session = None
        self.is_initialized = False
        
        logger.info(f"OpenRouter client created with model: {self.current_model}")
    
    def initialize(self) -> bool:
        """
        Initialize the OpenRouter client and validate credentials.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing OpenRouter client...")
            
            # Validate configuration
            if not self.config.validate():
                logger.error("Invalid AI configuration")
                return False
            
            # Create HTTP session with retry strategy
            self.session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.retry_attempts,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Set default headers
            self.session.headers.update(self.config.get_headers())
            
            # Validate API key
            if not self.validate_api_key():
                logger.error("API key validation failed")
                return False
            
            # Load available models
            self._load_available_models()
            
            self.is_initialized = True
            logger.info("OpenRouter client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            return False
    
    def validate_api_key(self) -> bool:
        """
        Validate OpenRouter API key.
        
        Returns:
            bool: True if API key is valid
        """
        try:
            logger.debug("Validating OpenRouter API key...")
            
            # Simple validation request to models endpoint
            response = self.session.get(
                f"{self.config.base_url}/models",
                timeout=10
            )
            
            if response.status_code == 401:
                raise InvalidAPIKeyError("Invalid or expired API key")
            elif response.status_code == 403:
                raise InvalidAPIKeyError("API key lacks required permissions")
            elif response.status_code != 200:
                raise OpenRouterAPIError(f"API validation failed: {response.status_code}")
            
            logger.info("API key validation successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API key validation: {e}")
            raise OpenRouterAPIError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return False
    
    def _load_available_models(self) -> None:
        """Load available models from OpenRouter."""
        try:
            logger.debug("Loading available models...")
            
            response = self.session.get(
                f"{self.config.base_url}/models",
                timeout=10
            )
            
            if response.status_code == 200:
                models_data = response.json()
                logger.info(f"Loaded {len(models_data.get('data', []))} available models")
            else:
                logger.warning(f"Could not load models list: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to load available models: {e}")
    
    def set_model(self, model_name: str) -> None:
        """
        Set the current AI model.
        
        Args:
            model_name (str): Model name (e.g., "claude-3-5-sonnet")
        """
        if model_name != self.current_model:
            logger.info(f"Switching model from {self.current_model} to {model_name}")
            self.current_model = model_name
        else:
            logger.debug(f"Model already set to {model_name}")
    
    def generate_response(self, messages: List[Message], stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Generate AI response from messages.
        
        Args:
            messages (List[Message]): Conversation messages
            stream (bool): Whether to stream the response
            
        Returns:
            Union[str, Iterator[str]]: Complete response or streaming chunks
        """
        if not self.is_initialized:
            raise OpenRouterAPIError("Client not initialized")
        
        start_time = time.time()
        
        try:
            # Check rate limits
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.statistics.rate_limit_hits += 1
            
            # Record request
            self.rate_limiter.record_request()
            self.statistics.total_requests += 1
            
            # Prepare request payload
            payload = self._prepare_request_payload(messages, stream)
            
            logger.debug(f"Sending request to OpenRouter: {self.current_model}, stream={stream}")
            
            # Make API request
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout_seconds,
                stream=stream
            )
            
            # Handle response
            if response.status_code == 200:
                if stream:
                    return self._handle_streaming_response(response, start_time)
                else:
                    return self._handle_complete_response(response, start_time)
            else:
                self._handle_error_response(response)
                
        except requests.exceptions.Timeout:
            self.statistics.error_count += 1
            raise OpenRouterAPIError("Request timeout")
        except requests.exceptions.RequestException as e:
            self.statistics.error_count += 1
            raise OpenRouterAPIError(f"Network error: {e}")
        except Exception as e:
            self.statistics.error_count += 1
            logger.error(f"Unexpected error in generate_response: {e}")
            raise OpenRouterAPIError(f"Unexpected error: {e}")
    
    def _prepare_request_payload(self, messages: List[Message], stream: bool) -> Dict[str, Any]:
        """Prepare the request payload for OpenRouter API."""
        # Convert messages to OpenAI format
        formatted_messages = [msg.to_openai_format() for msg in messages]
        
        # Add system message if not present
        if not any(msg['role'] == 'system' for msg in formatted_messages):
            formatted_messages.insert(0, {
                'role': 'system',
                'content': self.config.system_prompt
            })
        
        payload = {
            'model': self.current_model,
            'messages': formatted_messages,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'stream': stream
        }
        
        logger.debug(f"Request payload prepared: {len(formatted_messages)} messages")
        return payload
    
    def _handle_complete_response(self, response: requests.Response, start_time: float) -> str:
        """Handle complete (non-streaming) response."""
        try:
            response_data = response.json()
            
            # Extract response text
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                
                # Update statistics
                processing_time = time.time() - start_time
                self.statistics.average_response_time = (
                    (self.statistics.average_response_time * (self.statistics.total_requests - 1) + processing_time) /
                    self.statistics.total_requests
                )
                
                # Track token usage
                if 'usage' in response_data:
                    tokens_used = response_data['usage'].get('total_tokens', 0)
                    self.statistics.total_tokens_used += tokens_used
                
                # Track model usage
                model_key = self.current_model
                self.statistics.model_usage_distribution[model_key] = (
                    self.statistics.model_usage_distribution.get(model_key, 0) + 1
                )
                
                logger.debug(f"Response received: {len(content)} characters, {processing_time:.2f}s")
                return content
            else:
                raise OpenRouterAPIError("Invalid response format: no choices")
                
        except json.JSONDecodeError as e:
            raise OpenRouterAPIError(f"Invalid JSON response: {e}")
        except KeyError as e:
            raise OpenRouterAPIError(f"Missing required field in response: {e}")
    
    def _handle_streaming_response(self, response: requests.Response, start_time: float) -> Iterator[str]:
        """Handle streaming response."""
        try:
            full_content = ""
            chunk_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            
                            if 'content' in delta:
                                content_chunk = delta['content']
                                full_content += content_chunk
                                chunk_count += 1
                                
                                logger.debug(f"Streaming chunk {chunk_count}: {len(content_chunk)} chars")
                                yield content_chunk
                    
                    except json.JSONDecodeError:
                        # Skip invalid JSON chunks
                        continue
            
            # Update statistics after streaming completes
            processing_time = time.time() - start_time
            self.statistics.average_response_time = (
                (self.statistics.average_response_time * (self.statistics.total_requests - 1) + processing_time) /
                self.statistics.total_requests
            )
            
            # Track model usage
            model_key = self.current_model
            self.statistics.model_usage_distribution[model_key] = (
                self.statistics.model_usage_distribution.get(model_key, 0) + 1
            )
            
            logger.info(f"Streaming complete: {len(full_content)} chars, {chunk_count} chunks, {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise OpenRouterAPIError(f"Streaming error: {e}")
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from OpenRouter API."""
        self.statistics.error_count += 1
        
        try:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
        except:
            error_message = f"HTTP {response.status_code}"
        
        if response.status_code == 401:
            raise InvalidAPIKeyError(f"Authentication failed: {error_message}")
        elif response.status_code == 429:
            self.statistics.rate_limit_hits += 1
            raise RateLimitExceededError(f"Rate limit exceeded: {error_message}")
        elif response.status_code == 400:
            if 'context' in error_message.lower() or 'token' in error_message.lower():
                raise ContextWindowExceededError(f"Context window exceeded: {error_message}")
            else:
                raise OpenRouterAPIError(f"Bad request: {error_message}")
        elif response.status_code == 404:
            raise ModelNotAvailableError(f"Model not found: {self.current_model}")
        elif response.status_code >= 500:
            raise OpenRouterAPIError(f"Server error: {error_message}")
        else:
            raise OpenRouterAPIError(f"API error {response.status_code}: {error_message}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available AI models.
        
        Returns:
            List[str]: Available model names
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/models",
                timeout=10
            )
            
            if response.status_code == 200:
                models_data = response.json()
                return [model['id'] for model in models_data.get('data', [])]
            else:
                logger.warning(f"Could not fetch models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    def get_statistics(self) -> AIStatistics:
        """
        Get client statistics.
        
        Returns:
            AIStatistics: Performance and usage statistics
        """
        return self.statistics
    
    def shutdown(self) -> None:
        """Shutdown the OpenRouter client."""
        logger.info("Shutting down OpenRouter client...")
        
        if self.session:
            self.session.close()
            self.session = None
        
        self.is_initialized = False
        logger.info("OpenRouter client shutdown complete")
