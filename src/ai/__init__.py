"""
AI conversation package for the Astir Voice Assistant.

This package contains AI conversation services and OpenRouter integration
for natural language processing and conversation management.
"""

# AI Conversation Service
from .openrouter_client import OpenRouterClient
from .conversation_service import AIConversationService, ConversationManager

# Data structures and types
from .types import (
    ConversationRequest, ConversationResponse, ResponseChunk,
    ConversationTurn, ConversationContext, Message,
    AIStatistics, AIConfig, AIEventTypes,
    Speaker, MessageRole, ConversationState, Priority
)

__all__ = [
    # Core Services
    'OpenRouterClient',
    'AIConversationService',
    'ConversationManager',
    
    # Data Types
    'ConversationRequest',
    'ConversationResponse', 
    'ResponseChunk',
    'ConversationTurn',
    'ConversationContext',
    'Message',
    'AIStatistics',
    'AIConfig',
    'AIEventTypes',
    
    # Enums
    'Speaker',
    'MessageRole',
    'ConversationState',
    'Priority'
]
