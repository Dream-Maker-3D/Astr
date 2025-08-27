"""
Conversational Intelligence System.

This module implements advanced conversational AI capabilities including natural language
patterns, response brevity, personality consistency, and context-aware conversation
for truly human-like voice interactions.
"""

import re
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ResponseStyle(Enum):
    """Response style options for conversational intelligence."""
    CONCISE = auto()        # Brief, to-the-point responses
    DETAILED = auto()       # Comprehensive explanations
    CASUAL = auto()         # Relaxed, informal tone
    FORMAL = auto()         # Professional, structured tone
    CONVERSATIONAL = auto() # Natural, human-like responses


class PersonalityTrait(Enum):
    """Personality traits for consistent conversational behavior."""
    FRIENDLY = auto()       # Warm, approachable responses
    HELPFUL = auto()        # Solution-oriented, supportive
    CURIOUS = auto()        # Asks follow-up questions
    PATIENT = auto()        # Understanding, non-judgmental
    ENTHUSIASTIC = auto()   # Energetic, positive responses


@dataclass
class ConversationalConfig:
    """Configuration for conversational intelligence behavior."""
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    personality_traits: List[PersonalityTrait] = field(default_factory=lambda: [
        PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL
    ])
    max_response_length: int = 100  # Maximum tokens for concise responses
    use_contractions: bool = True   # Use natural contractions ("I'm", "you're")
    use_acknowledgments: bool = True # Use conversational acknowledgments
    formality_level: float = 0.3    # 0.0 = very casual, 1.0 = very formal
    context_awareness: bool = True   # Reference previous conversation
    personality_consistency: bool = True # Maintain consistent personality


@dataclass
class ConversationContext:
    """Extended conversation context for intelligent responses."""
    recent_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_tone: str = "neutral"
    user_name: Optional[str] = None
    relationship_level: str = "new"  # new, familiar, close
    conversation_history_summary: str = ""


class ConversationalIntelligence:
    """
    Handles conversational intelligence for natural, human-like AI responses.
    
    Provides response optimization, natural language patterns, personality consistency,
    and context-aware conversation management for seamless voice interactions.
    """
    
    def __init__(self, config: ConversationalConfig = None):
        """
        Initialize conversational intelligence system.
        
        Args:
            config: Configuration for conversational behavior
        """
        self.config = config or ConversationalConfig()
        self.context = ConversationContext()
        
        # Natural language patterns
        self._contractions = {
            "I am": "I'm", "you are": "you're", "we are": "we're", "they are": "they're",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't", "were not": "weren't",
            "have not": "haven't", "has not": "hasn't", "had not": "hadn't",
            "will not": "won't", "would not": "wouldn't", "could not": "couldn't",
            "should not": "shouldn't", "cannot": "can't", "do not": "don't", "does not": "doesn't",
            "did not": "didn't", "let us": "let's", "that is": "that's", "it is": "it's",
            "there is": "there's", "here is": "here's", "what is": "what's", "where is": "where's"
        }
        
        # Conversational acknowledgments
        self._acknowledgments = {
            "agreement": ["mm-hmm", "right", "exactly", "I see", "absolutely", "definitely"],
            "understanding": ["I get it", "I understand", "that makes sense", "I follow you"],
            "transition": ["so", "well", "anyway", "by the way", "speaking of which"],
            "thinking": ["let me think", "hmm", "well", "you know", "actually"],
            "clarification": ["what I mean is", "in other words", "to put it simply"]
        }
        
        # Response starters for natural flow
        self._response_starters = {
            "casual": ["Well", "So", "Actually", "You know", "I think", "Honestly"],
            "formal": ["Certainly", "Indeed", "Of course", "I believe", "It appears"],
            "friendly": ["Oh", "Ah", "Great question", "That's interesting", "I'd say"]
        }
        
        logger.info("Conversational Intelligence initialized")
    
    def optimize_response(self, raw_response: str, conversation_context: Dict[str, Any] = None) -> str:
        """
        Optimize AI response for natural conversation.
        
        Args:
            raw_response: Original AI response text
            conversation_context: Current conversation context
            
        Returns:
            str: Optimized conversational response
        """
        try:
            # Update context if provided
            if conversation_context:
                self._update_context(conversation_context)
            
            # Apply conversational optimizations
            optimized = raw_response
            
            # 1. Apply response brevity
            if self.config.response_style == ResponseStyle.CONCISE:
                optimized = self._make_concise(optimized)
            
            # 2. Apply natural language patterns
            if self.config.use_contractions:
                optimized = self._apply_contractions(optimized)
            
            # 3. Add conversational elements
            if self.config.use_acknowledgments:
                optimized = self._add_conversational_elements(optimized)
            
            # 4. Ensure natural flow
            optimized = self._ensure_natural_flow(optimized)
            
            logger.debug(f"Response optimized: '{raw_response[:50]}...' -> '{optimized[:50]}...'")
            return optimized.strip()
            
        except Exception as e:
            logger.error(f"Error optimizing response: {e}")
            return raw_response  # Return original if optimization fails
    
    def _make_concise(self, response: str) -> str:
        """Make response more concise and conversational."""
        # Remove verbose AI assistant phrases
        verbose_patterns = [
            r"I'd be happy to help you with that\.?\s*",
            r"As an AI assistant,?\s*",
            r"I'm here to help\.?\s*",
            r"Let me help you with that\.?\s*",
            r"I'll do my best to\.?\s*",
            r"I understand you're asking about\.?\s*",
            r"Based on your question,?\s*",
            r"To answer your question,?\s*"
        ]
        
        for pattern in verbose_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        
        # Simplify overly complex sentences
        response = re.sub(r"In order to", "To", response)
        response = re.sub(r"It is important to note that", "", response)
        response = re.sub(r"Please be aware that", "", response)
        response = re.sub(r"I would like to inform you that", "", response)
        
        # Remove redundant phrases
        response = re.sub(r"\s+", " ", response)  # Multiple spaces
        response = response.strip()
        
        return response
    
    def _apply_contractions(self, response: str) -> str:
        """Apply natural contractions to make speech more casual."""
        if not self.config.use_contractions:
            return response
        
        for full_form, contraction in self._contractions.items():
            # Case-insensitive replacement with proper capitalization
            pattern = r'\b' + re.escape(full_form) + r'\b'
            response = re.sub(pattern, contraction, response, flags=re.IGNORECASE)
            
            # Handle capitalized versions
            if full_form[0].isupper():
                cap_contraction = contraction.capitalize()
                cap_pattern = r'\b' + re.escape(full_form.capitalize()) + r'\b'
                response = re.sub(cap_pattern, cap_contraction, response)
        
        return response
    
    def _add_conversational_elements(self, response: str) -> str:
        """Add natural conversational elements and acknowledgments."""
        if not self.config.use_acknowledgments:
            return response
        
        # Add occasional acknowledgments based on response type
        if random.random() < 0.3:  # 30% chance to add acknowledgment
            if "?" in response:  # Question response
                ack = random.choice(self._acknowledgments["understanding"])
                response = f"{ack}. {response}"
            elif len(response.split()) > 15:  # Longer response
                ack = random.choice(self._acknowledgments["thinking"])
                response = f"{ack}, {response.lower()}"
        
        return response
    
    def _ensure_natural_flow(self, response: str) -> str:
        """Ensure natural conversational flow and transitions."""
        # Add natural response starters occasionally
        if random.random() < 0.25:  # 25% chance
            formality = "casual" if self.config.formality_level < 0.5 else "formal"
            if formality == "casual" and PersonalityTrait.FRIENDLY in self.config.personality_traits:
                formality = "friendly"
            
            starter = random.choice(self._response_starters.get(formality, ["Well"]))
            
            # Only add if response doesn't already start naturally
            first_word = response.split()[0] if response.split() else ""
            if first_word.lower() not in ["well", "so", "actually", "oh", "ah", "hmm"]:
                response = f"{starter}, {response.lower()}"
        
        # Ensure proper capitalization
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        return response
    
    def _update_context(self, conversation_context: Dict[str, Any]):
        """Update conversation context for better responses."""
        # Extract topics from recent conversation
        if "recent_turns" in conversation_context:
            for turn in conversation_context["recent_turns"]:
                content = turn.get("content", "")
                if content and len(content.split()) > 3:
                    # Extract key topics (simple keyword extraction)
                    words = content.lower().split()
                    topics = [w for w in words if len(w) > 4 and w.isalpha()]
                    if topics:
                        topic = " ".join(topics[:2])  # Take first 2 significant words
                        if topic not in self.context.recent_topics:
                            self.context.recent_topics.append(topic)
        
        # Limit topic history
        if len(self.context.recent_topics) > 5:
            self.context.recent_topics = self.context.recent_topics[-3:]
    
    def generate_system_prompt(self) -> str:
        """
        Generate system prompt for AI based on conversational configuration.
        
        Returns:
            str: Optimized system prompt for natural conversation
        """
        personality_desc = []
        for trait in self.config.personality_traits:
            if trait == PersonalityTrait.FRIENDLY:
                personality_desc.append("friendly and warm")
            elif trait == PersonalityTrait.HELPFUL:
                personality_desc.append("helpful and supportive")
            elif trait == PersonalityTrait.ENTHUSIASTIC:
                personality_desc.append("enthusiastic and positive")
            elif trait == PersonalityTrait.PATIENT:
                personality_desc.append("patient and understanding")
        
        personality_str = ", ".join(personality_desc) if personality_desc else "conversational"
        formality = "casual and natural" if self.config.formality_level < 0.5 else "professional but approachable"
        
        prompt = f"""You are having a natural spoken conversation with someone. Be {personality_str} in your responses.

Key guidelines:
- Keep responses concise and conversational (under {self.config.max_response_length} words typically)
- Use natural speech patterns with contractions when appropriate
- Don't mention that you're an AI assistant
- Respond like you're talking to a friend or colleague
- Be {formality} in tone
- If interrupted, don't reference the interruption - just respond naturally
- For corrections, acknowledge and adjust without making it awkward
- Ask follow-up questions when appropriate to keep conversation flowing

Remember: This is a voice conversation, so respond as you would speak naturally."""

        return prompt
