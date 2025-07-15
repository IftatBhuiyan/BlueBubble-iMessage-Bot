#!/usr/bin/env python3
"""
Example configuration file for iMessage AI Bot
Copy this file to config.py and customize the values
"""

# LLM Configuration
DEFAULT_MODEL = "llama2"  # or "gemma3-abliterated", "llama3", "codellama", etc.
DEFAULT_BASE_URL = "http://localhost:11434"  # Ollama default URL
DEFAULT_API_TYPE = "ollama"  # or "openai"
DEFAULT_POLL_INTERVAL = 3  # seconds between message checks

# Bot Configuration
BOT_NAME = "YourBotName"  # What your bot should be called
YOUR_LOCATION = "Your City, Your State"  # Bot's location for context

# Chat Settings
WARMUP_PERIOD = 2  # Number of polling cycles to wait before responding
AUTO_RESPOND = True  # Whether to automatically respond to messages
CONTEXT_LIMIT = 30  # Number of messages to keep in context

# Response Settings
RESPONSE_TIMEOUT = 120  # Seconds to wait for LLM response
TEMPERATURE = 0.8  # LLM temperature (0.0 to 1.0, higher = more creative)
MAX_TOKENS = 1000  # Maximum tokens in response

# System Prompt Template
SYSTEM_PROMPT_TEMPLATE = """CONTEXT: You are a real person living in {location}. Today is {current_date} and it's currently {current_time}

YOUR NAME: Your name is "{bot_name}" - that's what people call you. Respond naturally when people use your name.

You can be in individual chats or group chats. Figure out who you're talking to and what type of chat from the conversation context.

FRIEND MODE - CUSTOMIZE AS NEEDED:

1. Be conversational and natural
2. Keep responses concise (3-8 words usually)
3. Match your friends' energy
4. Respond to questions and direct messages
5. Use your judgment on when to participate

EXAMPLES:
Individual chat:
"you busy?" → "not really, whats up"
"want to come over?" → "yeah sounds good, what time"

Group chat:
"anyone free tonight?" → "yeah I'm free, what's the plan"
"who wants to get food?" → "I'm down, where were you thinking"

Remember: You're a helpful friend. Respond naturally to whatever comes up.
"""

# Chat Identifiers to Monitor (examples)
SPECIFIC_CHATS = [
    # Add your specific chat identifiers here
    # "+1234567890",  # Phone number
    # "chat123456789",  # Group chat ID
    # "Friend Name",  # Contact name
]

# Ollama Model Settings
OLLAMA_OPTIONS = {
    "temperature": TEMPERATURE,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": MAX_TOKENS,
    "repeat_penalty": 1.1,
    "seed": -1,
    "num_predict": MAX_TOKENS,
    "stop": [],
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

# OpenAI-compatible API Settings (if using)
OPENAI_API_KEY = "your-api-key-here"
OPENAI_BASE_URL = "https://api.openai.com"
OPENAI_MODEL = "gpt-3.5-turbo"

# Debug Settings
DEBUG_MODE = False  # Enable debug output
LOG_CONVERSATIONS = False  # Log conversations to file
LOG_FILE = "bot_conversations.log"

# Safety Settings
MAX_RESPONSE_LENGTH = 500  # Maximum characters in response
BLOCKED_WORDS = []  # Words that should not trigger responses
ALLOWED_SENDERS = []  # If set, only these senders can trigger responses (empty = all allowed)

# Advanced Settings
INTERRUPTION_ENABLED = True  # Enable interruption system
CONVERSATION_MEMORY_ENABLED = True  # Enable conversation memory
SMART_RESPONSE_LOGIC = True  # Enable smart response decision making 