# iMessage AI Bot

An intelligent AI bot that can monitor and respond to iMessage conversations using local LLM models. The bot can work with individual chats or group conversations and provides natural, context-aware responses.

## Features

- **🤖 AI-Powered Responses**: Uses local LLM models (via Ollama) for intelligent conversation
- **📱 iMessage Integration**: Direct integration with macOS Messages app
- **🔄 Real-time Monitoring**: Continuous monitoring of messages with configurable polling intervals
- **🧠 Smart Response Logic**: Intelligent decision-making on when to respond based on context
- **⚡ Interruption Handling**: Advanced interruption system for dynamic conversation flow
- **👥 Multi-Chat Support**: Works with both individual and group conversations
- **🔥 Warm-up Period**: Configurable startup period to avoid responding to old messages
- **📊 Conversation Memory**: Maintains context across conversations for better responses
- **🎯 Customizable Prompts**: Fully customizable system prompts for different personalities
- **🔧 Flexible Configuration**: Support for different LLM models and API endpoints

## Requirements

- **macOS**: This bot only works on macOS due to iMessage integration
- **Python 3.7+**: Required for running the bot
- **Ollama**: For local LLM model hosting
- **Messages App**: Must be signed into iMessage

## iMessage Bot Setup Guide

A comprehensive checklist to create a dedicated, blue-bubble iMessage "bot" account on macOS and verify end-to-end connectivity.

### 1. Create & Configure Your Bot Apple ID
1. **Sign up** for a new Apple ID at https://appleid.apple.com—use it exclusively for your bot.
2. **Enable Two-Factor Authentication** (required for iMessage).
3. Under **"Reachable At"**, add and verify an email address (no phone number needed).

### 2. Sign Into & Activate iMessage (Data-Only, No SMS)

#### On an iPhone or iPad
1. Open **Settings → Messages → Send & Receive**.
2. Sign in with your bot Apple ID.
3. Under **"You Can Be Reached By iMessage At"**, check your bot's email.
4. Send a few test messages between that bot email and another iMessage account (e.g. your personal Apple ID).

#### On macOS
> **💡 Important**: For best results, consider using a **second Mac** or creating a **new macOS user account** with admin privileges. This prevents conflicts with your personal iMessage account and ensures clean separation.

1. **If using a new Mac user**:
   - Create a new user account with admin privileges
   - Sign in to this new user account
   - This ensures complete isolation from your personal iMessage setup

2. **Configure iMessage**:
   - Open **Messages → Settings → iMessage**.
   - Sign in with the *same* bot Apple ID.
   - Confirm your bot's email is listed and checked under **"You Can Be Reached By iMessage At."**
   - In the macOS Messages app, send a test to your personal Apple ID—ensure you see a blue-bubble iMessage.

3. **Confirm Two-Way Blue-Bubble Chat**:
   - From your personal device, send a message to the bot's email-only address.
   - Reply from the Mac UI and verify it arrives as a blue-bubble iMessage.

### 3. Enable & Test Manual iMessage in the UI
1. In the macOS Messages app, send and receive a few more back-and-forths to prove reliability.
2. Make sure group chats, emojis, and attachments still behave as expected.

### 4. Important Contact & Group Management
1. **Add Contacts**: After sending messages to people, **add them as contacts** through the iMessage app on Mac. This ensures reliable chat identification.
2. **Name Group Chats**: For group conversations, it's **highly recommended** to give groups proper names. Unnamed groups will fall back to participant-based identifiers which can be unreliable.

### 5. Wire Up Your Automation Script
1. Copy your sample AppleScript (e.g. `send_imessage()` in Python) into the project.
2. Update to target your bot's service & buddy, for example:
   ```applescript
   tell application "Messages"
     set targetService to 1st service whose service type = iMessage
     set targetBuddy   to buddy "bot_email@domain.com" of targetService
     send "Hello from my bot!" to targetBuddy
   end tell
   ```
3. Run it once on the command line:
   ```bash
   osascript send_test_message.applescript
   ```
   —you should see that text arrive in Messages.

### 6. Verify Incoming-Message Hooks
1. Run your "read" script (either via AppleScript or by querying `~/Library/Messages/chat.db`).
2. Confirm it logs every test message you send from your phone or another iMessage client.
3. Iterate until parsing and filtering work reliably.

### 7. Integrate Your AI Loop
1. Plug `send_imessage()` and `read_recent_messages()` into your bot's main loop.
2. Add console logs for "📨 Received…" and "✅ Sent…".
3. Launch (`python run_dynamic_bot.py`) and watch your bot reply automatically in real time.

### 8. Final Smoke-Tests
- Send varied content (text, emojis, attachments, group-chat pings) to confirm graceful handling.
- Monitor logs for errors or unhandled chat identifiers.
- Tweak timeouts and error handling until the bot runs rock-solid overnight.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/imessage-bot.git
   cd imessage-bot
   ```

2. **Install Python dependencies**:
   ```bash
   pip install requests
   ```

3. **Install and set up Ollama**:
   ```bash
   # Install Ollama
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in a new terminal)
   ollama pull llama2
   # or for better performance with fewer restrictions:
   ollama pull gemma3-abliterated
   ```

4. **Configure Messages app**:
   - Open Messages app and sign in to iMessage
   - Grant necessary permissions when prompted
   - Test sending/receiving messages manually

## Configuration

### 1. Basic Configuration

Edit the model and settings in `run_dynamic_bot.py`:

```python
DEFAULT_MODEL = "llama2"  # or "gemma3-abliterated" for better performance
DEFAULT_POLL_INTERVAL = 3  # seconds between message checks
```

### 2. Custom System Prompt

Modify the `SYSTEM_PROMPT` in `run_dynamic_bot.py` to customize the bot's personality:

```python
SYSTEM_PROMPT = f"""CONTEXT: You are a helpful AI assistant. Today is {current_date} and it's currently {current_time_str}

YOUR NAME: Your name is "YourBotName" - that's what people call you.

[Add your custom instructions here]
"""
```

### 3. Bot Name Configuration

Replace `"YourBotName"` and `"yourbot"` throughout the code with your preferred bot name.

## Usage

### Quick Start

Run the dynamic bot with interactive setup:

```bash
python run_dynamic_bot.py
```

This will present you with options to:
1. Monitor ALL chats (respond to any new message)
2. Monitor specific chats (choose which ones)
3. List available chats first
4. Configure warm-up period

### Manual Usage

#### 1. List Available Chats

```python
from imessage import list_available_chats
list_available_chats()
```

#### 2. Monitor a Specific Chat

```python
from imessage import create_ai_bot, monitor_chat_with_ai

# Create bot for a specific chat
llm_client, system_prompt = create_ai_bot('PHONE_NUMBER_OR_CHAT_ID')

# Start monitoring
monitor_chat_with_ai('PHONE_NUMBER_OR_CHAT_ID', llm_client, system_prompt=system_prompt)
```

#### 3. Generate Single Response

```python
from imessage import create_ai_bot, generate_ai_response

# Create bot
llm_client, system_prompt = create_ai_bot('PHONE_NUMBER_OR_CHAT_ID')

# Generate response for latest message
response = generate_ai_response('PHONE_NUMBER_OR_CHAT_ID', llm_client, system_prompt)
print(response)
```

#### 4. Monitor All Chats

```python
from imessage import create_dynamic_ai_bot, monitor_all_chats_with_ai

# Create dynamic bot
llm_client, system_prompt = create_dynamic_ai_bot(
    model="llama2",
    system_prompt="Your custom prompt here"
)

# Monitor all chats
monitor_all_chats_with_ai(
    llm_client=llm_client,
    poll_interval=3,
    system_prompt=system_prompt,
    auto_respond=True,
    warmup_period=2
)
```

## Chat Identifiers

The bot works with different types of chat identifiers:

- **Phone Numbers**: `+1234567890` or `1234567890`
- **Group Chat IDs**: `chat123456789` (found in Messages database)
- **Contact Names**: Names as they appear in your contacts

## Advanced Features

### Interruption System

The bot includes an advanced interruption system that can:
- Detect when new messages should interrupt current response generation
- Prioritize urgent messages (direct mentions, questions)
- Handle conversation flow dynamically

### Conversation Memory

The bot maintains conversation context by:
- Storing recent message history
- Tracking conversation participants
- Understanding temporal context (recent vs. old messages)
- Recording bot responses for continuity

### Smart Response Logic

The bot uses intelligent analysis to decide when to respond:
- Direct addressing detection
- Question identification
- Conversation relevance analysis
- Context-aware decision making

## Customization

### Models

Supported LLM models (via Ollama):
- `llama2`: General purpose, good balance
- `llama3`: Improved performance
- `gemma3-abliterated`: Fewer restrictions, more natural responses
- `codellama`: Better for technical discussions
- Custom models: Any Ollama-compatible model

### API Endpoints

The bot supports different API types:
- **Ollama** (default): Local LLM hosting
- **OpenAI-compatible**: Any OpenAI-compatible API

```python
# OpenAI-compatible API
llm_client = LLMClient(
    base_url="https://api.openai.com",
    model="gpt-3.5-turbo",
    api_type="openai"
)
```

## Troubleshooting

### Common Issues

1. **Bot not responding**:
   - Check if Ollama is running: `ollama serve`
   - Verify model is available: `ollama list`
   - Check Messages app permissions

2. **Permission errors**:
   - Grant Full Disk Access to Terminal in System Preferences
   - Ensure Messages app is signed in to iMessage

3. **Database access issues**:
   - Restart Messages app
   - Check if chat database is accessible
   - Verify chat identifiers are correct

4. **Performance issues**:
   - Increase polling interval
   - Use smaller/faster models
   - Adjust context limit

### Debug Mode

Enable debug output by modifying the code:

```python
# Add debug prints
print(f"🔍 Debug: Last message ID: {last_message_id}")
print(f"🔍 Debug: Message text: {message_text}")
```

## Security Notes

- The bot accesses your Messages database directly
- All processing happens locally (no data sent to external servers)
- LLM models run locally via Ollama
- Messages are only stored temporarily for context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses Ollama for local LLM hosting
- Integrates with macOS Messages app
- Inspired by conversational AI research

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This bot is designed for personal use and educational purposes. Use responsibly and respect others' privacy in conversations. 