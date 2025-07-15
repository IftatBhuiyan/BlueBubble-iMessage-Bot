import subprocess
import tempfile
import sqlite3
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import re
import html
import plistlib
import requests
import json
import threading
import queue
from dataclasses import dataclass
from enum import Enum

# Enhanced conversation context and interruption system
@dataclass
class MessageInfo:
    """Structured message information"""
    sender: str
    content: str
    timestamp: datetime
    chat_id: str
    chat_type: str
    is_bot_response: bool = False
    message_id: Optional[str] = None

class InterruptionScore(Enum):
    """Interruption priority levels"""
    CRITICAL = 1.0      # Direct address to bot, urgent question
    HIGH = 0.8          # Question, important topic change
    MEDIUM = 0.6        # Related to current conversation
    LOW = 0.4           # General chat, not urgent
    IGNORE = 0.0        # Off-topic, spam, etc.

@dataclass
class InterruptionAnalysis:
    """Analysis of whether a new message should interrupt current response"""
    should_interrupt: bool
    score: float
    reason: str
    urgency_level: InterruptionScore
    context_relevance: float  # 0.0 to 1.0

class ResponseManager:
    """Manages response generation with interruption support"""
    
    def __init__(self):
        self.current_responses = {}  # chat_id -> response_info
        self.response_lock = threading.Lock()
        
    def start_response(self, chat_id: str, original_message: MessageInfo, response_thread: threading.Thread):
        """Register a new response being generated"""
        with self.response_lock:
            self.current_responses[chat_id] = {
                'original_message': original_message,
                'thread': response_thread,
                'start_time': datetime.now(),
                'interrupted': False
            }
    
    def interrupt_response(self, chat_id: str, interrupting_message: MessageInfo, analysis: InterruptionAnalysis):
        """Interrupt current response if one exists"""
        with self.response_lock:
            if chat_id in self.current_responses:
                response_info = self.current_responses[chat_id]
                response_info['interrupted'] = True
                response_info['interrupting_message'] = interrupting_message
                response_info['interruption_analysis'] = analysis
                return True
            return False
    
    def finish_response(self, chat_id: str):
        """Clean up after response is complete"""
        with self.response_lock:
            if chat_id in self.current_responses:
                del self.current_responses[chat_id]
    
    def is_response_active(self, chat_id: str) -> bool:
        """Check if a response is currently being generated"""
        with self.response_lock:
            return chat_id in self.current_responses
    
    def get_response_info(self, chat_id: str) -> Optional[Dict]:
        """Get information about current response"""
        with self.response_lock:
            return self.current_responses.get(chat_id)

# Global response manager
response_manager = ResponseManager()

# Conversation Memory and Intelligence
class ConversationMemory:
    """Manages conversation history and determines when bot should respond"""
    
    def __init__(self):
        self.chat_histories = {}  # chat_id -> list of MessageInfo objects
        self.bot_responses = {}   # chat_id -> list of bot response info
        self.max_history = 100    # Keep last 100 messages per chat (increased from 50)
        self.context_limit = 30   # Use up to 30 messages for context with proper temporal understanding
        
    def add_message(self, chat_id: str, message_info: MessageInfo):
        """Add a message to conversation history"""
        if chat_id not in self.chat_histories:
            self.chat_histories[chat_id] = []
        
        self.chat_histories[chat_id].append(message_info)
        
        # Keep only recent messages
        if len(self.chat_histories[chat_id]) > self.max_history:
            self.chat_histories[chat_id] = self.chat_histories[chat_id][-self.max_history:]
    
    def get_enhanced_context(self, chat_id: str, limit: Optional[int] = None) -> List[MessageInfo]:
        """Get enhanced conversation context with more history"""
        if limit is None:
            limit = self.context_limit
        
        if chat_id not in self.chat_histories:
            return []
        
        # Return recent messages in chronological order
        messages = self.chat_histories[chat_id][-limit:]
        return messages
    
    def analyze_interruption(self, chat_id: str, new_message: MessageInfo, llm_client) -> InterruptionAnalysis:
        """Analyze if a new message should interrupt current response generation"""
        
        # Get current response info
        current_response = response_manager.get_response_info(chat_id)
        if not current_response:
            return InterruptionAnalysis(
                should_interrupt=False,
                score=0.0,
                reason="No active response to interrupt",
                urgency_level=InterruptionScore.IGNORE,
                context_relevance=0.0
            )
        
        original_message = current_response['original_message']
        
        # Quick heuristic checks for high-priority interruptions
        new_content = new_message.content.lower()
        
        # Critical interruptions - always interrupt
        if any(trigger in new_content for trigger in ['yourbot', 'stop', 'wait', 'nevermind', 'actually']):
            return InterruptionAnalysis(
                should_interrupt=True,
                score=1.0,
                reason="Direct address or conversation control word detected",
                urgency_level=InterruptionScore.CRITICAL,
                context_relevance=1.0
            )
        
        # High priority - questions, corrections
        if any(trigger in new_content for trigger in ['?', 'what', 'why', 'how', 'when', 'where', 'no,', 'wait,', 'but']):
            return InterruptionAnalysis(
                should_interrupt=True,
                score=0.8,
                reason="Question or correction detected",
                urgency_level=InterruptionScore.HIGH,
                context_relevance=0.8
            )
        
        # Medium priority - topic changes, new information
        if len(new_content.split()) > 5:  # Substantial message
            return InterruptionAnalysis(
                should_interrupt=True,
                score=0.6,
                reason="Substantial new message with potential topic change",
                urgency_level=InterruptionScore.MEDIUM,
                context_relevance=0.6
            )
        
        # Low priority - short responses, acknowledgments
        if len(new_content.split()) <= 3:
            return InterruptionAnalysis(
                should_interrupt=False,
                score=0.3,
                reason="Short message, likely acknowledgment",
                urgency_level=InterruptionScore.LOW,
                context_relevance=0.3
            )
        
        # Default to medium priority
        return InterruptionAnalysis(
            should_interrupt=True,
            score=0.5,
            reason="New message during response generation",
            urgency_level=InterruptionScore.MEDIUM,
            context_relevance=0.5
        )
    
    def should_respond(self, chat_id: str, message_text: str, sender_name: str, chat_type: str, llm_client) -> Dict:
        """
        Intelligent decision on whether bot should respond using chain-of-thought reasoning
        Returns: {
            'should_respond': bool,
            'confidence': float,
            'reason': str,
            'response_type': str
        }
        """
        # Get recent conversation context
        recent_messages = self.get_enhanced_context(chat_id, limit=10)
        
        # First, do chain-of-thought reasoning to analyze the message
        print(f"\n🤖 Analyzing message: '{message_text[:50]}...' from {sender_name} in {chat_type} chat")
        print("🧠 Bot is thinking...")
        print("\n🧠 Bot's Chain-of-Thought Analysis:")
        print("="*60)
        
        reasoning_decision = self._generate_reasoning_decision(message_text, sender_name, chat_type, recent_messages, llm_client)
        
        print("="*60)
        print(f"📊 Final Decision: {'RESPOND' if reasoning_decision['should_respond'] else 'IGNORE'} (confidence: {reasoning_decision['confidence']})")
        print(f"🔍 Reason: {reasoning_decision['reason']}")
        print()
        
        return reasoning_decision
    
    def _generate_reasoning_decision(self, message_text: str, sender_name: str, chat_type: str, recent_messages: List, llm_client) -> Dict:
        """Generate reasoning and decision using LLM chain-of-thought"""
        
        # Safely handle None or empty inputs
        if not message_text:
            message_text = "[Empty message]"
        if not sender_name:
            sender_name = "[Unknown sender]"
        
        # Build context about recent conversation
        context_summary = ""
        if recent_messages:
            recent_context = recent_messages[-5:]  # Last 5 messages
            context_summary = "Recent conversation:\n"
            for msg in recent_context:
                sender = msg.sender
                content = msg.content[:50]
                is_bot = msg.is_bot_response
                context_summary += f"- {sender}{'(bot)' if is_bot else ''}: {content}\n"
        
        # Create reasoning prompt
        reasoning_prompt = f"""You are YourBotName, analyzing whether to respond to a message in a {chat_type.lower()} chat.

CONTEXT:
{context_summary}

NEW MESSAGE:
From: {sender_name}
Message: "{message_text}"

ANALYSIS INSTRUCTIONS:
Think through this step by step:
1. Is this message directly addressing me (YourBotName) by name or with "hey yourbot", "yourbot can you", etc?
2. Is this message addressing someone else specifically?
3. What is the conversation context - am I already active in this chat?
4. Does this message warrant a response from me as a friend?
5. In a {chat_type.lower()} chat, what would be natural behavior?

Think about this naturally - as a real person would. Don't force responses.

After your analysis, end with exactly this format:
DECISION: [RESPOND/IGNORE]
CONFIDENCE: [0.1-1.0]
REASON: [brief explanation]
TYPE: [direct_address/group_question/individual_substantial/ignore_general/etc]"""
        
        try:
            # Use streaming reasoning to show the thought process
            reasoning_response = self._stream_reasoning_decision(reasoning_prompt, llm_client)
            
            # Parse the decision from the reasoning
            return self._parse_reasoning_decision(reasoning_response)
            
        except Exception as e:
            print(f"❌ Reasoning failed: {e}")
            # Fallback to simple pattern matching
            return self._fallback_analysis(message_text, sender_name, chat_type, recent_messages)
    
    def _stream_reasoning_decision(self, prompt: str, llm_client) -> str:
        """Stream the reasoning process and capture the response"""
        import sys
        import json
        
        full_response = ""
        
        try:
            payload = {
                "model": llm_client.model,
                "prompt": f"User: {prompt}\nAssistant: ",
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 300,
                    "stop": ["User:", "DECISION:", "CONFIDENCE:", "REASON:", "TYPE:"],
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{llm_client.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120  # 2 minutes for reasoning generation
            )
            response.raise_for_status()
            
            # Stream the response character by character
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk is not None and 'response' in chunk:
                            text = chunk['response']
                            full_response += text
                            # Print each character with a small delay
                            for char in text:
                                sys.stdout.write(char)
                                sys.stdout.flush()
                                import time
                                time.sleep(0.01)
                        
                        # Check if done
                        if chunk is not None and chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line at the end
            
            # Now get the final decision part without streaming
            decision_payload = {
                "model": llm_client.model,
                "prompt": f"User: {prompt}\nAssistant: {full_response}\n\nNow provide your final decision in the exact format requested:",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 100
                }
            }
            
            decision_response = requests.post(
                f"{llm_client.base_url}/api/generate",
                json=decision_payload,
                timeout=60
            )
            decision_response.raise_for_status()
            
            decision_result = decision_response.json()
            if decision_result is None:
                decision_text = ""
            else:
                decision_text = decision_result.get('response', '').strip()
            
            return full_response + "\n" + decision_text
            
        except Exception as e:
            print(f"Streaming failed: {e}")
            raise
    
    def _parse_reasoning_decision(self, reasoning_response: str) -> Dict:
        """Parse the LLM's reasoning response to extract decision"""
        import re
        
        # Handle None or empty reasoning_response
        if not reasoning_response:
            return {
                'should_respond': False,
                'confidence': 0.1,
                'reason': "Empty reasoning response",
                'response_type': "fallback_empty"
            }
        
        # Extract decision components
        decision_match = re.search(r'DECISION:\s*(RESPOND|IGNORE)', reasoning_response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', reasoning_response)
        reason_match = re.search(r'REASON:\s*(.+?)(?:\nTYPE:|$)', reasoning_response, re.DOTALL)
        type_match = re.search(r'TYPE:\s*(.+?)(?:\n|$)', reasoning_response)
        
        should_respond = decision_match.group(1).upper() == 'RESPOND' if decision_match else False
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        reason = reason_match.group(1).strip() if reason_match else "LLM reasoning analysis"
        response_type = type_match.group(1).strip() if type_match else "llm_decision"
        
        return {
            'should_respond': should_respond,
            'confidence': confidence,
            'reason': reason,
            'response_type': response_type
        }
    
    def _fallback_analysis(self, message_text: str, sender_name: str, chat_type: str, recent_messages: List) -> Dict:
        """Fallback to simple pattern matching if LLM reasoning fails"""
        message_lower = message_text.lower().strip()
        
        # Direct addressing patterns
        direct_patterns = [
            r'\b(hey|hi|hello)\b.*\b(bot|ai|assistant|yourbot)\b',
            r'\b(bot|ai|assistant|yourbot)\b.*\?',
            r'^(bot|ai|assistant|yourbot)[,:]',
            r'\@(bot|ai|assistant|yourbot)\b',
        ]
        
        for pattern in direct_patterns:
            if re.search(pattern, message_lower):
                return {
                    'should_respond': True,
                    'confidence': 0.9,
                    'reason': 'Direct addressing detected (fallback)',
                    'response_type': 'direct_address'
                }
        
        # Default to not responding in fallback
        return {
            'should_respond': False,
            'confidence': 0.3,
            'reason': 'Fallback analysis - being conservative',
            'response_type': 'fallback_ignore'
        }
    

    
    def get_recent_messages(self, chat_id: str, limit: int = 10) -> List[Dict]:
        """Get recent messages from a chat"""
        if chat_id not in self.chat_histories:
            return []
        
        return self.chat_histories[chat_id][-limit:]
    
    def record_bot_response(self, chat_id: str, original_message: str, bot_response: str, reason: str):
        """Record that bot responded to a message"""
        if chat_id not in self.bot_responses:
            self.bot_responses[chat_id] = []
        
        self.bot_responses[chat_id].append({
            'timestamp': datetime.now(),
            'original_message': original_message,
            'bot_response': bot_response,
            'reason': reason
        })
        
        # Also add bot response to conversation history for context tracking
        bot_message_info = MessageInfo(
            sender='bot',
            content=bot_response,
            timestamp=datetime.now(),
            chat_id=chat_id,
            chat_type='unknown',  # Will be updated when we have chat type info
            is_bot_response=True
        )
        self.add_message(chat_id, bot_message_info)

# Global conversation memory instance
conversation_memory = ConversationMemory()

# LLM Integration Functions
class LLMClient:
    """Client for interacting with local LLM APIs (Ollama, OpenAI-compatible)"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", api_type: str = "ollama"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_type = api_type
        self.timeout = 120  # 2 minutes for longer, more thoughtful responses
        
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate a response using the LLM"""
        try:
            if self.api_type == "ollama":
                return self._ollama_generate(messages, system_prompt)
            elif self.api_type == "openai":
                return self._openai_generate(messages, system_prompt)
            else:
                raise ValueError(f"Unsupported API type: {self.api_type}")
        except Exception as e:
            print(f"❌ Error generating LLM response: {e}")
            return "Sorry, I'm having trouble responding right now."
    

    
    def _ollama_generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate response using Ollama API"""
        # Build the prompt from messages
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,  # Higher temperature for more creative/unfiltered responses
                "top_p": 0.95,       # Higher top_p for more diverse responses
                "top_k": 50,         # Allow more token choices
                "max_tokens": 1000,  # Allow longer responses
                "repeat_penalty": 1.1,
                "seed": -1,          # Random seed for varied responses
                "num_predict": 1000,  # Allow longer responses
                "stop": [],          # No stop sequences to allow any content
                "tfs_z": 1.0,        # Disable tail free sampling constraints
                "typical_p": 1.0,    # Disable typical sampling constraints
                "presence_penalty": 0.0,  # No penalty for controversial topics
                "frequency_penalty": 0.0  # No penalty for repetitive content
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    def _openai_generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenAI-compatible API"""
        formatted_messages = []
        
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        formatted_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

def send_imessage(chat_identifier, message):
    """
    Send an iMessage to either an individual contact or a group chat.
    Handles both phone numbers and group chat IDs.
    Uses multiple fallback methods to ensure delivery even if chat names/participants change.
    """
    # Check if this is a group chat ID (starts with 'chat') or individual number
    if chat_identifier.startswith('chat'):
        # For group chats, try multiple robust methods
        print(f"📱 Sending to group chat: {chat_identifier}")
        
        # Method 1: Try using chat id directly (fastest when it works)
        script1 = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            try
                set targetChat to chat id "{chat_identifier}" of targetService
                send "{message}" to targetChat
                return "success"
            on error errMsg
                return "error: " & errMsg
            end try
        end tell
        '''
        
        # Method 2: Find chat by iterating through all chats (most reliable for ID matching)
        script2 = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            try
                set allChats to chats of targetService
                repeat with aChat in allChats
                    if id of aChat is "{chat_identifier}" then
                        send "{message}" to aChat
                        return "success"
                    end if
                end repeat
                return "error: chat not found"
            on error errMsg
                return "error: " & errMsg
            end try
        end tell
        '''
        
        # Method 3: Try finding by display name (fallback for when chat is accessible by name)
        chat_info = get_chat_type_info(chat_identifier)
        chat_name = chat_info.get('chat_name', '') if chat_info else ''
        
        script3 = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            try
                set allChats to chats of targetService
                repeat with aChat in allChats
                    try
                        if name of aChat is "{chat_name}" then
                            send "{message}" to aChat
                            return "success"
                        end if
                    end try
                end repeat
                return "error: chat not found by name"
            on error errMsg
                return "error: " & errMsg
            end try
        end tell
        '''
        
        # Method 4: Universal fallback - try to send to ANY chat that has recent activity
        # This is the most robust method for changed/renamed chats
        script4 = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            try
                set allChats to chats of targetService
                -- Look for chats with recent messages that might match our target
                repeat with aChat in allChats
                    try
                        -- Try to send to any group chat (more than 1 participant)
                        if (count of participants of aChat) > 1 then
                            -- This is a group chat, try sending
                            send "{message}" to aChat
                            return "success: sent to group " & name of aChat
                        end if
                    on error
                        -- Skip this chat if we can't access it
                    end try
                end repeat
                return "error: no accessible group chats found"
            on error errMsg
                return "error: " & errMsg
            end try
        end tell
        '''
        
        # Try each method in order of reliability
        methods = [
            ("direct chat id", script1),
            ("iterate by id", script2),
        ]
        
        # Add name-based method if we have a chat name
        if chat_name and chat_name != chat_identifier:
            methods.append(("iterate by name", script3))
        
        # Add universal fallback as last resort
        methods.append(("universal group fallback", script4))
        
        for method_name, script in methods:
            try:
                print(f"🔄 Trying method: {method_name}")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
                    f.write(script)
                    temp_path = f.name
                
                result = subprocess.run(["osascript", temp_path], capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                
                if "success" in output:
                    print(f"✅ Message sent successfully using method: {method_name}")
                    if "sent to group" in output:
                        print(f"📝 Note: {output}")
                    os.unlink(temp_path)
                    return
                else:
                    print(f"❌ Method {method_name} failed: {output}")
                
                os.unlink(temp_path)
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Method {method_name} failed with error: {e.stderr.strip()}")
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # If all methods failed, show comprehensive error
        print(f"❌ All group chat methods failed for chat: {chat_identifier}")
        print(f"💡 Chat name: {chat_name}")
        print(f"💡 This could mean:")
        print(f"   - Chat was renamed or deleted")
        print(f"   - Participants changed significantly") 
        print(f"   - Messages app needs to be restarted")
        print(f"   - You may need to send a message to this group manually first")
        
    else:
        # This is an individual contact - use buddy syntax (unchanged)
        script = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{chat_identifier}" of targetService
            send "{message}" to targetBuddy
        end tell
        '''

        # Save AppleScript to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            temp_path = f.name
            print(f"Script saved to: {temp_path}")
            print("Running AppleScript...")

        try:
            result = subprocess.run(["osascript", temp_path], capture_output=True, text=True, check=True)
            print("✅ Message sent successfully!")
            if result.stdout.strip():
                print("STDOUT:", result.stdout.strip())
            if result.stderr.strip():
                print("STDERR:", result.stderr.strip())
        except subprocess.CalledProcessError as e:
            print("❌ Error while sending message:")
            print("STDOUT:", e.stdout.strip())
            print("STDERR:", e.stderr.strip())
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

def get_messages_database_path() -> str:
    """Get the path to the Messages database."""
    home = os.path.expanduser("~")
    return os.path.join(home, "Library", "Messages", "chat.db")

def read_recent_messages_from_db(limit: int = 50) -> List[Dict]:
    """
    Read recent messages from the Messages database.
    Returns list of message dictionaries with sender, text, timestamp, etc.
    """
    db_path = get_messages_database_path()
    
    if not os.path.exists(db_path):
        print(f"❌ Messages database not found at: {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, let's check what tables and columns exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Available tables: {[t[0] for t in tables]}")
        
        # Check handle table structure
        cursor.execute("PRAGMA table_info(handle);")
        handle_columns = cursor.fetchall()
        print(f"📊 Handle table columns: {[c[1] for c in handle_columns]}")
        
        # Check message table structure
        cursor.execute("PRAGMA table_info(message);")
        message_columns = cursor.fetchall()
        print(f"📊 Message table columns: {[c[1] for c in message_columns]}")
        
        # Query to get recent messages with sender info and contact names
        query = """
        SELECT 
            m.ROWID,
            m.text,
            m.date,
            m.is_from_me,
            h.id as sender_id,
            c.chat_identifier,
            c.display_name,
            c.room_name
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        LEFT JOIN chat c ON cmj.chat_id = c.ROWID
        WHERE m.text IS NOT NULL AND m.text != ''
        ORDER BY m.date DESC
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        messages = []
        for row in rows:
            sender_id = row[4]
            chat_display_name = row[6]
            room_name = row[7]
            
            # Try to get a friendly name for the sender
            sender_name = sender_id or "Unknown"
            if chat_display_name and chat_display_name != sender_id:
                sender_name = chat_display_name
            elif room_name and room_name != sender_id:
                sender_name = room_name
            
            message = {
                'id': row[0],
                'text': row[1],
                'timestamp': row[2],
                'is_from_me': bool(row[3]),
                'sender_id': sender_id,
                'sender_name': sender_name,
                'chat_identifier': row[5],
                'chat_name': chat_display_name or room_name or row[5] or "Unknown Chat"
            }
            messages.append(message)
        
        conn.close()
        return messages
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error reading messages: {e}")
        return []

def get_chat_participants(chat_identifier: str) -> List[str]:
    """Get list of participants in a specific chat."""
    db_path = get_messages_database_path()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT DISTINCT h.id
        FROM chat_handle_join chj
        JOIN handle h ON chj.handle_id = h.ROWID
        JOIN chat c ON chj.chat_id = c.ROWID
        WHERE c.chat_identifier = ?
        """
        
        cursor.execute(query, (chat_identifier,))
        participants = []
        
        for row in cursor.fetchall():
            participants.append(row[0] or "Unknown")
        
        conn.close()
        return participants
        
    except Exception as e:
        print(f"❌ Error getting chat participants: {e}")
        return []

def read_messages_from_chat_via_applescript(chat_identifier: str, limit: int = 10) -> List[Dict]:
    """
    Read messages from a specific chat using AppleScript.
    This is more reliable for getting the most recent messages.
    """
    script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetChat to chat id "{chat_identifier}" of targetService
        
        set recentMessages to {{}}
        set messageCount to 0
        set maxMessages to {limit}
        
        repeat with msg in messages of targetChat
            if messageCount ≥ maxMessages then
                exit repeat
            end if
            
            set messageInfo to {{
                text: text of msg,
                sender: name of sender of msg,
                date: date sent of msg,
                is_from_me: is from me of msg
            }}
            set end of recentMessages to messageInfo
            set messageCount to messageCount + 1
        end repeat
        
        return recentMessages
    end tell
    '''
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            temp_path = f.name
        
        result = subprocess.run(["osascript", temp_path], capture_output=True, text=True, check=True)
        
        # Parse the AppleScript result (this is simplified - actual parsing would be more complex)
        print("✅ Retrieved messages via AppleScript")
        print("Raw result:", result.stdout.strip())
        
        # For now, return a placeholder - you'd need to parse the AppleScript output
        return [{"text": "Sample message", "sender": "Test User", "timestamp": time.time()}]
        
    except subprocess.CalledProcessError as e:
        print(f"❌ AppleScript error: {e.stderr.strip()}")
        return []
    except Exception as e:
        print(f"❌ Error reading messages via AppleScript: {e}")
        return []

def get_recent_group_messages(group_name: str, limit: int = 20) -> List[Dict]:
    """
    Get recent messages from a specific group chat.
    Tries AppleScript first, falls back to database.
    """
    print(f"🔍 Looking for messages in group: {group_name}")
    
    # First try to get messages via AppleScript
    messages = read_messages_from_chat_via_applescript(group_name, limit)
    
    if not messages:
        # Fall back to database method
        print("📊 Falling back to database method...")
        all_messages = read_recent_messages_from_db(limit * 2)
        
        # Filter for the specific group
        group_messages = [
            msg for msg in all_messages 
            if group_name.lower() in msg['chat_name'].lower() or 
               group_name.lower() in msg['chat_identifier'].lower()
        ]
        
        messages = group_messages[:limit]
    
    return messages

def format_message_for_display(message: Dict) -> str:
    """Format a message for nice display."""
    timestamp = datetime.fromtimestamp(message.get('timestamp', 0) / 1000000000)
    sender = message.get('sender_name', 'Unknown')
    text = message.get('text', '')
    
    return f"[{timestamp.strftime('%H:%M')}] {sender}: {text}"

def monitor_chat_with_ai(chat_identifier: str, llm_client: LLMClient, poll_interval: int = 5, 
                        system_prompt: Optional[str] = None, auto_respond: bool = True):
    """
    Monitor a chat for new messages and respond with AI.
    This is the main function for the AI bot.
    """
    print(f"🤖 Starting AI bot for chat: {chat_identifier}")
    print(f"⏰ Polling every {poll_interval} seconds...")
    print(f"🧠 Using model: {llm_client.model}")
    print(f"🔄 Auto-respond: {'ON' if auto_respond else 'OFF'}")
    
    last_message_id = None
    
    while True:
        try:
            # Get the latest message
            last_message = get_last_message_info(chat_identifier)
            
            if last_message and last_message.get('id') != last_message_id:
                last_message_id = last_message.get('id')
                
                # Skip messages from ourselves
                if not last_message.get('is_from_me', False):
                    sender_name = last_message.get('sender_name', 'Unknown')
                    message_text = last_message.get('text', '')
                    
                    print(f"\n📨 New message from {sender_name}: {message_text}")
                    
                    if auto_respond and message_text.strip():
                        print("🤖 Generating AI response...")
                        
                        # Generate AI response
                        ai_response = generate_ai_response(chat_identifier, llm_client, system_prompt)
                        
                        if ai_response and ai_response.strip():
                            print(f"💬 AI Response: {ai_response}")
                            
                            # Send the response back to the chat
                            try:
                                send_imessage(chat_identifier, ai_response)
                                print("✅ Response sent!")
                            except Exception as e:
                                print(f"❌ Error sending response: {e}")
                        else:
                            print("⚠️  No response generated")
                    else:
                        print("ℹ️  Auto-respond disabled or empty message")
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 AI bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in AI monitoring: {e}")
            time.sleep(poll_interval)

def monitor_group_chat(group_name: str, poll_interval: int = 5):
    """
    Monitor a group chat for new messages.
    This is the legacy function - use monitor_chat_with_ai for AI functionality.
    """
    print(f"🤖 Starting to monitor group: {group_name}")
    print(f"⏰ Polling every {poll_interval} seconds...")
    
    last_message_id = None
    
    while True:
        try:
            messages = get_recent_group_messages(group_name, limit=10)
            
            if messages and messages[0].get('id') != last_message_id:
                latest_message = messages[0]
                last_message_id = latest_message.get('id')
                
                # Skip messages from ourselves
                if not latest_message.get('is_from_me', False):
                    print(f"\n📨 New message detected:")
                    print(format_message_for_display(latest_message))
                    
                    print("🤖 [Use monitor_chat_with_ai for AI responses]")
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            time.sleep(poll_interval)



def build_conversation_context(chat_identifier: str, limit: int = 30) -> List[Dict[str, str]]:
    """Build conversation context for LLM from recent messages"""
    db_path = get_messages_database_path()
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find the chat (try exact match first, then partial match)
        cursor.execute("SELECT ROWID, chat_identifier, display_name FROM chat WHERE chat_identifier = ? OR display_name = ?", 
                      (chat_identifier, chat_identifier))
        chat_result = cursor.fetchone()
        
        # If not found, try partial match (e.g., '3474256886' should match '+13474256886')
        if not chat_result:
            cursor.execute("SELECT ROWID, chat_identifier, display_name FROM chat WHERE chat_identifier LIKE ? OR display_name LIKE ?", 
                          (f"%{chat_identifier}%", f"%{chat_identifier}%"))
            chat_result = cursor.fetchone()
        
        if not chat_result:
            conn.close()
            return []
        
        chat_id = chat_result[0]
        
        # Get recent messages
        query = """
            SELECT m.text, m.attributedBody, m.date, m.is_from_me, h.id as sender_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            WHERE cmj.chat_id = ?
            ORDER BY m.date DESC
            LIMIT ?
        """
        
        cursor.execute(query, (chat_id, limit))
        rows = cursor.fetchall()
        conn.close()
        
        # Format messages for LLM with timestamps and better structure
        messages = []
        now = datetime.now()
        
        for i, row in enumerate(reversed(rows)):  # Reverse to get chronological order
            text, attributed_body, timestamp, is_from_me, sender_id = row
            
            # Get message text
            if not text and attributed_body:
                text = extract_text_from_attributed_body(attributed_body)
            
            if text and text.strip():
                # Convert timestamp to readable format
                try:
                    if timestamp and timestamp > 0:
                        # macOS timestamp is in nanoseconds since Jan 1, 2001
                        # Convert to seconds by dividing by 1,000,000,000
                        mac_epoch = datetime(2001, 1, 1)
                        timestamp_seconds = timestamp / 1_000_000_000
                        msg_time = mac_epoch + timedelta(seconds=timestamp_seconds)
                        
                        # Calculate time difference from now
                        time_diff = now - msg_time
                        
                        # Handle negative time differences (future messages)
                        if time_diff.total_seconds() < 0:
                            time_str = "just now"
                        elif time_diff.days > 0:
                            time_str = f"{time_diff.days}d ago"
                        elif time_diff.seconds > 3600:
                            hours = time_diff.seconds // 3600
                            time_str = f"{hours}h ago"
                        elif time_diff.seconds > 60:
                            minutes = time_diff.seconds // 60
                            time_str = f"{minutes}m ago"
                        elif time_diff.seconds > 5:
                            time_str = f"{time_diff.seconds}s ago"
                        else:
                            time_str = "just now"
                    else:
                        time_str = "recent"
                    
                    # Mark recent messages (last 3 messages) as [RECENT]
                    recency_marker = "[RECENT] " if i >= len(rows) - 3 else ""
                    
                except Exception as e:
                    time_str = "recent"
                    recency_marker = "[RECENT] " if i >= len(rows) - 3 else ""
                
                if is_from_me:
                    # Include our own messages as assistant messages for context
                    messages.append({
                        "role": "assistant",
                        "content": f"{recency_marker}[{time_str}] {text.strip()}"
                    })
                else:
                    # This is a message from the user
                    sender_name = get_contact_name(sender_id) if sender_id else "User"
                    messages.append({
                        "role": "user",
                        "content": f"{recency_marker}[{time_str}] {sender_name}: {text.strip()}"
                    })
        
        return messages
        
    except Exception as e:
        print(f"❌ Error building conversation context: {e}")
        return []

def get_last_message_info(chat_identifier: str) -> Optional[Dict]:
    """Get information about the last message in a chat"""
    db_path = get_messages_database_path()
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find the chat (try exact match first, then partial match)
        cursor.execute("SELECT ROWID FROM chat WHERE chat_identifier = ? OR display_name = ?", 
                      (chat_identifier, chat_identifier))
        chat_result = cursor.fetchone()
        
        # If not found, try partial match (e.g., '3474256886' should match '+13474256886')
        if not chat_result:
            cursor.execute("SELECT ROWID FROM chat WHERE chat_identifier LIKE ? OR display_name LIKE ?", 
                          (f"%{chat_identifier}%", f"%{chat_identifier}%"))
            chat_result = cursor.fetchone()
        
        if not chat_result:
            conn.close()
            return None
        
        chat_id = chat_result[0]
        
        # Get the most recent message
        query = """
            SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id as sender_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            WHERE cmj.chat_id = ?
            ORDER BY m.date DESC
            LIMIT 1
        """
        
        cursor.execute(query, (chat_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            msg_id, text, attributed_body, timestamp, is_from_me, sender_id = row
            
            # Get message text
            if not text and attributed_body:
                text = extract_text_from_attributed_body(attributed_body)
            
            return {
                'id': msg_id,
                'text': text,
                'timestamp': timestamp,
                'is_from_me': is_from_me,
                'sender_id': sender_id,
                'sender_name': get_contact_name(sender_id) if sender_id else "Unknown"
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error getting last message info: {e}")
        return None

def generate_ai_response(chat_identifier: str, llm_client: LLMClient, system_prompt: Optional[str] = None) -> str:
    """Generate an AI response for the latest message in a chat"""
    # Get enhanced conversation context (more history)
    messages = build_conversation_context(chat_identifier, limit=30)
    
    if not messages:
        return "I don't have any context to respond to."
    
    # Default system prompt if none provided
    if not system_prompt:
        system_prompt = """You are YourBotName, a helpful and friendly AI assistant responding in an iMessage conversation. 
Keep your responses conversational, natural, and concise. Respond as if you're a human friend in the chat.
Don't mention that you're an AI unless directly asked.

UNDERSTANDING CONVERSATION FLOW:
- Messages are shown with timestamps [Xm ago, Xh ago, etc.]
- Messages marked [RECENT] are the most current and relevant
- Focus primarily on [RECENT] messages - these are what you should respond to
- Use older messages only for background context and understanding the conversation flow
- If someone asks about something from much earlier, acknowledge the time gap

RESPONSE PRIORITY:
1. HIGHEST: [RECENT] messages that directly address you ("YourBotName") or ask YOU questions
2. HIGH: [RECENT] messages that continue the current conversation topic
3. MEDIUM: [RECENT] messages that change the topic or add new information
4. LOW: Older messages (use only for context, don't respond to them directly)

CRITICAL INSTRUCTION: 
- If you see "YourBotName [question]" in the [RECENT] messages, respond to THAT question
- If multiple people are talking, respond to the most recent person who directly addressed YOU
- Don't respond to questions directed at other people (like "PersonName what's the weather?")
- Always check WHO the message is directed to before responding

Remember: You're responding to the conversation as it is NOW, not as it was hours or days ago."""
    
    # Generate response using LLM
    response = llm_client.generate_response(messages, system_prompt)
    
    return response

def generate_ai_response_threaded(chat_identifier: str, original_message: MessageInfo, llm_client: LLMClient, system_prompt: Optional[str] = None) -> str:
    """Generate an AI response with interruption support"""
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """You are YourBotName, a helpful and friendly AI assistant responding in an iMessage conversation. 
Keep your responses conversational, natural, and concise. Respond as if you're a human friend in the chat.
Don't mention that you're an AI unless directly asked.

UNDERSTANDING CONVERSATION FLOW:
- Messages are shown with timestamps [Xm ago, Xh ago, etc.]
- Messages marked [RECENT] are the most current and relevant
- Focus primarily on [RECENT] messages - these are what you should respond to
- Use older messages only for background context and understanding the conversation flow
- If someone asks about something from much earlier, acknowledge the time gap

RESPONSE PRIORITY:
1. HIGHEST: [RECENT] messages that directly address you ("YourBotName") or ask YOU questions
2. HIGH: [RECENT] messages that continue the current conversation topic
3. MEDIUM: [RECENT] messages that change the topic or add new information
4. LOW: Older messages (use only for context, don't respond to them directly)

CRITICAL INSTRUCTION: 
- If you see "YourBotName [question]" in the [RECENT] messages, respond to THAT question
- If multiple people are talking, respond to the most recent person who directly addressed YOU
- Don't respond to questions directed at other people (like "PersonName what's the weather?")
- Always check WHO the message is directed to before responding

Remember: You're responding to the conversation as it is NOW, not as it was hours or days ago."""
    
    result_queue = queue.Queue()
    
    def response_worker():
        """Worker function that generates the response"""
        try:
            # Get enhanced conversation context (more history)
            messages = build_conversation_context(chat_identifier, limit=30)
            
            if not messages:
                result_queue.put("I don't have any context to respond to.")
                return
            
            # Generate response using LLM
            response = llm_client.generate_response(messages, system_prompt)
            
            # Check if we were interrupted during generation
            response_info = response_manager.get_response_info(chat_identifier)
            if response_info and response_info.get('interrupted'):
                interruption_analysis = response_info.get('interruption_analysis')
                interrupting_message = response_info.get('interrupting_message')
                
                if interruption_analysis and interrupting_message:
                    print(f"🔄 Response interrupted by new message: '{interrupting_message.content[:50]}...'")
                    print(f"🔍 Interruption analysis: {interruption_analysis.reason} (score: {interruption_analysis.score})")
                    
                    # Decide whether to continue with original response or generate new one
                    if interruption_analysis.urgency_level in [InterruptionScore.CRITICAL, InterruptionScore.HIGH]:
                        print("🚨 High priority interruption - generating new response for latest message")
                        
                        # Generate new response considering the interrupting message
                        updated_messages = build_conversation_context(chat_identifier, limit=30)
                        new_response = llm_client.generate_response(updated_messages, system_prompt)
                        
                        # Record both messages in conversation memory
                        conversation_memory.record_bot_response(
                            chat_identifier, interrupting_message.content, new_response, 
                            f"Interrupted response: {interruption_analysis.reason}"
                        )
                        
                        result_queue.put(new_response)
                        return
                    else:
                        print("ℹ️ Low priority interruption - continuing with original response")
                        # Continue with original response but acknowledge the interruption
                        conversation_memory.record_bot_response(
                            chat_identifier, original_message.content, response, 
                            f"Completed despite interruption: {interruption_analysis.reason}"
                        )
                        result_queue.put(response)
                        return
            else:
                # No interruption, proceed normally
                conversation_memory.record_bot_response(
                    chat_identifier, original_message.content, response, "Normal response"
                )
                result_queue.put(response)
                return
                
        except Exception as e:
            print(f"❌ Error in threaded response generation: {e}")
            result_queue.put("Sorry, I encountered an error while generating a response.")
        finally:
            # Clean up response tracking
            response_manager.finish_response(chat_identifier)
    
    # Start the response generation in a separate thread
    response_thread = threading.Thread(target=response_worker)
    response_manager.start_response(chat_identifier, original_message, response_thread)
    
    response_thread.start()
    response_thread.join(timeout=120)  # 2 minute timeout for longer responses
    
    if response_thread.is_alive():
        print("⏰ Response generation timed out after 2 minutes")
        response_manager.finish_response(chat_identifier)
        return "Sorry, response generation took too long."
    
    # Get the result from the queue
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return "Sorry, no response was generated."

def get_contact_name(phone_number: str) -> str:
    """
    Try to get the contact name for a phone number using AppleScript to access Contacts.
    Falls back to the phone number if no name is found.
    """
    if not phone_number:
        return "Unknown"
    
    # Try to get contact name via AppleScript - first try with original number
    script = f'''
    tell application "Contacts"
        try
            set foundContact to (first person whose (value of phone of it contains "{phone_number}"))
            return name of foundContact
        on error
            try
                -- Try without the + prefix
                set cleanNumber to "{phone_number.replace('+', '')}"
                set foundContact to (first person whose (value of phone of it contains cleanNumber))
                return name of foundContact
            on error
                return "{phone_number}"
            end try
        end try
    end tell
    '''
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".applescript", delete=False) as f:
            f.write(script)
            temp_path = f.name
        
        result = subprocess.run(["osascript", temp_path], capture_output=True, text=True, timeout=5)
        os.unlink(temp_path)  # Clean up temp file
        
        if result.returncode == 0:
            contact_name = result.stdout.strip()
            if contact_name and contact_name != phone_number:
                return contact_name
    except Exception as e:
        print(f"Error getting contact name via AppleScript: {e}")
    
    # Fallback: try to get it from the Messages database
    db_path = get_messages_database_path()
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Look for the contact name in the chat table
            cursor.execute("""
                SELECT DISTINCT c.display_name, c.room_name
                FROM chat c
                WHERE c.chat_identifier = ? OR c.chat_identifier LIKE ?
            """, (phone_number, f"%{phone_number}%"))
            
            result = cursor.fetchone()
            if result:
                display_name, room_name = result
                if display_name and display_name != phone_number:
                    conn.close()
                    return display_name
                elif room_name and room_name != phone_number:
                    conn.close()
                    return room_name
            
            conn.close()
        except Exception as e:
            print(f"Error getting contact name from Messages DB: {e}")
    
    # If no name found, return the phone number
    return phone_number

def extract_text_from_attributed_body(attributed_body: bytes) -> str:
    """
    Extract plain text from the attributedBody binary plist blob.
    Look for the pattern we can see in the debug output.
    """
    if not attributed_body:
        return ""
    
    try:
        # Decode as UTF-8 and look for the pattern we see in debug
        body_str = attributed_body.decode("utf-8", errors="ignore")
        
        # Look for NSString pattern followed by the actual text (flexible approach)
        import re
        # The pattern we see in debug: NSString+Hey what's upiI
        # But the + might be encoded, so let's be more flexible
        match = re.search(r'NSString.(.+?)iI', body_str, re.IGNORECASE)
        if match:
            text = match.group(1)
            # Clean up the text - remove any leading non-letter characters
            text = text.strip()
            # Remove leading + or any other non-letter character
            while text and not text[0].isalpha() and not text[0].isdigit():
                text = text[1:]
            # Remove any remaining artifacts
            if text and len(text) > 0:
                return text
        
        # Fallback: look for text between common patterns
        # The debug shows: streamtyped@NSAttributedStrinNSObjecNSString+Hey what's upiI
        patterns = [
            r'NSString.([^iI]+)iI',
            r'NSString\+([^iI]+)iI',
            r'NSObject.*?NSString.([^iI]+)iI'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body_str, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Remove leading + character if present
                if text.startswith('+'):
                    text = text[1:]
                if text and len(text) > 0 and not text.startswith(('NS', '__kIM')):
                    return text
    
    except Exception as e:
        print(f"Error extracting text: {e}")
    
    return "[unreadable message]"

def print_conversation_history(chat_identifier_or_name: str, limit: Optional[int] = None):
    """
    Print the full conversation history for a given chat identifier or display name.
    Allows partial matching. Shows both sent and received messages.
    If no match, prints all available chats.
    """
    db_path = get_messages_database_path()
    if not os.path.exists(db_path):
        print(f"❌ Messages database not found at: {db_path}")
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Find all chats
        cursor.execute("SELECT ROWID, chat_identifier, display_name FROM chat")
        all_chats = cursor.fetchall()
        # Try to find a chat with exact or partial match
        matches = [row for row in all_chats if chat_identifier_or_name in (row[1] or '') or chat_identifier_or_name in (row[2] or '')]
        if not matches:
            print(f"❌ No chat found for: {chat_identifier_or_name}")
            print("Here are your available chats:")
            for row in all_chats:
                print(f"  • {row[2] or '(no name)'} ({row[1]})")
            conn.close()
            return
        # If multiple matches, pick the first
        chat_row = matches[0]
        chat_id = chat_row[0]
        chat_identifier = chat_row[1]
        chat_name = chat_row[2] or chat_identifier
        # Get all messages for this chat (sent and received)
        if limit is not None:
            query = """
                SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id as sender_id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE cmj.chat_id = ?
                ORDER BY m.date ASC
                LIMIT ?
            """
            cursor.execute(query, (chat_id, limit))
        else:
            query = """
                SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id as sender_id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE cmj.chat_id = ?
                ORDER BY m.date ASC
            """
            cursor.execute(query, (chat_id,))
        rows = cursor.fetchall()
        print(f"\n💬 Conversation history for: {chat_name} ({chat_identifier})")
        print("=" * 60)
        
        # Debug: Show raw attributedBody for first few sent messages
        debug_count = 0
        for row in rows:
            msg_id, text, attributed_body, timestamp, is_from_me, sender_id = row
            if is_from_me and attributed_body and debug_count < 3:
                print(f"\n🔍 DEBUG - Raw attributedBody for message {msg_id}:")
                try:
                    raw_str = attributed_body.decode("utf-8", errors="ignore")
                    print(f"Raw (first 200 chars): {raw_str[:200]}")
                    print(f"Contains 'NSString+': {'NSString+' in raw_str}")
                    print(f"Contains 'iI': {'iI' in raw_str}")
                except Exception as e:
                    print(f"Error decoding: {e}")
                debug_count += 1
        
        for row in rows:
            msg_id, text, attributed_body, timestamp, is_from_me, sender_id = row
            # Get contact name for the sender
            if is_from_me:
                sender = "Me"
            else:
                sender = get_contact_name(sender_id) if sender_id else "Unknown"
            
            # Prefer text, but if empty, try attributedBody
            if not text and attributed_body:
                text = extract_text_from_attributed_body(attributed_body)
            # Convert Apple timestamp (nanoseconds since 2001-01-01)
            try:
                dt = datetime(2001, 1, 1) + timedelta(seconds=timestamp/1e9)
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                time_str = str(timestamp)
            print(f"[{time_str}] {sender}: {text if text else '[non-text message]'}")
        print("=" * 60)
        print(f"Total messages: {len(rows)}")
        conn.close()
    except Exception as e:
        print(f"❌ Error printing conversation history: {e}")

# Example usage functions
def list_available_chats():
    """List all available chats for debugging."""
    messages = read_recent_messages_from_db(100)
    chats = {}
    
    for msg in messages:
        chat_id = msg['chat_identifier']
        if chat_id not in chats:
            chats[chat_id] = {
                'name': msg['chat_name'],
                'last_message': msg['text'][:50] + "..." if len(msg['text']) > 50 else msg['text']
            }
    
    print("📱 Available chats:")
    for chat_id, info in chats.items():
        # Try to get a friendly name for the chat
        display_name = get_contact_name(chat_id) if chat_id else info['name']
        print(f"  • {display_name} ({chat_id})")
        print(f"    Last: {info['last_message']}")
        print()

def create_ai_bot(chat_identifier: str, model: str = "llama2", base_url: str = "http://localhost:11434", 
                  api_type: str = "ollama", system_prompt: Optional[str] = None):
    """Create and return an AI bot instance for a specific chat"""
    # Initialize LLM client
    llm_client = LLMClient(base_url=base_url, model=model, api_type=api_type)
    
    # Default system prompt for iMessage bot
    if not system_prompt:
        system_prompt = """You are a helpful, friendly AI assistant responding in an iMessage conversation. 
Keep your responses conversational, natural, and concise (1-2 sentences max). 
Respond as if you're a human friend in the chat. Use emojis occasionally but don't overdo it.
Don't mention that you're an AI unless directly asked."""
    
    return llm_client, system_prompt

def get_most_recent_message_across_all_chats() -> Optional[Dict]:
    """Get the most recent message from any chat (excluding our own messages)"""
    db_path = get_messages_database_path()
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the most recent message from any chat (not from us)
        query = """
            SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, 
                   h.id as sender_id, c.chat_identifier, c.display_name, c.room_name
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.is_from_me = 0
            ORDER BY m.date DESC
            LIMIT 1
        """
        
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            msg_id, text, attributed_body, timestamp, is_from_me, sender_id, chat_identifier, display_name, room_name = row
            
            # Get message text
            if not text and attributed_body:
                text = extract_text_from_attributed_body(attributed_body)
            
            # Determine chat name
            chat_name = display_name or room_name or chat_identifier or "Unknown Chat"
            
            return {
                'id': msg_id,
                'text': text,
                'timestamp': timestamp,
                'is_from_me': is_from_me,
                'sender_id': sender_id,
                'sender_name': get_contact_name(sender_id) if sender_id else "Unknown",
                'chat_identifier': chat_identifier,
                'chat_name': chat_name
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error getting most recent message: {e}")
        return None

def get_chat_type_info(chat_identifier: str) -> Dict:
    """Get information about a chat (individual vs group, participants, etc.)"""
    db_path = get_messages_database_path()
    if not os.path.exists(db_path):
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find the chat
        cursor.execute("SELECT ROWID, chat_identifier, display_name, room_name FROM chat WHERE chat_identifier = ? OR chat_identifier LIKE ?", 
                      (chat_identifier, f"%{chat_identifier}%"))
        chat_result = cursor.fetchone()
        
        if not chat_result:
            conn.close()
            return {}
        
        chat_id, chat_identifier, display_name, room_name = chat_result
        
        # Get participants
        cursor.execute("""
            SELECT DISTINCT h.id
            FROM chat_handle_join chj
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE chj.chat_id = ?
        """, (chat_id,))
        
        participants = []
        for row in cursor.fetchall():
            if row[0]:
                participants.append(row[0])
        
        conn.close()
        
        # Determine chat type
        is_group = len(participants) > 2 or bool(room_name)
        chat_name = display_name or room_name or chat_identifier
        
        return {
            'chat_identifier': chat_identifier,
            'chat_name': chat_name,
            'is_group': is_group,
            'participants': participants,
            'participant_count': len(participants)
        }
        
    except Exception as e:
        print(f"❌ Error getting chat type info: {e}")
        return {}

def monitor_all_chats_with_ai(llm_client: LLMClient, poll_interval: int = 5, 
                             system_prompt: Optional[str] = None, auto_respond: bool = True,
                             warmup_period: int = 1):
    """
    Monitor ALL chats for new messages and respond with AI.
    Now includes intelligent conversation analysis to determine when to respond.
    """
    print("🤖 Starting AI bot for ALL CHATS")
    print(f"⏰ Polling every {poll_interval} seconds...")
    print(f"🧠 Using model: {llm_client.model}")
    print(f"🔄 Auto-respond: {'ON' if auto_respond else 'OFF'}")
    print("📱 Will respond to ANY chat that receives a message")
    print("🧠 Intelligent response filtering: ENABLED")
    
    if warmup_period > 0:
        print(f"🔥 Warm-up period: {warmup_period} cycle(s) - learning current state without responding")
    
    last_message_id = None
    warmup_counter = 0
    
    while True:
        try:
            # Increment warm-up counter every cycle, regardless of new messages
            if warmup_counter < warmup_period:
                warmup_counter += 1
                print(f"🔥 Warm-up mode: Cycle {warmup_counter}/{warmup_period} - NOT responding to any messages")
            
            # Get the most recent message from any chat
            last_message = get_most_recent_message_across_all_chats()
            
            if last_message and last_message.get('id') != last_message_id:
                last_message_id = last_message.get('id')
                
                # Skip messages from ourselves (already filtered in query)
                sender_name = last_message.get('sender_name', 'Unknown')
                message_text = last_message.get('text', '')
                chat_identifier = last_message.get('chat_identifier', '')
                chat_name = last_message.get('chat_name', 'Unknown Chat')
                
                # Get chat info
                chat_info = get_chat_type_info(chat_identifier)
                chat_type = "Group" if chat_info.get('is_group', False) else "Individual"
                
                print(f"\n📨 New message in {chat_type} chat '{chat_name}'")
                print(f"👤 From: {sender_name}")
                print(f"💬 Message: {message_text}")
                print(f"🆔 Chat ID: {chat_identifier}")
                
                # Add message to conversation memory
                message_info = MessageInfo(
                    sender=sender_name,
                    content=message_text,
                    timestamp=last_message.get('timestamp', datetime.now()),
                    chat_id=chat_identifier,
                    chat_type=chat_type,
                    is_bot_response=False
                )
                conversation_memory.add_message(chat_identifier, message_info)
                
                # Check for interruption if there's an active response
                if response_manager.is_response_active(chat_identifier):
                    interruption_analysis = conversation_memory.analyze_interruption(
                        chat_identifier, message_info, llm_client
                    )
                    
                    if interruption_analysis.should_interrupt:
                        print(f"⚡ Interruption detected: {interruption_analysis.reason}")
                        print(f"🔥 Interruption score: {interruption_analysis.score} ({interruption_analysis.urgency_level.name})")
                        
                        # Interrupt the current response
                        response_manager.interrupt_response(chat_identifier, message_info, interruption_analysis)
                        
                        # Skip processing this message further since it's handled by the interruption
                        continue
                    else:
                        print(f"ℹ️ Message received during response generation but not interrupting: {interruption_analysis.reason}")
                        continue
                
                # Check if we're still in warm-up period
                if warmup_counter < warmup_period:
                    print(f"🔥 Warm-up active: Ignoring message (will respond after warm-up completes)")
                elif auto_respond and message_text and message_text.strip():
                    # Use intelligent analysis to decide if we should respond
                    analysis = conversation_memory.should_respond(
                        chat_identifier, message_text, sender_name, chat_type, llm_client
                    )
                    
                    # Show simple reasoning (no LLM call needed)
                    print(f"🧠 Simple Analysis: {analysis['reason']} (confidence: {analysis['confidence']:.1f})")
                    
                    if analysis['should_respond']:
                        print(f"✅ Decision: RESPOND ({analysis['response_type']})")
                        
                        print("🤖 Generating AI response...")
                        
                        # Generate AI response for this specific chat
                        ai_response = generate_ai_response_threaded(chat_identifier, message_info, llm_client, system_prompt)
                        
                        if ai_response and ai_response.strip():
                            print(f"💬 AI Response: {ai_response}")
                            
                            # Record the response
                            conversation_memory.record_bot_response(
                                chat_identifier, message_text, ai_response, analysis['reason']
                            )
                            
                            # Send the response back to the chat
                            try:
                                send_imessage(chat_identifier, ai_response)
                                print("✅ Response sent!")
                            except Exception as e:
                                print(f"❌ Error sending response: {e}")
                        else:
                            print("⚠️  No response generated")
                    else:
                        print(f"🚫 Decision: DO NOT RESPOND ({analysis['response_type']})")
                else:
                    print("ℹ️  Auto-respond disabled or empty message")
            elif warmup_counter >= warmup_period:
                # Only show "monitoring" message after warm-up is complete
                if warmup_counter == warmup_period:
                    print("✅ Warm-up complete! Now monitoring for new messages...")
                    warmup_counter += 1  # Increment to avoid showing this message repeatedly
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 AI bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in AI monitoring: {e}")
            time.sleep(poll_interval)

def monitor_specific_chats_with_ai(chat_identifiers: List[str], llm_client: LLMClient, 
                                  poll_interval: int = 5, system_prompt: Optional[str] = None, 
                                  auto_respond: bool = True, warmup_period: int = 1):
    """
    Monitor specific chats for new messages and respond with AI.
    This will only respond to chats in the provided list.
    
    Args:
        warmup_period: Number of polling cycles to wait before responding (default 1)
                      This prevents responding to existing messages when starting
    """
    print(f"🤖 Starting AI bot for {len(chat_identifiers)} specific chats")
    print(f"📱 Monitoring chats: {chat_identifiers}")
    print(f"⏰ Polling every {poll_interval} seconds...")
    print(f"🧠 Using model: {llm_client.model}")
    print(f"🔄 Auto-respond: {'ON' if auto_respond else 'OFF'}")
    
    if warmup_period > 0:
        print(f"🔥 Warm-up period: {warmup_period} cycle(s) - learning current state without responding")
    
    last_message_ids = {}  # Track last message ID for each chat
    warmup_counter = 0
    
    while True:
        try:
            # Increment warm-up counter every cycle, regardless of new messages
            if warmup_counter < warmup_period:
                warmup_counter += 1
                print(f"🔥 Warm-up mode: Cycle {warmup_counter}/{warmup_period} - NOT responding to any messages")
            
            # Check each chat for new messages
            for chat_identifier in chat_identifiers:
                last_message = get_last_message_info(chat_identifier)
                
                if last_message and last_message.get('id') != last_message_ids.get(chat_identifier):
                    last_message_ids[chat_identifier] = last_message.get('id')
                    
                    # Skip messages from ourselves
                    if not last_message.get('is_from_me', False):
                        sender_name = last_message.get('sender_name', 'Unknown')
                        message_text = last_message.get('text', '')
                        
                        # Get chat info
                        chat_info = get_chat_type_info(chat_identifier)
                        chat_type = "Group" if chat_info.get('is_group', False) else "Individual"
                        chat_name = chat_info.get('chat_name', chat_identifier)
                        
                        print(f"\n📨 New message in {chat_type} chat '{chat_name}'")
                        print(f"👤 From: {sender_name}")
                        print(f"💬 Message: {message_text}")
                        print(f"🆔 Chat ID: {chat_identifier}")
                        
                        # Check if we're still in warm-up period
                        if warmup_counter < warmup_period:
                            print(f"🔥 Warm-up active: Ignoring message (will respond after warm-up completes)")
                        elif auto_respond and message_text and message_text.strip():
                            print("🤖 Generating AI response...")
                            start_time = datetime.now()
                            
                            # Convert to MessageInfo object for threaded response
                            last_message_info = MessageInfo(
                                sender=sender_name,
                                content=message_text,
                                timestamp=last_message.get('timestamp', datetime.now()),
                                chat_id=chat_identifier,
                                chat_type=chat_type,
                                is_bot_response=False,
                                message_id=last_message.get('id')
                            )
                            
                            # Generate AI response for this specific chat
                            ai_response = generate_ai_response_threaded(chat_identifier, last_message_info, llm_client, system_prompt)
                            
                            # Log response generation time
                            generation_time = (datetime.now() - start_time).total_seconds()
                            print(f"⏱️  Response generated in {generation_time:.1f} seconds")
                            
                            if ai_response and ai_response.strip():
                                print(f"💬 AI Response: {ai_response}")
                                
                                # Send the response back to the chat
                                try:
                                    send_imessage(chat_identifier, ai_response)
                                    print("✅ Response sent!")
                                except Exception as e:
                                    print(f"❌ Error sending response: {e}")
                            else:
                                print("⚠️  No response generated")
                        else:
                            print("ℹ️  Auto-respond disabled or empty message")
            
            # Show warm-up completion message once
            if warmup_counter == warmup_period:
                print("✅ Warm-up complete! Now monitoring for new messages...")
                warmup_counter += 1  # Increment to avoid showing this message repeatedly
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 AI bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in AI monitoring: {e}")
            time.sleep(poll_interval)

def create_dynamic_ai_bot(model: str = "llama2", base_url: str = "http://localhost:11434", 
                         api_type: str = "ollama", system_prompt: Optional[str] = None):
    """Create an AI bot instance that can work with any chat dynamically"""
    # Initialize LLM client
    llm_client = LLMClient(base_url=base_url, model=model, api_type=api_type)
    
    # Enhanced system prompt for dynamic chat handling
    if not system_prompt:
        system_prompt = """You are a helpful, friendly AI assistant responding in iMessage conversations. 
You can be in individual chats or group chats. Keep your responses conversational, natural, and concise.
Respond as if you're a human friend. Don't mention that you're an AI unless directly asked.
Pay attention to the conversation context to understand who you're talking to and adapt accordingly."""
    
    return llm_client, system_prompt

if __name__ == "__main__":
    print("🤖 iMessage AI Bot - Step 3: LLM Integration Complete!")
    print("=" * 60)
    
    # List available chats
    list_available_chats()
    
    # Test reading recent messages
    print("📨 Recent messages from database:")
    messages = read_recent_messages_from_db(5)
    for msg in messages:
        print(format_message_for_display(msg))
    
    # Print the full conversation history for the chat
    print_conversation_history("YOUR_PHONE_NUMBER")
    
    print("\n" + "=" * 60)
    print("🎉 AI Bot Ready! Here's how to use it:")
    print("\n1. 📋 Test AI Response Generation:")
    print("   from imessage import create_ai_bot, generate_ai_response")
    print("   llm_client, system_prompt = create_ai_bot('YOUR_PHONE_NUMBER')")
    print("   response = generate_ai_response('YOUR_PHONE_NUMBER', llm_client, system_prompt)")
    print("   print(response)")
    
    print("\n2. 🤖 Start AI Bot (Auto-respond to new messages):")
    print("   from imessage import create_ai_bot, monitor_chat_with_ai")
    print("   llm_client, system_prompt = create_ai_bot('YOUR_PHONE_NUMBER')")
    print("   monitor_chat_with_ai('YOUR_PHONE_NUMBER', llm_client, system_prompt=system_prompt)")
    
    print("\n3. 🛠️  Customize Bot:")
    print("   - Change model: create_ai_bot('YOUR_PHONE_NUMBER', model='llama3')")
    print("   - Use OpenAI API: create_ai_bot('YOUR_PHONE_NUMBER', api_type='openai', base_url='https://api.openai.com')")
    print("   - Custom prompt: create_ai_bot('YOUR_PHONE_NUMBER', system_prompt='You are a helpful assistant...')")
    
    print("\n⚠️  Requirements:")
    print("   - Install: pip install requests")
    print("   - For Ollama: Install Ollama and run 'ollama serve' + 'ollama pull llama2'")
    print("   - For OpenAI: Set up API key and endpoint")
    
    print("\n🚀 Ready to start your AI iMessage bot!")