@extends('layouts.admin')

@section('page-title')
    {{ __('AI Agent Chat') }}
@endsection

@push('css-page')
    {{-- Add specific CSS for chat interface if needed --}}
    <style>
        #ai-chat-area {
            height: 60vh;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 15px;
            margin-bottom: 15px;
        }
        .chat-message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 10px;
            max-width: 80%;
        }
        .user-message {
            background-color: #d1e7fd;
            margin-left: auto;
            text-align: right;
        }
        .agent-message {
            background-color: #f8f9fa;
            margin-right: auto;
            text-align: left;
        }
        .message-sender {
             font-weight: bold;
             margin-bottom: 4px;
             font-size: 0.9em;
         }
         #ai-input-area {
             display: flex;
         }
         #ai-message-input {
             flex-grow: 1;
             margin-right: 10px;
         }
    </style>
@endpush

@push('script-page')
    {{-- Add JavaScript for sending messages and handling responses later --}}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const sendButton = document.getElementById('ai-send-button');
            const messageInput = document.getElementById('ai-message-input');
            const chatArea = document.getElementById('ai-chat-area');
            let conversationId = null; // Variable to store the current conversation ID

            // Function to add a message to the chat area
            function addMessage(sender, text, type, messageId = null) {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('chat-message', type + '-message');
                if (messageId) {
                    messageDiv.id = messageId;
                }
                
                const senderDiv = document.createElement('div');
                senderDiv.classList.add('message-sender');
                senderDiv.textContent = sender;
                
                const textDiv = document.createElement('div');
                // Basic Markdown support (bold, italics, code)
                text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
                text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');       // Italics
                text = text.replace(/`(.*?)`/g, '<code>$1</code>');       // Inline code
                // Convert newlines to <br>
                text = text.replace(/\n/g, '<br>');
                textDiv.innerHTML = text; // Use innerHTML to render basic markdown
                
                messageDiv.appendChild(senderDiv);
                messageDiv.appendChild(textDiv);
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight; // Scroll to bottom
                return messageDiv;
            }

            // Function to update an existing message
            function updateMessage(messageElement, newText) {
                 if (messageElement) {
                    const textDiv = messageElement.querySelector(':scope > div:nth-child(2)');
                    if(textDiv){
                        // Render markdown in updated message too
                        newText = newText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                        newText = newText.replace(/\*(.*?)\*/g, '<em>$1</em>');
                        newText = newText.replace(/`(.*?)`/g, '<code>$1</code>');
                        newText = newText.replace(/\n/g, '<br>');
                        textDiv.innerHTML = newText;
                        chatArea.scrollTop = chatArea.scrollHeight;
                    }
                }
            }

            // Function to handle sending message
            async function sendMessage() {
                 const messageText = messageInput.value.trim();
                if (messageText) {
                    addMessage('You', messageText, 'user');
                    messageInput.value = '';
                    sendButton.disabled = true;
                    messageInput.disabled = true;

                    // Add a temporary "Thinking..." message
                    const thinkingMessageId = 'agent-thinking-' + Date.now();
                    const thinkingMessageElement = addMessage('AI Agent', 'Thinking...', 'agent', thinkingMessageId);

                    try {
                        const response = await fetch("{{ route('ai_agent.process') }}", {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRF-TOKEN': "{{ csrf_token() }}", // Crucial for Laravel POST requests
                                'Accept': 'application/json'
                            },
                            body: JSON.stringify({
                                message: messageText,
                                conversation_id: conversationId // Send current conversation ID
                            })
                        });

                        const data = await response.json();

                        if (response.ok) {
                            // Update the "Thinking..." message with the actual response
                            updateMessage(thinkingMessageElement, data.message);
                            // Store the conversation ID from the response for the next message
                            conversationId = data.conversation_id;
                        } else {
                            // Update the "Thinking..." message with the error
                            updateMessage(thinkingMessageElement, `Error: ${data.error || 'Failed to get response.'}`);
                            console.error("API Error:", data);
                        }

                    } catch (error) {
                        console.error('Fetch Error:', error);
                        // Update the "Thinking..." message with the fetch error
                         updateMessage(thinkingMessageElement, 'Error: Could not reach the AI service.');
                    } finally {
                         sendButton.disabled = false;
                         messageInput.disabled = false;
                         messageInput.focus();
                    }
                }
            }

            // Event listeners
            sendButton.addEventListener('click', sendMessage);

            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
             // Add initial welcome message
             addMessage('AI Agent', 'Hello! How can I help you today?', 'agent');
        });
    </script>
@endpush

@section('content')
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-body">
                    <div id="ai-chat-area">
                        {{-- Chat messages will appear here --}}
                    </div>
                    <div id="ai-input-area">
                        <textarea id="ai-message-input" class="form-control" placeholder="{{ __('Type your message here...') }}" rows="2"></textarea>
                        <button id="ai-send-button" class="btn btn-primary">{{ __('Send') }}</button>
                    </div>
                    <small class="text-muted mt-2">{{ __('Note: AI responses are generated based on available data and may require verification.') }}</small>
                     {{-- Placeholder for token count display --}}
                     <div id="ai-token-info" class="mt-2 text-muted small">
                         {{-- Tokens remaining: X --}}
                     </div>
                </div>
            </div>
        </div>
    </div>
@endsection 