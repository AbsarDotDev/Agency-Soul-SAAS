@extends('layouts.admin')

@section('page-title')
    {{ __('AI Agent Chat') }}
@endsection

@push('css-page')
    <style>
        .chat-sidebar {
            height: calc(100vh - 180px); /* Adjust based on header/footer height */
            overflow-y: auto;
            border-right: 1px solid #dee2e6;
        }
        .chat-main {
            height: calc(100vh - 180px); /* Adjust based on header/footer height */
            display: flex;
            flex-direction: column;
        }
        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
            padding: 15px;
            margin-bottom: 15px;
            border-bottom: 1px solid #dee2e6;
        }
        .chat-input-area {
            padding: 15px;
            border-top: 1px solid #dee2e6;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #e6f7ff; /* Light blue */
            align-self: flex-end;
            margin-left: auto;
            border-bottom-right-radius: 0;
        }
        .agent-message {
            background-color: #f0f0f0; /* Light grey */
            align-self: flex-start;
            margin-right: auto;
            border-bottom-left-radius: 0;
        }
        
        /* Improved agent response styling */
        .agent-response {
            display: flex;
            gap: 10px;
        }
        
        .agent-avatar {
            width: 36px;
            height: 36px;
            flex-shrink: 0;
            background-color: #0d6efd; /* Primary blue */
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        
        .agent-content {
            flex-grow: 1;
            line-height: 1.5;
        }
        
        .avatar-icon {
            font-size: 18px;
        }
        
        .conversation-list .list-group-item {
            cursor: pointer;
        }
         .conversation-list .list-group-item.active {
            background-color: #0d6efd;
            color: white;
            border-color: #0d6efd;
        }
        .loading-indicator {
            display: none; /* Hidden by default */
            text-align: center;
            padding: 10px;
        }
         /* Simple spinner animation */
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s ease infinite;
            display: inline-block;
            margin-top: 5px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
         }

        /* Style for the floating AI chat button */
        #ai-chat-floating-button {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1050;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: var(--color-primary, #0d6efd); /* Use theme primary color */
            color: white;
             display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px; /* Adjust icon size */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease;
        }

        #ai-chat-floating-button:hover {
            background-color: var(--color-primary-darker, #0b5ed7); /* Slightly darker on hover */
        }

        /* Add icon using :before and Tabler Icons font */
        #ai-chat-floating-button:before {
            font-family: 'tabler-icons';
            content: "\\ea0f"; /* Unicode for ti-brain */
            font-style: normal;
            font-weight: normal;
            font-variant: normal;
            text-transform: none;
            line-height: 1;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
         }
         
        /* Visualization container responsive styling */
        .visualization-container {
            margin: 1rem 0;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .chart-wrapper {
            position: relative;
            margin: 0 auto;
            transition: height 0.3s ease;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .chart-wrapper {
                height: 200px !important;
                max-width: 100% !important;
            }
        }
        
        @media (max-width: 576px) {
            .chart-wrapper {
                height: 180px !important;
            }
        }

    </style>
@endpush

@push('script-page')
    {{-- Include Chart.js Library with specific version for better compatibility --}}
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js" integrity="sha256-+8WU/VeFMsRGTzCzRFVcMKz6Tg3lpHDy9O+7AMOt5jM=" crossorigin="anonymous"></script>
    {{-- Chart.js plugins for better visualization --}}
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>

    <script>
    (function ($) {
        // --- Constants ---
        const CHAT_API_URL = '{{ route('ai_agent.chat.post') }}';
        const CSRF_TOKEN = $('meta[name="csrf-token"]').attr('content');

        // --- DOM Elements ---
        const $chatHistory = $('#chat-history');
        const $messageInput = $('#message-input');
        const $sendButton = $('#send-button');
        const $conversationList = $('#conversation-list');
        const $loadingIndicator = $('#loading-indicator');
        const $tokenDisplay = $('#header-token-display'); // Assuming this ID exists in layout

        // --- State ---
        let currentConversationId = null; // Will be set when loading/starting a convo
        let isLoading = false;
        let chartJsLoaded = false;

        // --- Chart.js Initialization Check ---
        function checkChartJsLoaded() {
            if (typeof Chart !== 'undefined') {
                console.log("Chart.js is loaded successfully");
                chartJsLoaded = true;
                return true;
            }
            
            console.warn("Chart.js not loaded, attempting to load dynamically");
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js';
            script.onload = function() {
                console.log("Chart.js loaded dynamically");
                chartJsLoaded = true;
            };
            script.onerror = function() {
                console.error("Failed to load Chart.js dynamically");
                // Add a global alert
                $('<div class="alert alert-warning alert-dismissible fade show" role="alert">')
                    .html('Warning: Chart.js could not be loaded. Visualizations may not display properly. <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>')
                    .prependTo($('.chat-main'));
            };
            document.head.appendChild(script);
            return false;
        }

        // --- Functions ---

        // Display a message in the chat history
        function displayMessage(sender, text, agentType) {
            const messageClass = sender === 'user' ? 'user-message' : 'agent-message';
            const $messageDiv = $('<div>').addClass('message').addClass(messageClass);
            
            if (sender === 'agent') {
                // Create badge for agent type if available
                let agentBadge = '';
                if (agentType) {
                    const badgeColorClass = {
                        'sql': 'bg-primary',
                        'hrm': 'bg-success',
                        'finance': 'bg-info',
                        'crm': 'bg-warning',
                        'sales': 'bg-danger'
                    }[agentType.toLowerCase()] || 'bg-secondary';
                    
                    agentBadge = `<span class="badge ${badgeColorClass} ms-2">${agentType.toUpperCase()}</span>`;
                }
                
                // For agent messages, always call formatAgentResponse and render as HTML
                const formattedHtml = formatAgentResponse(text);
                $messageDiv.html(`
                    <div class="agent-response">
                        <div class="agent-avatar">
                            <span class="avatar-icon"><i class="ti ti-robot"></i></span>
                        </div>
                        <div class="agent-content">
                            <div class="d-flex align-items-center mb-1">
                                <span class="fw-bold">AI Assistant</span>
                                ${agentBadge}
                            </div>
                            ${formattedHtml} 
                        </div>
                    </div>
                `);
            } else {
                // For user messages, use simple text formatting
                $messageDiv.text(text);
            }
            
            $chatHistory.append($messageDiv);
            scrollToBottom();
        }

        // Scroll chat history to the bottom
        function scrollToBottom() {
            $chatHistory.scrollTop($chatHistory[0].scrollHeight);
        }

        // Show/hide loading indicator
        function setLoading(loading) {
            isLoading = loading;
            if (loading) {
                $loadingIndicator.show();
                $sendButton.prop('disabled', true);
                $messageInput.prop('disabled', true);
            } else {
                $loadingIndicator.hide();
                // Only enable if we have tokens available
                const tokensRemaining = parseInt($tokenDisplay.find('.remaining-tokens').text().replace(/,/g, '')) || 0;
                const shouldEnable = tokensRemaining > 0;
                $sendButton.prop('disabled', !shouldEnable);
                $messageInput.prop('disabled', !shouldEnable);
                if (shouldEnable) {
                    $messageInput.focus();
                }
            }
        }

        // Update token display in header (if element exists)
        function updateTokenDisplay(tokensRemaining) {
             if ($tokenDisplay.length && tokensRemaining !== undefined && tokensRemaining !== null) {
                $tokenDisplay.find('.remaining-tokens').text(tokensRemaining);
                 // Maybe add color coding based on remaining tokens
                 if (tokensRemaining < 500) { // Example threshold
                     $tokenDisplay.removeClass('text-success').addClass('text-warning');
                 } else if (tokensRemaining <= 0) {
                     $tokenDisplay.removeClass('text-success text-warning').addClass('text-danger');
                     // Disable inputs if we run out of tokens
                     $sendButton.prop('disabled', true);
                     $messageInput.prop('disabled', true);
                     // Show warning
                     const $warningDiv = $('<div class="alert alert-warning mt-2 mb-0 py-2"><i class="ti ti-alert-triangle me-1"></i>' + 
                         "{{ __('You have used all your tokens. Please contact your administrator.') }}" + '</div>');
                     $('.chat-input-area small').replaceWith($warningDiv);
                 } else {
                     $tokenDisplay.removeClass('text-warning text-danger').addClass('text-success');
                 }
            }
        }

        // Load conversation history
        function loadConversation(conversationId) {
            if (!conversationId || isLoading) {
                return;
            }
            setLoading(true);
            $chatHistory.empty(); // Clear current chat
            currentConversationId = conversationId;

            const historyUrl = `{{ url('/ai-agent/conversation') }}/${conversationId}`;
            console.log(`[loadConversation] Loading conversation history from: ${historyUrl}`);

            $.ajax({
                url: historyUrl,
                method: 'GET',
                headers: {
                    'X-CSRF-TOKEN': CSRF_TOKEN
                },
                success: function(response) {
                    console.log("[loadConversation] Full history response:", response);
                    if (response.messages && Array.isArray(response.messages)) {
                         if(response.messages.length === 0) {
                             displayMessage('agent', 'This conversation is empty or could not be loaded.', null);
                         } else {
                             // Log each message for debugging
                             response.messages.forEach((msg, index) => {
                                 console.log(`[loadConversation] Message ${index+1}:`, msg);
                                 displayMessage(msg.role, msg.content, msg.agent_type);
                                 
                                 // Check if this message has visualization data
                                 if (msg.visualization) {
                                     console.log(`[loadConversation] Found visualization in message ${index+1}:`, msg.visualization);
                                     try {
                                         // Debug the visualization data structure
                                         console.log(`[loadConversation] Visualization Chart Type: ${msg.visualization.chart_type}`);
                                         console.log(`[loadConversation] Visualization Labels: ${JSON.stringify(msg.visualization.labels)}`);
                                         console.log(`[loadConversation] Visualization Datasets: ${JSON.stringify(msg.visualization.datasets)}`);
                                         console.log(`[loadConversation] Visualization Options: ${JSON.stringify(msg.visualization.options)}`);
                                         
                                         displayVisualization(msg.visualization);
                                     } catch (err) {
                                        console.error(`[loadConversation] Error displaying visualization from history for message ${index+1}:`, err);
                                     }
                                 } else {
                                     console.log(`[loadConversation] No visualization data in message ${index+1}`);
                                 }
                             });
                         }
                     } else {
                        displayMessage('agent', 'Error: Could not load conversation history.', null);
                        console.error("[loadConversation] Invalid history response format:", response);
                    }
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    console.error("[loadConversation] Error loading history:", textStatus, errorThrown, jqXHR.responseText);
                    displayMessage('agent', 'Error: Failed to load conversation history. ' + (jqXHR.responseJSON?.error || ''), null);
                },
                complete: function() {
                    setLoading(false);
                    scrollToBottom();
                    $messageInput.focus();
                }
            });
        }

        // Core function to send message and handle response
        function sendMessage() {
            if (isLoading) return;
            
            const message = $messageInput.val().trim();
            if (!message) return;
            
            // Display the user's message
            displayMessage('user', message, null);
            $messageInput.val(''); // Clear input field
            
            // Set loading state
            setLoading(true);
            
            // Send request to backend
            $.ajax({
                url: '{{ route("ai_agent.chat.post") }}',
                type: 'POST',
                data: {
                    message: message,
                    conversation_id: currentConversationId,
                    _token: '{{ csrf_token() }}'
                },
                dataType: 'json',
                success: function (response) {
                    console.log("AI Response received:", response);
                    
                    // Update conversation ID if needed
                    if (response.conversation_id) {
                        currentConversationId = response.conversation_id;
                        
                        // Add to sidebar if not exists and has a title
                        if (response.conversation_title && !conversationExistsInSidebar(currentConversationId)) {
                            const cleanTitle = response.conversation_title.replace(/\s+-\s*\d+$/, '').trim();
                            addConversationToSidebar(currentConversationId, cleanTitle);
                        }
                    }
                    
                    // Update token display
                    if (response.tokens_remaining !== undefined) {
                        updateTokenDisplay(response.tokens_remaining);
                    }
                    
                    // Display agent response text
                    displayMessage('agent', response.text, response.agent_type);
                    
                    // Check for and display visualization if present
                    if (response.visualization) {
                        console.log("Visualization data found:", JSON.stringify(response.visualization));
                        try {
                            // Add a small delay to ensure the text response is fully rendered
                            setTimeout(() => {
                                displayVisualization(response.visualization);
                            }, 100);
                        } catch (err) {
                            console.error("Error displaying visualization:", err);
                            displayMessage('agent', 'There was an error displaying the visualization.', response.agent_type);
                        }
                    } else {
                        console.log("No visualization data in response. Full response:", JSON.stringify(response));
                    }
                },
                error: function (xhr, status, error) {
                    console.error("AJAX Error:", status, error);
                    
                    // Default error message
                    let errorMessage = '{{ __("Sorry, there was an error processing your request. Please try again.") }}';
                    
                    // Try to extract more specific error message if available
                    try {
                        const responseData = JSON.parse(xhr.responseText);
                        if (responseData && responseData.error) {
                            errorMessage = responseData.error;
                        } else if (responseData && responseData.message) {
                            errorMessage = responseData.message;
                        }
                    } catch (e) {
                        console.error("Error parsing error response:", e);
                    }
                    
                    displayMessage('agent', errorMessage, null);
                },
                complete: function () {
                    setLoading(false);
                }
            });
        }

        // Event listener for conversation list items
        $conversationList.on('click', '.list-group-item', function(e) {
            e.preventDefault(); // Prevent default anchor behavior
            const $this = $(this);
            const conversationId = $this.data('conversation-id');
            
            if (conversationId && conversationId !== currentConversationId && !isLoading) {
                loadConversation(conversationId);
                $conversationList.find('.list-group-item').removeClass('active');
                $this.addClass('active');
                    }
        });

        // --- Event Listeners ---
        $sendButton.on('click', sendMessage);

        $messageInput.on('keypress', function (e) {
            if (e.which === 13 && !e.shiftKey) { // Enter key without Shift
                    e.preventDefault();
                    sendMessage();
                }
            });
            
        // New Chat Button Logic
        $('#new-chat-button').on('click', function() {
            if (isLoading) return; // Don't allow new chat while loading

            console.log("Starting new chat...");
            currentConversationId = null;
            $chatHistory.empty(); // Clear current chat history display
            displayMessage('agent', '{{ __("Hello! How can I help you analyze your company's data today?") }}', null); // Display initial message
            $conversationList.find('.list-group-item').removeClass('active'); // Deselect sidebar item
            $messageInput.val(''); // Clear input field
            
            // Re-enable input if tokens are available
            setLoading(false); 
            $messageInput.focus();
        });

        // --- Initialization ---
        function initializeChat() {
             console.log("Chat Initializing...");
             
             // Check if Chart.js is loaded
             chartJsLoaded = checkChartJsLoaded();
             
             // Set focus and ensure correct initial state
             $messageInput.focus();
             setLoading(false); // Ensure loading is off initially
             updateTokenDisplay({{ $remainingTokens ?? 'null' }}); // Update with initial tokens passed from controller
             console.log("Initial Tokens:", {{ $remainingTokens ?? 'null' }});
        }

        // --- Run ---
        $(document).ready(initializeChat);

        // Function to format agent responses
        function formatAgentResponse(text) {
            if (!text) return 'No response received.';
            
            // Replace newlines with <br> tags
            let formatted = text.replace(/\n/g, '<br>');
            
            // Format numbers for better readability
            formatted = formatted.replace(/\b(\d{4,})\b/g, '<span class="font-semibold">$1</span>');
            
            // Make URLs clickable
            formatted = formatted.replace(
                /(https?:\/\/[^\s]+)/g, 
                '<a href="$1" target="_blank" class="text-blue-600 hover:underline">$1</a>'
            );
            
            return formatted;
        }

        // Find the part that handles agent responses and modify it to detect and format error messages better
        function displayAgentResponse(response) {
            const chatMessagesDiv = document.getElementById('chat-messages');
            const loadingMessage = document.getElementById('loading-message');
            
            // Hide loading indicator
            if (loadingMessage) {
                loadingMessage.style.display = 'none';
            }
            
            // Create response message element
            const messageElement = document.createElement('div');
            messageElement.className = 'chat-message agent mb-3';
            
            // Check if response is an error message about missing filters
            const isFilterError = response.includes('Query does not contain required company isolation filters');
            
            if (isFilterError) {
                // Enhanced formatting for filter errors
                const errorParts = response.split('Error:');
                const errorDetail = errorParts.length > 1 ? errorParts[1].trim() : response;
                
                messageElement.innerHTML = `
                    <div class="flex items-start">
                        <div class="avatar">
                            <span class="w-8 h-8 rounded-full flex items-center justify-center bg-red-100 text-red-500">
                                <i class="fas fa-exclamation-triangle"></i>
                            </span>
                        </div>
                        <div class="message p-3 ml-2 bg-red-50 rounded-lg shadow-sm">
                            <div class="text-red-700"><strong>Data Security Notice:</strong></div>
                            <p class="text-red-600">${errorDetail}</p>
                            <div class="mt-2 text-sm text-gray-600 italic">
                                Please try your question again. I'll make sure to properly secure your data.
                            </div>
                        </div>
                    </div>
                `;
            } else if (response.includes('Error:') || response.includes('error')) {
                // General error message formatting
                messageElement.innerHTML = `
                    <div class="flex items-start">
                        <div class="avatar">
                            <span class="w-8 h-8 rounded-full flex items-center justify-center bg-yellow-100 text-yellow-600">
                                <i class="fas fa-exclamation-circle"></i>
                            </span>
                        </div>
                        <div class="message p-3 ml-2 bg-yellow-50 rounded-lg shadow-sm">
                            <p class="text-gray-700">${response}</p>
                            <div class="mt-2 text-sm text-gray-600 italic">
                                Please try rephrasing your question or asking about something different.
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Normal response formatting
                messageElement.innerHTML = `
                    <div class="flex items-start">
                        <div class="avatar">
                            <span class="w-8 h-8 rounded-full flex items-center justify-center bg-blue-100 text-blue-500">
                                <i class="fas fa-robot"></i>
                            </span>
                        </div>
                        <div class="message p-3 ml-2 bg-blue-50 rounded-lg shadow-sm">
                            <p class="text-gray-700">${formatAgentResponse(response)}</p>
                        </div>
                    </div>
                `;
            }
            
            // Add to chat window
            chatMessagesDiv.appendChild(messageElement);
            
            // Scroll to bottom of chat
            scrollToBottom();
        }

        // Helper function to check if a conversation exists in the sidebar
        function conversationExistsInSidebar(conversationId) {
            return $conversationList.find(`[data-conversation-id="${conversationId}"]`).length > 0;
        }

        // Helper function to add a conversation to the sidebar
        function addConversationToSidebar(conversationId, title) {
            // Clean up the title by removing any trailing numbers or hyphens
            const cleanTitle = title.replace(/\s+-\s*\d+$/, '').trim();
            
            const $newConversation = $(`<a href="#" class="list-group-item list-group-item-action" data-conversation-id="${conversationId}">${cleanTitle}</a>`);
            $conversationList.prepend($newConversation);
            $conversationList.find('.list-group-item').removeClass('active');
            $newConversation.addClass('active');
        }

        // --- NEW: Function to display visualization ---
        function displayVisualization(vizData) {
            console.log("[displayVisualization] Called with data:", vizData);
            
            // Handle potential stringified JSON object
            if (typeof vizData === 'string') {
                try {
                    console.log("[displayVisualization] Converting string to JSON object");
                    vizData = JSON.parse(vizData);
                } catch (err) {
                    console.error("[displayVisualization] Failed to parse visualization string data:", err);
                    return;
                }
            }
            
            if (!vizData || !vizData.chart_type || !vizData.labels || !vizData.datasets) {
                console.error('[displayVisualization] Invalid visualization data received:', vizData);
                displayMessage('agent', 'Received invalid data for visualization.', null);
                return;
            }

            // Normalize the data to avoid common issues
            if (Array.isArray(vizData.options)) {
                vizData.options = {};
                console.log("[displayVisualization] Converted options from array to object");
            }

            // Generate unique ID for this chart
            const chartId = `chart-${Date.now()}`;
            console.log(`[displayVisualization] Creating chart container with ID: ${chartId}`);
            
            // Create a dedicated container with better styling for the visualization
            const $vizContainer = $('<div class="visualization-container p-3 my-3 border rounded bg-white shadow-sm"></div>');
            
            // Add a message indicating this is a visualization
            const $vizHeader = $('<div class="mb-2 text-sm text-gray-600"><i class="fas fa-chart-bar mr-1"></i> Generated visualization:</div>');
            
            // Add a title if available, with better styling
            const $title = vizData.title ? 
                $(`<h6 class="font-weight-bold text-center mb-2">${vizData.title}</h6>`) : '';
            
            // Create responsive chart wrapper
            const $chartWrapper = $('<div class="chart-wrapper" style="position: relative; height: 250px; width: 100%; max-width: 500px; margin: 0 auto;"></div>');
            
            // Create canvas with proper dimensions
            const $canvas = $(`<canvas id="${chartId}" style="max-width: 100%;"></canvas>`);
            
            // Add canvas to the wrapper
            $chartWrapper.append($canvas);

            // Assemble the container
            $vizContainer.append($vizHeader);
            $vizContainer.append($title);
            $vizContainer.append($chartWrapper);
            
            // Append container and ensure it's in the DOM before getting context
            $chatHistory.append($vizContainer);
            console.log("[displayVisualization] Appended viz container to chat history.");
            
            // Scroll to bottom to show the new visualization
            scrollToBottom();

            // Function to create the chart with retry capability
            function createChart(attempt = 1) {
                const canvas = document.getElementById(chartId);
                if (!canvas) {
                    console.error(`[displayVisualization] Canvas element #${chartId} NOT found on attempt ${attempt}`);
                    if (attempt < 3) {
                        console.log(`[displayVisualization] Retrying in ${attempt * 200}ms...`);
                        setTimeout(() => createChart(attempt + 1), attempt * 200);
                    } else {
                        $vizContainer.append(`
                            <div class="alert alert-warning mt-2">
                                <strong>Visualization error:</strong> Could not create the chart canvas after ${attempt} attempts.
                                <br><small>Please try refreshing the page.</small>
                            </div>
                        `);
                    }
                    return;
                }

                console.log(`[displayVisualization] Canvas element #${chartId} found. Creating chart...`);
                try {
                    // Ensure Chart.js is available
                    if (typeof Chart === 'undefined') {
                        console.error('[displayVisualization] Chart.js library is not available!');
                        
                        if (!chartJsLoaded) {
                            // Try loading Chart.js if not already attempted
                            const loaded = checkChartJsLoaded();
                            if (loaded) {
                                // If loaded successfully, retry after a brief delay
                                setTimeout(() => createChart(attempt + 1), 500);
                                return;
                            }
                        }
                        
                        $vizContainer.append('<div class="alert alert-danger mt-2">Chart library not loaded. Please refresh the page.</div>');
                        return;
                    }
                    
                    // Prepare chart configuration
                    const chartOptions = typeof vizData.options === 'object' && !Array.isArray(vizData.options) 
                        ? vizData.options : {};
                    
                    // Add responsive option if not set
                    chartOptions.responsive = true;
                    chartOptions.maintainAspectRatio = false;
                    
                    console.log(`[displayVisualization] Creating ${vizData.chart_type} chart with options:`, chartOptions);
                    
                    // Specific configurations based on chart type
                    if (vizData.chart_type === 'pie' || vizData.chart_type === 'doughnut') {
                        // For pie charts, center the legend and make it smaller
                        chartOptions.plugins = chartOptions.plugins || {};
                        chartOptions.plugins.legend = chartOptions.plugins.legend || {};
                        chartOptions.plugins.legend.position = 'bottom';
                        chartOptions.plugins.legend.labels = chartOptions.plugins.legend.labels || {};
                        chartOptions.plugins.legend.labels.boxWidth = 12;
                        chartOptions.plugins.legend.labels.font = { size: 11 };
                        
                        // Add percentage to tooltips
                        chartOptions.plugins.tooltip = chartOptions.plugins.tooltip || {};
                        chartOptions.plugins.tooltip.callbacks = chartOptions.plugins.tooltip.callbacks || {};
                        chartOptions.plugins.tooltip.callbacks.label = function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        };
                    } else if (vizData.chart_type === 'bar') {
                        // For bar charts, prevent overcrowding with custom layout options
                        chartOptions.plugins = chartOptions.plugins || {};
                        chartOptions.plugins.legend = chartOptions.plugins.legend || {};
                        chartOptions.plugins.legend.position = 'top';
                        
                        // Rotate labels if there are many
                        if (vizData.labels && vizData.labels.length > 5) {
                            chartOptions.scales = chartOptions.scales || {};
                            chartOptions.scales.x = chartOptions.scales.x || {};
                            chartOptions.scales.x.ticks = chartOptions.scales.x.ticks || {};
                            chartOptions.scales.x.ticks.maxRotation = 45;
                            chartOptions.scales.x.ticks.minRotation = 45;
                        }
                    }
                    
                    // Add animation for better UX
                    chartOptions.animation = {
                        duration: 800,
                        easing: 'easeOutQuad'
                    };
                    
                    // Add tooltip configuration
                    chartOptions.plugins = chartOptions.plugins || {};
                    chartOptions.plugins.tooltip = chartOptions.plugins.tooltip || {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y || context.parsed || context.raw || '';
                                return label;
                            }
                        }
                    };
                    
                    console.log("[displayVisualization] Chart config:", {
                        type: vizData.chart_type,
                        data: {
                            labels: vizData.labels,
                            datasets: vizData.datasets
                        },
                        options: chartOptions
                    });
                    
                    // Create the chart
                    const chart = new Chart(canvas, {
                        type: vizData.chart_type,
                        data: {
                            labels: vizData.labels,
                            datasets: vizData.datasets
                        },
                        options: chartOptions
                    });
                    
                    console.log(`[displayVisualization] Chart #${chartId} created successfully.`);
                    
                    // Add a compact data summary below the chart for accessibility
                    let dataSummary = '<div class="mt-2 p-2 bg-light rounded text-muted small" style="font-size: 0.85rem;">';
                    
                    // Calculate total for percentage if applicable
                    let total = 0;
                    if (vizData.chart_type === 'pie' || vizData.chart_type === 'doughnut') {
                        total = vizData.datasets[0].data.reduce((sum, val) => sum + val, 0);
                    }
                    
                    // Format the summary based on chart type
                    if (vizData.chart_type === 'pie' || vizData.chart_type === 'doughnut') {
                        // For pie charts, show percentages
                        dataSummary += '<div class="row gx-1">';
                        vizData.labels.forEach((label, i) => {
                            const value = vizData.datasets[0].data[i];
                            const percent = Math.round((value / total) * 100);
                            const bgColor = vizData.datasets[0].backgroundColor[i] || '#ccc';
                            
                            dataSummary += `
                                <div class="col-6 col-sm-4 mb-1">
                                    <span class="d-inline-block me-1" style="width:10px; height:10px; background-color:${bgColor}"></span>
                                    <strong>${label}:</strong> ${value} (${percent}%)
                                </div>`;
                        });
                        dataSummary += '</div>';
                    } else {
                        // For other charts, simple list
                        dataSummary += '<div><strong>Data:</strong> ';
                        dataSummary += vizData.labels.map((label, i) => {
                            const value = vizData.datasets[0].data[i];
                            return `${label}: ${value}`;
                        }).join(', ');
                        dataSummary += '</div>';
                    }
                    
                    dataSummary += '</div>';
                    $vizContainer.append(dataSummary);
                    
                } catch (error) {
                    console.error(`[displayVisualization] Error creating chart #${chartId}:`, error);
                    
                    // Try to load Chart.js if it failed
                    if (typeof Chart === 'undefined' && attempt < 2) {
                        console.log("[displayVisualization] Attempting to load Chart.js dynamically...");
                        const script = document.createElement('script');
                        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
                        script.onload = function() {
                            console.log("[displayVisualization] Chart.js loaded successfully, retrying...");
                            createChart(attempt + 1);
                        };
                        document.head.appendChild(script);
                        return;
                    }
                    
                    // Display an error message in the visualization container
                    $vizContainer.append(`
                        <div class="alert alert-danger mt-2">
                            <strong>Error creating chart:</strong> ${error.message}
                            <br><small>Please try a different visualization type.</small>
                            <button class="btn btn-sm btn-outline-primary mt-2 retry-chart-btn" data-chart-id="${chartId}">
                                Retry
                            </button>
                        </div>
                    `);
                    
                    // Add retry handler
                    $vizContainer.find('.retry-chart-btn').on('click', function() {
                        $(this).prop('disabled', true).text('Retrying...');
                        createChart(attempt + 1);
                    });
                }
            }
            
            // Start chart creation with a small delay
            setTimeout(() => createChart(1), 300);
        }
        // --- END NEW Function ---

    })(jQuery);
    </script>

@endpush

@section('content')
    <div class="row">
        <div class="col-3">
            {{-- Sidebar for Conversation History --}}
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">{{__('Conversations')}}</h5>
                </div>
                <div class="chat-sidebar">
                    <div class="list-group list-group-flush conversation-list" id="conversation-list">
                        @forelse ($conversations as $convo)
                            {{-- Assuming $convo has 'id' and 'title' keys --}}
                            <a href="#" class="list-group-item list-group-item-action" data-conversation-id="{{ $convo['conversation_id'] }}">
                                {{ preg_replace('/\s+-\s*\d+$/', '', $convo['title'] ?? 'Conversation ' . $loop->iteration) }}
                            </a>
                        @empty
                            <div class="list-group-item text-muted text-center">{{__('No conversations yet.')}}</div>
                        @endforelse
                    </div>
                </div>
                 <div class="card-footer text-center">
                    {{-- Button to start a new chat - could clear currentConversationId and chat history --}}
                    <button class="btn btn-primary btn-sm w-100" id="new-chat-button">
                        <i class="ti ti-plus me-1"></i>{{__('New Chat')}}
                    </button>
                </div>
            </div>
                    </div>

        <div class="col-9">
            {{-- Main Chat Area --}}
            <div class="card chat-main">
                {{-- Chat History Display --}}
                <div class="chat-history" id="chat-history">
                    {{-- Messages will be appended here by JavaScript --}}
                     <div class="message agent-message">
                        @if($tokens_remaining <= 0)
                            {{ __("You have used all your available tokens. Please contact your administrator to add more tokens to your account.") }}
                        @else
                            {{ __("Hello! How can I help you analyze your company's data today?") }}
                        @endif
                     </div>
                </div>

                 {{-- Loading Indicator --}}
                <div class="loading-indicator" id="loading-indicator">
                     <div class="spinner"></div>
                     <div>{{ __('Waiting for response...') }}</div>
                 </div>

                {{-- Message Input Area --}}
                <div class="chat-input-area">
                     {{-- Removed form tag as AJAX handles submission --}}
                     <div class="input-group">
                        <textarea class="form-control" id="message-input" rows="3" placeholder="{{__('Type your message here...')}}" {{ $tokens_remaining <= 0 ? 'disabled' : '' }}></textarea>
                        <button class="btn btn-primary" type="button" id="send-button" {{ $tokens_remaining <= 0 ? 'disabled' : '' }}>
                            <i class="ti ti-send"></i> {{-- Assuming ti icons --}}
                        </button>
                    </div>
                    @if($tokens_remaining <= 0)
                        <div class="alert alert-warning mt-2 mb-0 py-2">
                            <i class="ti ti-alert-triangle me-1"></i> 
                            {{ __('You need tokens to use the AI agent. Please contact your administrator.') }}
                        </div>
                    @else
                        <small class="text-muted mt-1 d-block">{{ __('Shift+Enter for newline.') }}</small>
                    @endif
                </div>
            </div>
        </div>
    </div>
@endsection 