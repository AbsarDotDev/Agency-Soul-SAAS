<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use App\Models\User;
use Illuminate\Support\Facades\DB;

class AIAgentController extends Controller
{
    /**
     * Helper function to get/refresh AI Agent API token.
     * Stores the token in session to avoid re-fetching on every request.
     */
    private function getOrRefreshAiAgentToken(User $company, User $user)
    {
        $sessionKey = 'ai_agent_token_' . $company->id;
        $expiryKey = 'ai_agent_token_expiry_' . $company->id;

        $token = session($sessionKey);
        $expiry = session($expiryKey);

        // Check if token exists and is not (too close to) expired
        // A more robust solution might involve decoding the JWT if possible on Laravel side to check 'exp'
        if ($token && $expiry && now()->timestamp < ($expiry - 120)) { // 120 seconds buffer
            Log::debug('Using cached AI Agent token from session.', ['company_id' => $company->id]);
            return $token;
        }

        $agentApiUrl = config('services.ai_agent.url'); // Recommended: move to services config
        if (!$agentApiUrl) {
            $agentApiUrl = config('app.ai_agent_url'); // Fallback to old config key
        }

        if (!$agentApiUrl) {
            Log::error('AI Agent API URL is not configured (checked services.ai_agent.url and app.ai_agent_url).');
            return null;
        }

        Log::info('Attempting to fetch new AI Agent token.', ['company_id' => $company->id, 'user_id' => $user->id]);
        try {
            $internalApiKey = config('services.ai_agent.internal_api_key');
            if (!$internalApiKey) {
                Log::error('Internal API Key for AI Agent service is not configured in Laravel (services.ai_agent.internal_api_key).');
                return null;
            }

            $authResponse = Http::timeout(15)->post($agentApiUrl . '/api/auth', [ 
                'company_id' => $company->id,
                'user_id' => (string) $user->id,
                'api_key' => $internalApiKey, // Send the configured internal API key
            ]);

            if ($authResponse->successful() && $authResponse->json('access_token')) {
                $newToken = $authResponse->json('access_token');
                
                // Attempt to get expiry from Python settings if possible, otherwise use a default.
                // The Python's ACCESS_TOKEN_EXPIRE_MINUTES is the source of truth.
                // For simplicity here, we'll use a configurable value on Laravel side or a default.
                $tokenLifetimeMinutes = config('services.ai_agent.token_lifetime_minutes', 60 * 24 * 7); // Default 7 days

                session([$sessionKey => $newToken]);
                session([$expiryKey => now()->addMinutes($tokenLifetimeMinutes)->timestamp]);
                
                Log::info('Successfully fetched and cached new AI Agent token.', ['company_id' => $company->id]);
                return $newToken;
            } else {
                Log::error('Failed to fetch AI Agent token from /api/auth.', [
                    'status' => $authResponse->status(),
                    'body' => $authResponse->body(),
                    'company_id' => $company->id,
                ]);
                return null;
            }
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('ConnectionException while fetching AI Agent token.', [
                'error' => $e->getMessage(),
                'company_id' => $company->id
            ]);
            return null;
        } catch (\Exception $e) {
            Log::error('Generic Exception while fetching AI Agent token.', [
                'error' => $e->getMessage(),
                'company_id' => $company->id
            ]);
            return null;
        }
    }

    /**
     * Show the AI Agent chat interface.
     *
     * @return \Illuminate\View\View
     */
    public function showChat()
    {
        // Middleware already ensures the user/company has AI agent enabled in their plan.
        $user = Auth::user();
        $company = ($user->type == 'company') ? $user : User::find($user->created_by);
        
        // Get the plan
        $plan = \App\Models\Plan::find($company->plan);
        
        // Check if AI agent is enabled in the plan - this is redundant since middleware already checks,
        // but we include it for code clarity and in case middleware changes
        $ai_feature_enabled = $plan && $plan->ai_agent_enabled == 1;

        // Fetch conversation history for the sidebar
        // We need to get one title per conversation_id, preferably the one from the first message.
        // And order by the most recent message in each conversation group.
        
        // Subquery to get the first non-null title for each conversation_id
        $first_titles = \App\Models\AgentConversation::select('conversation_id', DB::raw('(SELECT title 
                                                                                          FROM agent_conversations ac2 
                                                                                          WHERE ac2.conversation_id = agent_conversations.conversation_id 
                                                                                          AND ac2.title IS NOT NULL 
                                                                                          ORDER BY ac2.created_at ASC 
                                                                                          LIMIT 1) as conversation_title'))
            ->where('company_user_id', $company->id)
            ->groupBy('conversation_id');

        // Main query to get the list of conversations, ordered by the last message in each.
        $conversations = \App\Models\AgentConversation::select('agent_conversations.conversation_id', 'ft.conversation_title as title', DB::raw('MAX(agent_conversations.created_at) as last_message_at'))
            ->joinSub($first_titles, 'ft', function ($join) {
                $join->on('agent_conversations.conversation_id', '=', 'ft.conversation_id');
            })
            ->where('agent_conversations.company_user_id', $company->id)
            ->groupBy('agent_conversations.conversation_id', 'ft.conversation_title')
            ->orderBy('last_message_at', 'desc')
            ->limit(20)
            ->get();

        // Calculate remaining tokens based on plan allocation and user usage
        $tokens_allocated = $plan ? $plan->ai_agent_default_tokens : 0;
        $tokens_remaining = max(0, $tokens_allocated - $company->ai_agent_tokens_used);

        return view('ai_agent.chat', compact('conversations', 'tokens_remaining', 'ai_feature_enabled'));
    }

    /**
     * Handle incoming chat messages for the AI Agent.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function chat(Request $request)
    {
        $user = Auth::user();
        $company = null;
        $plan = null;

        // Validate input
        $validated = $request->validate([
            'message' => 'required|string',
            'conversation_id' => 'nullable|string|uuid',
        ]);

        // Identify the company user based on the logged-in user's role
        if ($user->type == 'company') {
            $company = $user;
        } elseif ($user->created_by) {
            // Assuming regular users have a created_by pointing to the company user
            $company = User::find($user->created_by);
        }

        if (!$company || $company->type !== 'company') {
            Log::warning('AI Agent chat attempt by invalid user or company structure.', ['user_id' => $user->id]);
            return response()->json(['error' => 'Unauthorized or invalid company context.'], 403);
        }

        // Get the plan
        $plan = \App\Models\Plan::find($company->plan);
        
        // Check if AI agent is enabled in the plan
        if (!$plan || $plan->ai_agent_enabled != 1) {
            return response()->json(['error' => 'AI Agent is not enabled for your subscription plan.'], 403);
        }

        // Check if the company has tokens remaining based on plan allocation
        $tokens_allocated = $plan->ai_agent_default_tokens;
        if ($tokens_allocated <= $company->ai_agent_tokens_used) {
            return response()->json(['error' => 'You have used all your AI Agent tokens. Please contact the administrator to add more tokens.'], 403);
        }

        $aiAgentToken = $this->getOrRefreshAiAgentToken($company, $user);
        if (!$aiAgentToken) {
            Log::error('Failed to obtain AI Agent token for chat.', ['company_id' => $company->id, 'user_id' => $user->id]);
            return response()->json(['error' => 'Could not authenticate with the AI Agent service. Please try again later.'], 503); // Service Unavailable
        }

        $agentApiUrl = config('services.ai_agent.url'); // Recommended: move to services config
        if (!$agentApiUrl) {
            $agentApiUrl = config('app.ai_agent_url'); // Fallback to old config key
        }
        
        if (!$agentApiUrl) {
            Log::error('AI Agent API URL is not configured for chat.');
            return response()->json(['error' => 'AI Agent service is not configured correctly.'], 500);
        }

        try {
            $response = Http::timeout(120) // Increased timeout for potentially long LLM responses
                ->withToken($aiAgentToken) // Send the JWT
                ->post($agentApiUrl . '/api/message', [ // Path confirmed earlier
                    'message' => $validated['message'],
                    'company_id' => $company->id,
                    'user_id' => (string) $user->id, 
                    'conversation_id' => $validated['conversation_id'] ?? null,
                    // 'agent_type' is now determined by the Python dispatcher
                ]);

            if ($response->failed()) {
                Log::error('AI Agent API request failed.', [
                    'status' => $response->status(),
                    'body' => $response->body(),
                    'company_id' => $company->id,
                    'user_id' => $user->id,
                ]);
                // Attempt to parse the error from the AI service
                $errorDetail = $response->json('detail', 'Error communicating with the AI Agent service.');

                // --- Log Raw Response Body ---
                Log::info('Raw AI Agent Response Body:', ['body' => $response->body()]);
                
                // Attempt to parse the visualization data specifically for debugging
                $responseData = $response->json();
                if (isset($responseData['visualization'])) {
                    Log::info('Visualization data found in response:', [
                        'visualization_type' => $responseData['visualization']['chart_type'] ?? 'unknown',
                        'has_labels' => isset($responseData['visualization']['labels']),
                        'has_datasets' => isset($responseData['visualization']['datasets']),
                        'options_type' => is_array($responseData['visualization']['options']) ? 'array' : (is_object($responseData['visualization']['options']) ? 'object' : 'other')
                    ]);
                } else {
                    Log::warning('No visualization data found in AI Agent response');
                }
                // --- End Log Raw Response Body ---

                return response()->json(['error' => $errorDetail], $response->status() >= 500 ? 503 : $response->status());
            }

            // --- Log Raw Response Body ---
            Log::info('Raw AI Agent Response Body:', ['body' => $response->body()]);
            
            // --- Token Usage Update Logic ---
            $responseData = $response->json();
            Log::info('Parsed AI Agent Response Data:', ['parsed_data' => $responseData]);

            $visualizationData = null;
            if (isset($responseData['visualization']) && is_array($responseData['visualization'])) {
                $visualizationData = $responseData['visualization'];
                if (isset($visualizationData['options']) && (is_array($visualizationData['options']) && empty($visualizationData['options']))) {
                    $visualizationData['options'] = new \stdClass();
                }
            } else {
                Log::warning('No visualization data found in AI Agent response');
            }
            
            // Python service returns 'token_usage' for the current request.
            $tokensConsumedInRequest = 0;
            if (isset($responseData['token_usage'])) {
                $tokensConsumedInRequest = (int) $responseData['token_usage'];
            } else {
                // This log was already present, but we should ensure we handle the case.
                Log::warning('AI Agent response did not include "token_usage" field. Assuming 0 for this request for local count.', [
                    'company_id' => $company->id,
                    'response_body' => $responseData
                ]);
            }

            // Update the company's total used tokens in Laravel's database
            // The Python agent itself does not update Laravel's users.ai_agent_tokens_used.
            // It operates based on the token count passed in its JWT and its own internal logic.
            if ($tokensConsumedInRequest > 0) {
                $company->ai_agent_tokens_used += $tokensConsumedInRequest;
                $company->save();
                Log::info('Updated company token usage in Laravel DB.', [
                    'company_id' => $company->id, 
                    'tokens_added_to_used' => $tokensConsumedInRequest,
                    'new_total_used' => $company->ai_agent_tokens_used
                ]);
            }
            
            // The Python service returns 'tokens_remaining' which is the authoritative count after its processing.
            // Use this value directly for the frontend.
            $finalTokensRemaining = $responseData['tokens_remaining'] ?? ($tokens_allocated - $company->ai_agent_tokens_used);

            // Frontend expects 'text' for the message. Python sends 'message'.
            return response()->json([
                'text' => $responseData['response'] ?? 'Error: No response text received from agent.', // Map 'response' to 'text'
                'conversation_id' => $responseData['conversation_id'] ?? null,
                'visualization' => $visualizationData,
                'tokens_used_in_request' => $tokensConsumedInRequest, // For frontend to know what this call consumed
                'tokens_remaining' => $finalTokensRemaining,          // The authoritative remaining tokens after this call
                'new_title_generated' => $responseData['conversation_title'] ?? null,
                'agent_type' => $responseData['agent_type'] ?? 'sql'  // Include agent type, default to 'sql' if not provided
            ]);

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Could not connect to AI Agent API.', ['error' => $e->getMessage()]);
            return response()->json(['error' => 'Could not connect to the AI Agent service.'], 504); // Gateway Timeout
        } catch (\Exception $e) {
            Log::error('Error during AI Agent chat processing.', [
                'error' => $e->getMessage(),
                'company_id' => $company->id,
                'user_id' => $user->id,
            ]);
            return response()->json(['error' => 'An unexpected error occurred.'], 500);
        }
    }

    /**
     * Get the message history for a specific conversation.
     *
     * @param  string $id The conversation_id
     * @return \Illuminate\Http\JsonResponse
     */
    public function getConversationHistory(string $id)
    {
        $user = Auth::user();
        $company = null;

        // Identify the company user
        if ($user->type == 'company') {
            $company = $user;
        } elseif ($user->created_by) {
            $company = User::find($user->created_by);
        }

        if (!$company || $company->type !== 'company') {
            return response()->json(['error' => 'Unauthorized'], 403);
        }
        
        // Check if AI Agent is enabled for the plan (consistency check)
        $plan = \App\Models\Plan::find($company->plan);
        if (!$plan || $plan->ai_agent_enabled != 1) {
            return response()->json(['error' => 'AI Agent is not enabled for your subscription plan.'], 403);
        }

        // Get token for AI Agent service
        $aiAgentToken = $this->getOrRefreshAiAgentToken($company, $user);
        if (!$aiAgentToken) {
            Log::warning('Failed to obtain AI Agent token for getConversationHistory, falling back to DB.', ['company_id' => $company->id, 'user_id' => $user->id]);
            // Fall back to database if token cannot be obtained for agent service
            return $this->getConversationHistoryFromDatabase($id, $company->id);
        }

        $agentApiUrl = config('services.ai_agent.url');
        if (!$agentApiUrl) {
            $agentApiUrl = config('app.ai_agent_url');
        }

        if (!$agentApiUrl) {
            Log::error('AI Agent API URL is not configured for history retrieval, falling back to DB.');
            return $this->getConversationHistoryFromDatabase($id, $company->id);
        }

        try {
            // Try fetching from agent service first
            // The Python service's /api/conversation/{id} endpoint might need to be created or adjusted.
            // For now, assuming it exists and is protected.
            $response = Http::timeout(30) 
                ->withToken($aiAgentToken) // Send the JWT
                ->get($agentApiUrl . '/api/conversation/' . $id, [ // Assuming this is the correct endpoint
                    // 'company_id' => $company->id, // company_id is in the token, not needed in query params for GET
                ]);

            if ($response->failed()) {
                Log::error('AI Agent history API request failed, falling back to database.', [
                    'status' => $response->status(),
                    'body' => $response->body(),
                    'conversation_id' => $id,
                    'company_id' => $company->id,
                ]);
                
                // Fall back to database
                return $this->getConversationHistoryFromDatabase($id, $company->id);
            }

            // Process the response from the agent service
            $responseData = $response->json();
            
            // Log the visualization data for debugging
            if (isset($responseData['messages'])) {
                foreach ($responseData['messages'] as $msg) {
                    if (isset($msg['visualization'])) {
                        Log::info("API response contains visualization data for message", [
                            'conversation_id' => $id,
                            'visualization_type' => $msg['visualization']['chart_type'] ?? 'unknown'
                        ]);
                    }
                }
            }
            
            return response()->json($responseData);

        } catch (\Exception $e) {
            Log::error('Error retrieving AI Agent conversation history, falling back to database.', [
                'error' => $e->getMessage(),
                'conversation_id' => $id,
                'company_id' => $company->id,
            ]);
            
            // Fall back to database if API fails
            return $this->getConversationHistoryFromDatabase($id, $company->id);
        }
    }

    /**
     * Get conversation history directly from the database.
     * This is a fallback method when the agent service is unavailable.
     * 
     * @param string $conversationId
     * @param int $companyId
     * @return \Illuminate\Http\JsonResponse
     */
    private function getConversationHistoryFromDatabase(string $conversationId, int $companyId)
    {
        try {
            $conversations_data = \App\Models\AgentConversation::where('conversation_id', $conversationId)
                ->where('company_user_id', $companyId)
                ->orderBy('created_at', 'asc')
                ->get();

            if ($conversations_data->isEmpty()) {
                return response()->json(['messages' => []]);
            }

            // Format messages for the frontend
            $messages = [];
            foreach ($conversations_data as $conv) {
                // Add user message
                $messages[] = [
                    'role' => 'user',
                    'content' => $conv->message,
                    'timestamp' => $conv->created_at ? $conv->created_at->toIso8601String() : now()->toIso8601String()
                ];
                
                // Add agent message with visualization if available
                $agentMessage = [
                    'role' => 'agent',
                    'content' => $conv->response,
                    'timestamp' => $conv->created_at ? $conv->created_at->toIso8601String() : now()->toIso8601String(),
                    'agent_type' => $conv->agent_type ?? 'unknown'
                ];
                
                // Add visualization data if present
                if ($conv->visualization) {
                    $agentMessage['visualization'] = $conv->visualization;
                    \Log::info("Including visualization data from database for conversation {$conversationId}", [
                        'chart_type' => $conv->visualization['chart_type'] ?? 'unknown'
                    ]);
                }
                
                $messages[] = $agentMessage;
            }
            
            // Log the data being returned for debugging
            \Log::info("Returning conversation history", [
                'conversation_id' => $conversationId,
                'message_count' => count($messages),
                'has_visualization' => collect($messages)->contains(function ($message) {
                    return isset($message['visualization']);
                })
            ]);

            return response()->json(['messages' => $messages]);
        } catch (\Exception $e) {
            \Log::error('Error retrieving conversation history from database', [
                'error' => $e->getMessage(),
                'conversation_id' => $conversationId,
                'company_id' => $companyId,
            ]);
            
            return response()->json([
                'error' => 'Failed to retrieve conversation history from database',
                'messages' => []
            ], 500);
        }
    }
} 