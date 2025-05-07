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

        $agentApiUrl = config('app.ai_agent_url'); // Get URL from config (pulled from .env)

        if (!$agentApiUrl) {
            Log::error('AI Agent API URL is not configured.');
            return response()->json(['error' => 'AI Agent service is not configured correctly.'], 500);
        }

        try {
            $response = Http::timeout(60) // Set a timeout (e.g., 60 seconds)
                ->post($agentApiUrl . '/api/message', [
                    'message' => $validated['message'],
                    'company_id' => $company->id,
                    'user_id' => (string) $user->id, // Pass the specific user who sent the message
                    'conversation_id' => $validated['conversation_id'] ?? null,
                    'agent_type' => 'sql' // Defaulting to SQL agent for now
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
            
            // Attempt to parse the visualization data specifically for debugging
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
            
            // Log Parsed Response Data for comparison
            Log::info('Parsed AI Agent Response Data:', ['parsed_data' => $responseData]);
            
            // Correctly extract the agent's answer from the 'response' field
            $aiResponseText = $responseData['response'] ?? 'Error: No response text received from agent.'; 
            $tokensUsedInRequest = $responseData['tokens_used'] ?? 0; 
            $conversationId = $responseData['conversation_id'] ?? $validated['conversation_id'];
            $conversationTitle = $responseData['conversation_title'] ?? null; // Get the title

            if ($tokensUsedInRequest == 0) {
                Log::warning('AI Agent response did not include "tokens_used" field.', ['company_id' => $company->id, 'response_body' => $responseData]);
            }
            
            // Update token usage in the database
            $company->ai_agent_tokens_used += $tokensUsedInRequest; // Use the value from response, default to 0
            $company->save();

            // Recalculate remaining tokens
            $tokens_remaining = max(0, $tokens_allocated - $company->ai_agent_tokens_used);

            // Return the successful response from the agent
            return response()->json([
                'response' => $aiResponseText, // Use the correct variable holding the agent's answer
                'conversation_id' => $conversationId,
                'conversation_title' => $conversationTitle,
                'tokens_remaining' => $tokens_remaining,
                'tokens_used' => $tokensUsedInRequest, // Also return tokens used for this request if needed
                'visualization' => $responseData['visualization'] ?? null // Include visualization data
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

        $agentApiUrl = config('app.ai_agent_url');
        if (!$agentApiUrl) {
            Log::error('AI Agent API URL is not configured for history retrieval.');
            return response()->json(['error' => 'AI Agent service is not configured correctly.'], 500);
        }

        try {
            // Try fetching from agent service first
            $response = Http::timeout(30) // Shorter timeout for history retrieval
                ->get($agentApiUrl . '/api/conversation/' . $id, [
                    'company_id' => $company->id // Pass company ID for validation/scoping in the agent service
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