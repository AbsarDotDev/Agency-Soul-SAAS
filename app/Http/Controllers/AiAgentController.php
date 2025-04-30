<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use App\Models\User;

class AIAgentController extends Controller
{
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

        // Check if AI Agent is enabled for the company
        if (!$company->ai_agent_enabled) {
            return response()->json(['error' => 'AI Agent is not enabled for your subscription.'], 403);
        }

        // Check if the company has tokens remaining
        if ($company->ai_agent_tokens_allocated <= $company->ai_agent_tokens_used) {
            return response()->json(['error' => 'You have used all your AI Agent tokens.'], 403);
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
                return response()->json(['error' => $errorDetail], $response->status() >= 500 ? 503 : $response->status());
            }

            // Note: Token decrement happens within the Python service upon successful processing now.
            // We rely on the Python service's checks.

            return $response->json();

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
} 