<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use App\Models\User;

class EnsureAIAgentEnabled
{
    /**
     * Handle an incoming request.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  \Closure(\Illuminate\Http\Request): (\Illuminate\Http\Response|\Illuminate\Http\RedirectResponse)  $next
     * @return \Illuminate\Http\Response|\Illuminate\Http\RedirectResponse
     */
    public function handle(Request $request, Closure $next)
    {
        $user = Auth::user();
        $company = null;
        $plan = null;

        if (!$user) {
            return redirect('login');
        }

        // Find the company user record
        if ($user->type == 'company') {
            $company = $user;
        } elseif ($user->created_by) {
            $company = User::find($user->created_by);
        }

        // Get the plan
        if ($company && $company->type == 'company') {
            $plan = \App\Models\Plan::find($company->plan);
        }

        // Check if company context is valid and AI agent is enabled in the plan
        if (!$company || $company->type !== 'company' || !$plan || $plan->ai_agent_enabled != 1) {
            // Redirect with error message
            return redirect()->route('dashboard')->with('error', __('AI Agent feature is not enabled for your subscription plan.'));
        }

        return $next($request);
    }
} 