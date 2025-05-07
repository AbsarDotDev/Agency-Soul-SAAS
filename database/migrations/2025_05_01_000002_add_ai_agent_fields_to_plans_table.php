<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            // Add AI agent feature flags and token allocation to plans
            $table->boolean('ai_agent_enabled')->default(false)->after('trial_days');
            $table->integer('ai_agent_default_tokens')->default(0)->after('ai_agent_enabled')
                ->comment('Default number of AI agent tokens allocated to companies on this plan');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn('ai_agent_enabled');
            $table->dropColumn('ai_agent_default_tokens');
        });
    }
}; 