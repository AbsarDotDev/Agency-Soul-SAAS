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
        Schema::table('users', function (Blueprint $table) {
            // Add AI agent token allocation and usage tracking for company users
            $table->integer('ai_agent_tokens_allocated')->default(0)->after('plan_expire_date');
            $table->integer('ai_agent_tokens_used')->default(0)->after('ai_agent_tokens_allocated');
            $table->boolean('ai_agent_enabled')->default(false)->after('ai_agent_tokens_used');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->dropColumn('ai_agent_tokens_allocated');
            $table->dropColumn('ai_agent_tokens_used');
            $table->dropColumn('ai_agent_enabled');
        });
    }
};
