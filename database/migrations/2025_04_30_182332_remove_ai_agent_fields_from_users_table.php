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
            if (Schema::hasColumn('users', 'ai_agent_enabled')) {
                $table->dropColumn('ai_agent_enabled');
            }
            if (Schema::hasColumn('users', 'ai_agent_tokens_allocated')) {
                $table->dropColumn('ai_agent_tokens_allocated');
            }
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->boolean('ai_agent_enabled')->default(0)->after('active_status'); // Adjust 'after' if needed
            $table->integer('ai_agent_tokens_allocated')->default(0)->after('ai_agent_enabled');
        });
    }
};
