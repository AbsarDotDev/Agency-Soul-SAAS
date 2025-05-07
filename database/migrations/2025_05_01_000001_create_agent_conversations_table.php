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
        Schema::create('agent_conversations', function (Blueprint $table) {
            $table->id();
            $table->uuid('conversation_id')->index();
            $table->unsignedBigInteger('company_user_id')->comment('User ID of the company (user with type=company)');
            $table->unsignedBigInteger('user_id')->comment('User ID who sent the message');
            $table->text('message');
            $table->text('response');
            $table->string('agent_type')->nullable()->comment('Type of agent that handled the request (HRM, Sales, etc.)');
            $table->integer('tokens_used')->default(1);
            $table->timestamps();

            // Foreign keys - both reference the users table but with different purposes
            $table->foreign('company_user_id')->references('id')->on('users')->onDelete('cascade');
            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade');
            
            // Indexes for faster retrieval
            $table->index('company_user_id');
            $table->index('user_id');
            $table->index('created_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('agent_conversations');
    }
}; 