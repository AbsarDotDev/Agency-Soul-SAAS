<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class AgentConversation extends Model
{
    use HasFactory;

    protected $fillable = [
        'conversation_id',
        'company_user_id',
        'user_id',
        'title',
        'message',
        'response',
        'agent_type',
        'tokens_used',
        'visualization',
    ];

    /**
     * The attributes that should be cast.
     *
     * @var array
     */
    protected $casts = [
        'visualization' => 'json',
        'created_at' => 'datetime',
        'updated_at' => 'datetime',
    ];

    /**
     * Get the company user that owns the conversation.
     */
    public function companyUser()
    {
        return $this->belongsTo(User::class, 'company_user_id');
    }

    /**
     * Get the user that owns the conversation.
     */
    public function user()
    {
        return $this->belongsTo(User::class, 'user_id');
    }
} 