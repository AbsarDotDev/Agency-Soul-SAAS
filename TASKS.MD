# AgenySoul AI Agent Implementation Tasks

## Phase 1: Foundation & Core Components (MVP)
- [x] Create planning document (PLANNING.md)
- [x] Set up Python microservice structure
  - [x] Create basic FastAPI application
  - [x] Set up project directory structure
  - [x] Configure LLM integration (Gemini/OpenAI)
  - [x] Set up LangChain dependencies
- [x] Implement database connection (Python)
- [x] Implement token tracking system schema (DB Migrations)
  - [x] Add token/AI fields to `users` table
  - [x] Create `agent_conversations` table
  - [x] Add AI fields to `plans` table
- [x] Develop basic agent architecture (Python)
  - [x] Create `BaseAgent`
  - [x] Implement `SQLAgent` with LangChain SQL Toolkit
  - [x] Implement dynamic SQL query generation
  - [x] Implement robust company data isolation in SQL queries
- [x] Implement conversation context management (Basic history injection)
- [x] Implement token checks and usage updates (Python Agent & Laravel Controller)
- [x] Create Laravel Controller (`AIAgentController`) and basic route (`/ai-agent/chat`)

## Phase 2: Plan Integration & Access Control
- [ ] Update Super Admin Plan Edit UI (`plan.edit`) to include AI Agent toggle and token allocation fields.
- [ ] Update `PlanController@update` to save new AI Agent plan fields.
- [ ] Update plan activation logic (e.g., in payment controllers like `StripePaymentController`, `PaypalController`, etc.) to set `ai_agent_enabled` and `ai_agent_tokens_allocated` on the company `User` record based on the subscribed plan.
- [ ] Add route middleware to protect `/ai-agent/chat` page, checking `ai_agent_enabled`.
- [ ] Test plan activation/deactivation and token allocation flow.

## Phase 3: Frontend Integration (Chat Interface)
- [ ] Create `ai_agent/chat.blade.php` view.
- [ ] Implement basic layout: sidebar for history, main chat area.
- [ ] Implement AJAX call from chat view to `/ai-agent/chat` endpoint in `AIAgentController`.
- [ ] Implement logic to display user messages and agent responses dynamically.
- [ ] Implement UI handling for token limits (disable input if tokens <= 0).
- [ ] Add floating chat button to main layout (visible only if `ai_agent_enabled`).
- [ ] Add remaining token display to main header (visible only if `ai_agent_enabled`).

## Phase 4: Conversation History & Memory
- [ ] Add `title` column (VARCHAR) to `agent_conversations` table migration.
- [ ] Implement logic to generate conversation titles (e.g., summarize first message using LLM via Python service or simple truncation in Laravel).
- [ ] Implement sidebar in `chat.blade.php` to fetch and display conversation history titles, allowing users to load past chats.
- [ ] Refine/verify conversation context handling in `SQLAgent` (ensure history is used effectively by the LLM).

## Phase 5: Visualization & Refinement
- [ ] Implement chart rendering in `chat.blade.php` (if visualization data is present in the response).
- [ ] Add user-specific role checks for AI agent access (Future enhancement).
- [ ] Comprehensive testing: Data isolation, token logic, UI interactions, error handling.
- [ ] Optimize performance and LLM prompts.

## Current Focus (Next Tasks)
- [ ] Update Super Admin Plan Edit UI (`plan.edit`) for AI settings.
- [ ] Update `PlanController@update` to save AI settings.
- [ ] Update plan activation logic in payment controllers.
- [ ] Add route middleware for AI chat page access.
- [ ] Create basic `ai_agent/chat.blade.php` view.

## Discovered During Work
- [ ] Debug Pusher integration for existing chat feature (Path not found, cluster issues, 500 error on send) - 2024-07-26
