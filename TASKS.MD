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

## Tasks

### Core Infrastructure

- [x] Set up initial FastAPI structure
- [x] Add database connection handling
- [x] Implement token validation and authentication
- [x] Create token usage tracking mechanisms
- [x] Implement conversation history storage
- [x] Implement LLM conversation title generation
- [x] Add LLM safety settings and fallbacks
- [x] Setup proper error handling across application
- [x] Optimize LLM initialization to reduce latency
- [x] Implement LangGraph-based multi-agent architecture
- [x] Enable LLM instance caching/reuse across requests

### Agent Development

- [x] Create BaseAgent abstract class
- [x] Implement SQL Agent with company data isolation
- [ ] Complete HRM Agent development
  - [ ] Create dedicated schemas for employee-related tables
  - [ ] Implement employee information retrieval
  - [ ] Add leave management capabilities
  - [ ] Add payroll query handling
  - [ ] Integrate SQL toolkit within HRM agent
- [ ] Complete Finance Agent development
  - [ ] Create dedicated schemas for financial tables
  - [ ] Implement transaction analysis
  - [ ] Add budget/expense reporting
  - [ ] Integrate SQL toolkit within Finance agent
- [ ] Complete CRM Agent development
  - [ ] Create dedicated schemas for customer-related tables
  - [ ] Implement customer data retrieval
  - [ ] Add support ticket capabilities
  - [ ] Integrate SQL toolkit within CRM agent
- [ ] Complete Sales Agent development
  - [ ] Create dedicated schemas for sales-related tables
  - [ ] Implement leads and deals analysis
  - [ ] Add pipeline visualization capabilities
  - [ ] Integrate SQL toolkit within Sales agent

### Visualization

- [x] Implement basic Chart.js visualization generation
- [ ] Add customization options for visualization types
- [ ] Create summarization for visualization data
- [ ] Enable exporting visualizations

### Action Capabilities

- [ ] Implement HRM actions (add/update employees, departments)
- [ ] Implement Finance actions (record transactions, update budgets)
- [ ] Implement CRM actions (add/update customer records, tickets)
- [ ] Implement Sales actions (add/update leads, deals)

### Testing & Deployment

- [x] Set up CI pipeline basics
- [x] Add FastAPI health check endpoints
- [ ] Implement comprehensive unit tests
- [ ] Add integration tests for agent interactions
- [ ] Document API endpoints for frontend integration
- [ ] Optimize Docker container for production
- [ ] Set up monitoring for token usage and performance

### Documentation

- [x] Create basic README with setup instructions
- [x] Document architecture in PLANNING.md
- [ ] Add API documentation for all endpoints
- [ ] Create user guide for agent capabilities
- [ ] Document database schema relationships
- [ ] Add developer guides for extending agents

## Completed Tasks (Latest First)

- **2023-07-15**: Fixed router node to use consistent state-based conditional routing
- **2023-07-15**: Fixed LangGraph node return values and conditional edge routing
- **2023-07-15**: Fixed async/sync function mismatch in LangGraph dispatcher
- **2023-07-15**: Fixed LangGraph import issues with conda environment and improved deployment process
- **2023-07-14**: Implemented LangGraph-based multi-agent architecture with specialized agents and fallback mechanisms
- **2023-07-12**: Fixed LLM initialization issues and enhanced error handling
- **2023-07-10**: Added conversation title generation and history retrieval
- **2023-07-08**: Implemented SQLAgent with company data isolation
- **2023-07-05**: Set up token tracking and validation
- **2023-07-01**: Created initial FastAPI application structure
## Discovered During Work
- [ ] Debug Pusher integration for existing chat feature (Path not found, cluster issues, 500 error on send) - 2024-07-26
