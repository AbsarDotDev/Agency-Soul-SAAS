# AgenySoul AI Agent - Planning Document

## 1. High-Level Vision & Goals
- Integrate an AI-powered chatbot ("AI Agent") into the existing AgenySoul Laravel application.
- Allow authenticated company users to interact with *their own* company data (ERP, CRM, HRM, accounting) via a conversational interface.
- Provide insights, answer questions (e.g., HR overview, specific salaries, sales predictions) through natural language.
- Ensure strict data privacy and isolation between different companies using the platform.
- Visualize data appropriately (charts, graphs) when requested.
- **Allow super admins to enable/disable the AI Agent feature and set default token allocation within specific subscription plans.**
- **Control AI Agent usage by assigning consumable "agent tokens" to companies based on their plan, limiting the number of interactions.**
- **Provide a dedicated chat interface with conversation history.**

## 2. Architecture Overview
- **Frontend:**
    - **Chat Page:** A dedicated page (`/ai-agent/chat`) within the AgenySoul Laravel application. Accessible only to users whose company plan has the AI Agent enabled. Built using Blade and JS (likely Alpine.js or Vue/React snippet for interactivity).
        - Layout: Sidebar listing past conversations (with titles), main chat area on the right.
        - Functionality: Input field, message display, AJAX calls to Laravel backend, handling of conversation state (ID), error display.
        - Token Limit Handling: Input disabled if tokens are zero, but history remains viewable.
    - **Header Display:** Show remaining AI tokens in the main application header for authorized users.
    - **Floating Button:** A floating chat icon (bottom-right) visible only to authorized users, linking to the Chat Page.
- **Backend (Laravel Controller/Route Layer):**
    - `AIAgentController`: Handles requests from the agent frontend (`/ai-agent/chat` endpoint), manages authentication/authorization, performs plan/token checks, and interacts with the AI Agent Core via HTTP requests.
    - Middleware: Protects the `/ai-agent/chat` page route, ensuring only users from companies with the feature enabled can access it.
- **AI Agent Core (Python Microservice):**
    - **API Endpoints (`/api/message`, `/api/visualization`):** Receive requests from the Laravel backend including `company_id`, `user_id`, `message`, and `conversation_id`.
    - **Data Access Layer:** Uses LangChain SQL Toolkit (`SQLAgent`) to dynamically generate and execute SQL queries based on user questions, automatically scoped to the `company_id`.
    - **LLM/NLP Service:** Integration with configured LLM (Gemini, OpenAI).
    - **Visualization Module:** Generates chart data based on query results (called by `SQLAgent`).
    - **Conversation Management:** Uses the `conversation_id` passed from Laravel to maintain context within a chat session. The `SQLAgent` loads previous messages for context.
    - **Token Consumption:** The Python service handles the primary logic but relies on Laravel for the *initial* check of token availability. Laravel controller performs checks before calling the API. Token usage is recorded in the `agent_conversations` table.
- **Database (MySQL):**
    - `plans` table: Added `ai_agent_enabled` (boolean) and `ai_agent_default_tokens` (integer) columns.
    - `users` table: Added `ai_agent_enabled` (boolean), `ai_agent_tokens_allocated` (integer), `ai_agent_tokens_used` (integer) columns (for `type='company'` users).
    - `agent_conversations` table: Stores chat history, linked to `company_user_id` and `user_id`, includes `conversation_id`, `message`, `response`, `agent_type`, `tokens_used`, and a `title` (varchar) for the sidebar.
- **Super Admin Interface:** Update the existing plans page (`/plans/{id}/edit`) in the super admin panel to include form fields for `ai_agent_enabled` and `ai_agent_default_tokens`.

## 3. Tech Stack
- **Backend Framework:** Laravel (Existing)
- **Database:** MySQL (Existing)
- **Frontend:** Laravel Blade, JS (potentially Alpine.js/jQuery for AJAX and dynamic UI elements in the chat interface).
- **AI Agent:** Python Microservice with FastAPI, LangChain.
- **LLM Integration:** Gemini / OpenAI (via LangChain wrappers).
- **Database Access (AI):** LangChain SQL Toolkit (`SQLDatabase`, `create_sql_agent`).
- **Charting Library:** TBD (likely Chart.js or similar, integrated into the Blade view).

## 4. Key Constraints & Considerations
- **Data Privacy:** Paramount. Company-specific data isolation enforced in Laravel checks and Python agent's database interaction.
- **Plan/Token Enforcement:** Access control and token limits strictly enforced in Laravel *before* calling the Python API.
- **User Role System:** Use the existing `users` table (`type` column and `created_by`) for company identification and access control.
- **Conversation Memory:** Must handle multi-turn conversations effectively using `conversation_id` and context injection in prompts.
- **UI/UX:** Chat interface should be intuitive, provide clear feedback (token usage, errors), and manage conversation history effectively.
- **Scalability:** Consider potential load on both Laravel and Python services.
- **Security:** Protect against prompt injection, unauthorized access. Ensure admin controls are secure.
- **Maintainability:** Modular design in both Laravel and Python components.

## 5. Database Integration Strategy
- **Leverage Existing Schema:** Use `users` table for company/user info.
- **Company Data Isolation:** Enforced via `company_id` checks in Laravel controller and automated query scoping in Python `CompanyIsolationSQLDatabase`.
- **New Schema:**
    - `plans`: Add `ai_agent_enabled`, `ai_agent_default_tokens`.
    - `users`: Add `ai_agent_enabled`, `ai_agent_tokens_allocated`, `ai_agent_tokens_used`.
    - `agent_conversations`: Add `title`.
- **Plan Activation Logic:** Update payment/plan controllers to populate `users` table fields based on the chosen plan.

## 6. SQL Query Strategy (AI Agent)
- **Dynamic SQL Generation:** LangChain SQL toolkit (`SQLAgent`) generates queries.
- **Database Connection (AI):** Secure MySQL connection from Python service.
- **Schema Understanding (AI):** Agent prompt includes relevant table info.
- **Company Data Scoping (AI):** `CompanyIsolationSQLDatabase` automatically adds `WHERE` clauses based on `company_id`.
- **Query Validation (AI):** Basic safety checks in LangChain agent.

## 7. Implementation Phases (Revised)
1.  **Foundation & SQL Toolkit:** (Largely Complete) Python service setup, LangChain SQL agent, basic data isolation.
2.  **Plan & Access Control:**
    *   Update DB migrations (`plans`, `users`).
    *   Update Super Admin UI & Controller for plan settings.
    *   Update plan activation logic.
3.  **Core Frontend Integration:**
    *   Create `AIAgentController` & routes (Done).
    *   Implement Header Token Display.
    *   Implement Floating Chat Button.
    *   Create basic `chat.blade.php` view.
    *   Add Route Middleware for chat page access.
4.  **Chat Interface & Functionality:**
    *   Implement AJAX communication in `chat.blade.php`.
    *   Implement message display logic.
    *   Implement token limit UI handling.
5.  **Conversation History & Memory:**
    *   Add `title` column to `agent_conversations` migration.
    *   Implement title generation logic.
    *   Implement sidebar history display in `chat.blade.php`.
    *   Verify/refine conversation context handling in `SQLAgent`.
6.  **Visualization Integration (Deferred/Optional):** Connect visualization generation if needed.
7.  **Testing & Refinement:** Thorough testing of access control, token logic, chat functionality, data isolation.