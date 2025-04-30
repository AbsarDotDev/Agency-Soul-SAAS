# AgenySoul AI Agent - Planning Document

## 1. High-Level Vision & Goals
- Integrate an AI-powered chatbot ("AI Agent") into the existing AgenySoul Laravel application.
- Allow authenticated company users to interact with *their own* company data (ERP, CRM, HRM, accounting) via a conversational interface.
- Provide insights, answer questions (e.g., HR overview, specific salaries, sales predictions), and eventually perform actions (e.g., add employee).
- Ensure strict data privacy and isolation between different companies using the platform.
- Visualize data appropriately (charts, graphs) when requested.
- **Allow super admins to enable/disable the AI Agent feature within specific subscription plans.**
- **Control AI Agent usage by assigning consumable "agent tokens" to companies based on their plan, limiting the number of interactions.**

## 2. Architecture Overview
- **Frontend:** New web interface page within the existing AgenySoul Laravel application. Accessible only to users with the 'company' role. Built using standard Laravel views (Blade) and potentially a modern JS framework. This page can be opened b clicking floating chat button icon
- **Backend (Controller/Route Layer):** Laravel routes and controllers to handle requests from the agent frontend, manage authentication/authorization, and interact with the AI Agent Core.
- **AI Agent Core:**
    - **Middleware/Pre-check:** Before hitting the dispatcher, incoming requests must pass through checks verifying:
        - User authentication/authorization.
        - Company's subscription plan allows AI Agent access.
        - Company has sufficient agent tokens remaining.
    - **Dispatcher:** A central Laravel service/component that receives validated user queries, identifies the user's company, potentially determines the relevant specialized agent (HRM, Sales, etc.), and routes the request.
    - **Multi-Agent System:** Specialized services/modules responsible for specific domains (HRM, Sales, CRM, Accounting, etc.).
    - **Data Access Layer:** Interacts with the MySQL database *strictly* using Eloquent models with global scopes or other robust mechanisms to enforce company data isolation based on the authenticated user.
    - **LLM/NLP Service (Optional but likely):** Integration with Gemini 2.5 Pro
    - **Task Execution Module:** Handles requests to perform actions. Needs validation, authorization, and *must trigger token consumption*.
    - **Visualization Module:** Generates chart data.
    - **Token Management Service:** Responsible for decrementing tokens upon successful agent interaction completion (likely triggered by the Dispatcher or specific Agents/Modules).
- **Database:** Existing MySQL database. Requires schema updates for:
    - Plan definitions (adding AI Agent enabled flag, default tokens).
    - Company records or a dedicated table to track allocated/remaining agent tokens.
    - Conversation history tables (including company identifiers).
- **Super Admin Interface:** Update the existing plans page in super admin panel for updating plan settings (AI Agent toggle, token allocation).

## 3. Tech Stack
- **Backend Framework:** Laravel (Existing)
- **Database:** MySQL (Existing)
- **Frontend:** Laravel Blade, analyse existing views for more details on how to build frontend
- **AI/AI agnet:** (Python Microservice) 
- **Charting Library:** Backend or Frontend (TBD).

## 4. Key Constraints & Considerations
- **Data Privacy:** Paramount.
- **Scalability:** Consider load, especially with token checks on potentially frequent interactions.
- **Security:** Protect against prompt injection, unauthorized access/modification, ensure admin controls are secure.
- **Accuracy:** Ensure data retrieval/analysis accuracy.
- **Maintainability:** Modular design.
- **User Experience:** Intuitive interface, clear feedback on token usage/limits.
- **Billing/Plan Integration:** Requires tight integration with existing subscription plan logic. How are tokens replenished (e.g., monthly reset, purchase)?
- **Usage Tracking:** Need for reliable, auditable tracking of agent calls and token consumption.
- **Rate Limiting:** The token system serves as a usage cap based on plan value.