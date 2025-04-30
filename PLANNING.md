# AgenySoul AI Agent - Planning Document

## 1. High-Level Vision & Goals
- Integrate an AI-powered chatbot ("AI Agent") into the existing AgenySoul Laravel application.
- Allow authenticated company users to interact with *their own* company data (ERP, CRM, HRM, accounting) via a conversational interface.
- Provide insights, answer questions (e.g., HR overview, specific salaries, sales predictions) through natural language.
- Ensure strict data privacy and isolation between different companies using the platform.
- Visualize data appropriately (charts, graphs) when requested.
- **Allow super admins to enable/disable the AI Agent feature within specific subscription plans.**
- **Control AI Agent usage by assigning consumable "agent tokens" to companies based on their plan, limiting the number of interactions.**

## 2. Architecture Overview
- **Frontend:** New web interface page within the existing AgenySoul Laravel application. Accessible only to users with the 'company' role. Built using standard Laravel views (Blade) and potentially a modern JS framework. This page can be opened by clicking a floating chat button icon.
- **Backend (Controller/Route Layer):** Laravel routes and controllers to handle requests from the agent frontend, manage authentication/authorization, and interact with the AI Agent Core.
- **AI Agent Core:**
    - **Middleware/Pre-check:** Before hitting the dispatcher, incoming requests must pass through checks verifying:
        - User authentication/authorization.
        - Company's subscription plan allows AI Agent access.
        - Company has sufficient agent tokens remaining.
    - **Dispatcher:** A central Laravel service/component that receives validated user queries, identifies the user's company, potentially determines the relevant specialized agent (HRM, Sales, etc.), and routes the request.
    - **Multi-Agent System:** Specialized services/modules responsible for specific domains (HRM, Sales, CRM, Accounting, etc.).
    - **Data Access Layer:** Uses LangChain SQL Toolkit to dynamically generate and execute SQL queries based on user questions, rather than predefined static queries.
    - **LLM/NLP Service:** Integration with Gemini 2.5 Pro
    - **Visualization Module:** Generates chart data based on query results.
    - **Token Management Service:** Responsible for decrementing tokens upon successful agent interaction completion (likely triggered by the Dispatcher or specific Agents/Modules).
- **Database:** Existing MySQL database. Requires schema updates for:
    - Plan definitions (adding AI Agent enabled flag, default tokens).
    - Token tracking and usage history.
    - Conversation history tables (including company identifiers).
- **Super Admin Interface:** Update the existing plans page in super admin panel for updating plan settings (AI Agent toggle, token allocation).

## 3. Tech Stack
- **Backend Framework:** Laravel (Existing)
- **Database:** MySQL (Existing), currently using through phpMyAdmin but also have deployed this project on hostinger so we can use cloud database as well or make it flexible to work on both local and cloud after deployment.
- **Frontend:** Laravel Blade, analyze existing views for more details on building frontend
- **AI Agent:** (Python Microservice) with FastAPI, LangChain, LangGraph
- **LLM Integration:** Gemini 2.5 Pro
- **Database Access:** LangChain SQL Toolkit for dynamic SQL generation and execution
- **Charting Library:** Backend or Frontend (TBD).

## 4. Key Constraints & Considerations
- **Data Privacy:** Paramount. Company-specific data isolation must be enforced at all times.
- **Minimal Database Schema Changes:** Any database modifications must be carefully analyzed to ensure they don't disrupt existing functionality. Use existing tables and relationships wherever possible.
- **User Role System:** Use the existing user role system rather than creating new company-specific tables. The current system uses `type` column in the users table for role designation.
- **Dynamic SQL Queries:** Implement LangChain SQL toolkit to enable flexible, ad-hoc questions about company data rather than predefined query patterns.
- **Scalability:** Consider load, especially with token checks on potentially frequent interactions.
- **Security:** Protect against prompt injection, unauthorized access/modification, ensure admin controls are secure.
- **Accuracy:** Ensure data retrieval/analysis accuracy.
- **Maintainability:** Modular design.
- **User Experience:** Intuitive interface, clear feedback on token usage/limits.
- **Billing/Plan Integration:** Requires tight integration with existing subscription plan logic. How are tokens replenished (e.g., monthly reset, purchase)?
- **Usage Tracking:** Need for reliable, auditable tracking of agent calls and token consumption.
- **Rate Limiting:** The token system serves as a usage cap based on plan value.

## 5. Database Integration Strategy 
- **Use Existing Schema:** Rather than creating new tables like 'companies', leverage the existing users table with 'type' column for role-based identification.
- **Company Data Isolation:** Every SQL query generated by the agent must include appropriate scoping based on the authenticated user's company ID (created_by field for company-type users).
- **Minimal Schema Updates:** Only add necessary tables/columns for:
    - Token tracking and usage history 
    - Conversation storage
    - AI agent feature flags in plans
- **Schema Migration Approach:**
    - Always analyze existing schema first before creating migrations
    - Test migrations on a separate environment before applying to production
    - Ensure backward compatibility
    - Create migration rollback plans

## 6. SQL Query Strategy
- **Dynamic SQL Generation:** Use LangChain SQL toolkit to create a flexible system that can answer any question about company data, rather than predefined queries.
- **Database Connection:** Set up a secure MySQL connection between the Python microservice and the database.
- **Schema Understanding:** Load database schema details into the LLM context for accurate query generation.
- **Company Data Scoping:** Automatically apply company-specific filters to all generated SQL.
- **Query Validation:** Implement safety checks on generated SQL queries before execution.
- **Query Execution:** Execute validated queries and transform results into natural language responses or visualizations.

## 7. Implementation Phases
1. **Foundation:** Set up Python microservice, database connections, and basic agent architecture
2. **SQL Toolkit Integration:** Implement LangChain SQL toolkit for dynamic query generation
3. **Conversation & Token Management:** Create conversation storage and token tracking
4. **Visualization:** Add data visualization capabilities
5. **Frontend Integration:** Create chat interface in Laravel application
6. **Admin Controls:** Implement plan management for AI agent features
7. **Testing & Security:** Comprehensive testing with focus on data isolation