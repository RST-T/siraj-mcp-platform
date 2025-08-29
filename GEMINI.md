# GEMINI.md

## Project Overview

This project, SIRAJ v6.1, is a sophisticated Computational Hermeneutics MCP Server. It's designed for advanced linguistic analysis, with a special focus on Arabic and Semitic languages, while also incorporating cultural sovereignty protection. The project is a hybrid, utilizing a Node.js wrapper to manage a powerful Python-based engine.

The core of the project is its revolutionary "methodology-first" architecture. Instead of providing pre-analyzed data, SIRAJ teaches AI how to perform scholarly analysis through sophisticated methodological frameworks. This approach is a paradigm shift from data-centric to methodology-centric computational linguistics.

The project has a strong focus on Islamic Tafsir methodology, comparative linguistics, and modern NLP. It also has a strong focus on cultural sovereignty and community validation. The 72-node archetypal table is a descriptive computational linguistics framework for semantic mapping. The Adaptive Semantic Architecture (ASA) is a dynamic framework that evolves based on convergent validation from traditional scholarship, computational analysis, and community feedback.

## Building and Running

### Prerequisites

*   Node.js (v18+)
*   Python (v3.10+)

### Installation

1.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

2.  **Install Python dependencies:**
    ```bash
    npm run install-deps
    ```
    or
    ```bash
    python -m pip install -r requirements.txt
    ```

### Running the Server

The server can be started using the following command:

```bash
npm start
```

This will start the server with the default `stdio` transport. To use the `sse` transport, you can use the following command:

```bash
npm run dev -- --transport sse --port 3001
```

### Running Tests

The project has tests for both the Node.js wrapper and the Python engine.

*   **Node.js tests:**
    ```bash
    npm test
    ```

*   **Python tests:**
    ```bash
    pytest
    ```

## Development Conventions

*   The Node.js part of the project is written in TypeScript and uses Prettier for formatting and ESLint for linting.
*   The Python part of the project follows the PEP 8 style guide and uses Black for formatting.
*   The project uses a hybrid approach, with a Node.js wrapper and a Python engine. The two parts communicate using the Model Context Protocol (MCP).
*   The Python engine provides a set of tools that return a methodology for how to perform the analysis, rather than performing the analysis directly.
*   The project has a strong focus on cultural sovereignty and provides tools for analyzing culturally sensitive content in a respectful manner.