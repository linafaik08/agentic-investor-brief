# AgentOps: Operational Frameworks for LLM-Powered Agent Systems
**A Practical Guide to Managing Production Agents with MLflow**

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: October 2025  
- Last update: October 2025

## Objective

This repository demonstrates **production-grade observability for AI agent systems** using MLflow. It bridges the gap between AgentOps theory and practice through a real-world multi-agent investment brief generator that autonomously fetches market data, analyzes financials, monitors news, and synthesizes investment recommendations.

For a deeper dive into the concepts and implementation details, check out the full article on [The AI Practitioner](https://aipractitioner.substack.com/).

## Project Description

Traditional MLOps tools fall short when monitoring AI agents—systems that reason iteratively, use tools dynamically, and fail silently with confident hallucinations. This project implements the **AgentOps lifecycle** (monitoring → anomaly detection → root cause analysis → resolution) using MLflow's tracing, prompt management, and evaluation capabilities.

### System Architecture

The investment brief generator consists of four specialized agents:

1. **Data Fetch Agent**: Retrieves stock prices and financial KPIs from external APIs
2. **Financial Analysis Agent**: Computes valuation metrics and financial health scores
3. **News Monitoring Agent**: Summarizes relevant market news and sentiment
4. **Investment Reasoning Agent**: Synthesizes inputs into coherent recommendations

### Code Structure

```
notebooks/
├── demo_agent.ipynb           # Complete agent implementation walkthrough
├── demo_qa.ipynb              # Quality assurance and evaluation examples
├── demo_tools.ipynb           # Tool integration demonstrations

analysis/
├── v1.py                      # Initial agent architecture
├── v2.py                      # Improved version with enhanced observability

brief/
├── v1.py                      # Investment brief generation logic

src/agent_investor_brief/
├── agents/
│   ├── investor_agent.py      # Main orchestration agent
│   └── qa_agent.py            # Quality assurance agent
├── tools/
│   ├── financial_data.py      # Financial API integrations
│   └── industry_research.py   # News and research tools
├── config.py                  # Configuration management
├── prompt_manager.py          # MLflow prompt versioning
└── utils.py                   # Utilities and helpers

mlflow.db                      # MLflow tracking database
pyproject.toml                 # Project dependencies
README.md
```

## How to Use This Repository?

### Requirements

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

Main libraries:
```
# MLflow and AI framework
mlflow>=3.0.0
openai>=1.109

# Financial data and web scraping
yfinance>=0.2.0
requests>=2.31.0
beautifulsoup4>=4.12.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Configuration and utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
tavily-python>=0.7.12
```

### Installation

1. **Install uv** (if not already installed):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clone the repository**:
```bash
git clone https://github.com/linafaik08/agentic-investor-brief.git
cd agentic-investor-brief
```

3. **Install dependencies with uv**:
```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

Alternatively, install in development mode:
```bash
uv pip install -e .
```

### Setup

1. **Create a `.env` file** based on `.example.env`:
```bash
cp .example.env .env
```

2. **Add your API keys** to `.env`:
```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here  # Optional: for enhanced web search
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

3. **Initialize MLflow**:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Access the MLflow UI at http://localhost:5000 or http://127.0.0.1:5000/

### Running the Project

1. **Start with the demos**: Open `notebooks/demo_agent.ipynb` to see the complete agent system in action with MLflow tracing enabled.

2. **Explore observability**: 
   - View execution traces in the MLflow UI (http://localhost:5000 or http://127.0.0.1:5000/)
   - Compare prompt versions in the Prompt Registry
   - Analyze evaluation metrics across runs

3. **Understand the evolution**: Compare `analysis/v1.py` and `analysis/v2.py` to see how observability improvements enable iterative refinement.

4. **Run evaluations**: Use `notebooks/demo_qa.ipynb` to see systematic quality validation with correctness checking and custom safety criteria.

### Key Features Demonstrated

- **Hierarchical Span Tracing**: Captures complete cognitive flows from high-level tasks to individual LLM calls
- **Prompt Versioning**: Treats prompts as production artifacts with A/B testing and rollback capabilities  
- **Multi-dimensional Evaluation**: Validates quality beyond binary correctness using LLM-as-judge and custom scorers
- **Cost & Performance Monitoring**: Tracks token usage, latency, and success rates across agent executions