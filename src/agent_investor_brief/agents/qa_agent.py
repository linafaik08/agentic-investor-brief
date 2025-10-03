"""
Company Q&A Agent with MLflow Tracing and Tool Calling
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from openai import OpenAI
import mlflow
from mlflow.entities import SpanType
from ..config import settings, MLFLOW_CONFIG
from ..tools.industry_research import IndustryResearchTool
from ..tools.financial_data import FinancialDataTool

logging.basicConfig(level=logging.INFO)


class CompanyQAAgent:
    """Company Q&A agent with MLflow tracing and proper tool calling"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """Initialize the Q&A agent"""

        # Setup MLflow tracking
        self._setup_mlflow()
        
        # Initialize OpenAI client
        mlflow.openai.autolog()
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Model configuration
        self.model_name = model_name or getattr(settings, 'default_model', 'gpt-4o-mini')
        self.temperature = temperature or getattr(settings, 'llm_temperature', 0.1)

        # Initialize tools
        self.industry_tool = IndustryResearchTool()
        self.financial_tool = FinancialDataTool()
        
        # Define tool schemas for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_financial_data",
                    "description": "Get financial data, stock prices, metrics, and financial performance for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company": {
                                "type": "string",
                                "description": "Company name or stock ticker"
                            }
                        },
                        "required": ["company"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_industry_research",
                    "description": "Get industry analysis, market trends, competitive position, and business overview for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company": {
                                "type": "string",
                                "description": "Company name"
                            }
                        },
                        "required": ["company"]
                    }
                }
            }
        ]
        
        logging.info("Company Q&A Agent initialized with tool calling")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiments"""
        try:
            mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
            mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
            
            logging.info("MLflow tracking initialized")
            
        except Exception as e:
            logging.warning(f"MLflow setup warning: {e}")
    
    @mlflow.trace(name="get_financial_data", span_type=SpanType.TOOL)
    def get_financial_data(self, company: str) -> Dict[str, Any]:
        """Tool function to get financial data"""
        try:
            logging.info(f"Getting financial data for {company}")
            result = self.financial_tool._run(company)
            data = json.loads(result)
            
            if "error" in data:
                return {"error": f"Failed to get financial data: {data['error']}"}
            
            return data
        except Exception as e:
            logging.error(f"Financial tool error: {str(e)}")
            return {"error": f"Financial data error: {str(e)}"}
    
    @mlflow.trace(name="get_industry_research", span_type=SpanType.TOOL)
    def get_industry_research(self, company: str) -> Dict[str, Any]:
        """Tool function to get industry research"""
        try:
            logging.info(f"Getting industry research for {company}")
            result = self.industry_tool._run(company)
            data = json.loads(result)
            
            if "error" in data:
                return {"error": f"Failed to get industry data: {data['error']}"}
            
            return data
        except Exception as e:
            logging.error(f"Industry tool error: {str(e)}")
            return {"error": f"Industry research error: {str(e)}"}
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def ask_question(self, question: str, company: str) -> Dict[str, Any]:
        """
        Ask a question about a company using tool calling
        
        Args:
            question: Question about the company
            company: Company name or ticker
            
        Returns:
            Dictionary with answer and metadata
        """
        
        logging.info(f"Question: {question}")
        logging.info(f"Company: {company}")
        
        # Prepare the conversation with system message
        messages = [
            {
                "role": "system",
                "content": """You are a helpful financial assistant that answers questions about companies.

                You have access to two tools:
                - get_financial_data: For financial metrics, stock prices, ratios, earnings, revenue
                - get_industry_research: For market analysis, competition, business model, industry trends
                
                Use the appropriate tool(s) to gather information, then provide a clear, accurate answer to the user's question.
                Be specific and include relevant numbers/data from the tool results."""
            },
            {
                "role": "user", 
                "content": f"Question about {company}: {question}"
            }
        ]
        
        try:

            with mlflow.start_run(run_name=f"QA_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Step 1: Get initial response with potential tool calls
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    temperature=self.temperature
                )
                
                ai_msg = response.choices[0].message
                messages.append(ai_msg)
                
                tools_used = []
                
                # Step 2: Handle tool calls if requested
                if tool_calls := ai_msg.tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        # Execute the appropriate tool
                        if function_name == "get_financial_data":
                            tool_result = self.get_financial_data(company)
                            tools_used.append("financial")
                        elif function_name == "get_industry_research":
                            tool_result = self.get_industry_research(company)
                            tools_used.append("industry")
                        else:
                            tool_result = {"error": f"Unknown tool: {function_name}"}
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        })
                    
                    # Step 3: Get final response with tool results
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature
                    )
                
                final_answer = response.choices[0].message.content
                
                return {
                    "answer": final_answer,
                    "tools_used": tools_used,
                    "success": True,
                    "company": company,
                    "question": question,
                    "conversation": messages
                }
            
        except Exception as e:
            logging.error(f"Error in ask_question: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "tools_used": [],
                "success": False,
                "company": company,
                "question": question,
                "error": str(e)
            }
