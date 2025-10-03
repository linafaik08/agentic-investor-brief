"""
Simplified Investor Agent - Direct OpenAI API approach
"""

import json
import mlflow
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI

from ..config import settings, MLFLOW_CONFIG
from ..tools.industry_research import IndustryResearchTool
from ..tools.financial_data import FinancialDataTool
from ..prompt_manager import PromptVersionManager

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class InvestorAgent:
    """
    Simplified agent that executes steps sequentially using direct OpenAI API calls
    """
    
    def __init__(
            self,
            model_name: str = None, 
            temperature: float = None,
            analysis_prompt_version: str = None, 
            brief_prompt_version: str = None,
            prompts_dir: str = "prompts"
            ):
        """Initialize the simplified investor agent"""
        
        # Setup MLflow tracking
        self._setup_mlflow()
        
        # Initialize OpenAI client
        mlflow.openai.autolog()
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Model configuration
        self.model_name = model_name or getattr(settings, 'default_model', 'gpt-5-nano')
        self.temperature = temperature or getattr(settings, 'llm_temperature', 0.1)

        # Prompt versions

        self.prompt_manager = PromptVersionManager(prompts_dir)
        self.analysis_prompt_version = analysis_prompt_version or getattr(settings, 'analysis_prompt_version', 1)
        self.brief_prompt_version = brief_prompt_version or getattr(settings, 'brief_prompt_version', 1)
        # Auto-register prompts in MLflow if requested
        self._register_current_prompts_in_mlflow()
        
        # Initialize tools
        self.industry_tool = IndustryResearchTool()
        self.financial_tool = FinancialDataTool()
        
        logging.info("Simplified Investor Agent initialized successfully")
        logging.info(f"Model: {self.model_name}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiments"""
        try:
            mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
            mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
            
            logging.info("MLflow tracking initialized")
            
        except Exception as e:
            logging.warning(f"MLflow setup warning: {e}")
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to OpenAI API"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return f"Error: {str(e)}"

    def _register_current_prompts_in_mlflow(self, author: str = None):
        """Register currently used prompts in MLflow"""
        # Register analysis prompt
        analysis_mlflow_name = self.prompt_manager.register_prompt_in_mlflow(
            "analysis", 
            self.analysis_prompt_version,
              author,
            tags={"current_analysis": "true"}
        )
        
        # Register brief prompt  
        brief_mlflow_name = self.prompt_manager.register_prompt_in_mlflow(
            "brief", self.brief_prompt_version, author,
            tags={"current_brief": "true"}
        )
        
        # Store MLflow names for later reference
        self.analysis_mlflow_prompt = analysis_mlflow_name
        self.brief_mlflow_prompt = brief_mlflow_name
        
        logging.info(f"Registered current prompts: {analysis_mlflow_name}, {brief_mlflow_name}")
    
    
    def create_investment_brief(self, company_or_ticker: str) -> Dict[str, Any]:
        """
        Create comprehensive investment brief using direct sequential execution
        """
        
        with mlflow.start_run(run_name=f"{company_or_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            
            # Log input parameters
            mlflow.log_param("company_ticker", company_or_ticker)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("analysis_prompt_version", self.analysis_prompt_version)
            mlflow.log_param("brief_prompt_version", self.brief_prompt_version)
            
            logging.info(f"Creating Investment Brief for: {company_or_ticker}")
            
            # Step 1: Gather Financial Data
            logging.info("Step 1: Gathering financial data...")
            
            financial_result = self.financial_tool._run(company_or_ticker)
            financial_data = json.loads(financial_result)
            logging.info("✅ Financial data completed")
            
            # Step 2: Gather Industry Research
            logging.info("Step 2: Gathering industry research...")
            # Get company name from financial data
            if financial_data and "detailed_data" in financial_data:
                company_name = financial_data["detailed_data"].get("company_info", {}).get("company_name", company_or_ticker)
            else:
                company_name = company_or_ticker
            
            industry_result = self.industry_tool._run(company_name)
            industry_research = json.loads(industry_result)
            logging.info("✅ Industry research completed")
            
            # Step 3: Create Investment Analysis
            logging.info("Step 3: Creating investment analysis...")
            investment_analysis = self._create_analysis(
                company_or_ticker, industry_research, financial_data
            )
            
            # Step 4: Generate Final Brief
            logging.info("Step 4: Generating final brief...")
            final_brief = self._generate_brief(
                company_or_ticker, industry_research, financial_data, investment_analysis
            )
            
            # Compile results
            results = {
                "company_ticker": company_or_ticker,
                "brief_generated_at": datetime.now().isoformat(),
                "investment_brief": final_brief,
                "supporting_data": {
                    "industry_research": industry_research,
                    "financial_data": financial_data,
                    "investment_analysis": investment_analysis
                },
                "agent_metadata": {
                    "model_used": self.model_name,
                    "approach": "simplified_sequential_openai",
                    "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
                    "brief_prompt_version": self.brief_prompt_version,
                    "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
                }
            }
            
            # Log results to MLflow
            if isinstance(results["investment_brief"], str):
                mlflow.log_text(results["investment_brief"], "investment_brief.md")
                mlflow.log_text(json.dumps(results["supporting_data"], indent=2), "supporting_data.json")
                
                if results["investment_brief"]:
                    mlflow.log_metric("brief_length_words", len(results["investment_brief"].split()))
                    logging.info("✅ Investment Brief Created Successfully!")
                else:
                    logging.error("❌ Investment Brief is empty")
            
            return results
    
    def _create_analysis(self, ticker: str, industry_data: Dict, financial_data: Dict) -> Dict:
        """Create investment analysis using universal prompt manager"""
        
        logging.info(f"Creating investment analysis with prompt version {self.analysis_prompt_version}...")
        
        # Check if we have required data
        if not industry_data or "error" in industry_data or not financial_data or "error" in financial_data:
            error_msg = f"Missing or failed required data for analysis."
            logging.error(error_msg)
            return {"error": error_msg}
        
        # Get formatted prompts using universal manager
        formatted_prompts = self.prompt_manager.format_prompt(
            "analysis", 
            self.analysis_prompt_version,
            ticker=ticker,
            industry_data=json.dumps(industry_data, indent=2),
            financial_data=json.dumps(financial_data, indent=2)
        )
        
        logging.info(f"Using analysis prompt version: {self.analysis_prompt_version}")
        logging.info(f"Prompt length: {len(formatted_prompts['user'])} characters")
        
        # Call OpenAI API with formatted prompts
        response = self._call_openai(formatted_prompts["system"], formatted_prompts["user"])
        
        if response and not response.startswith("Error:"):
            analysis_result = {
                "analysis": response,
                "prompt_version": self.analysis_prompt_version
            }
            logging.info(f"Investment analysis completed successfully. Length: {len(response)}")
            return analysis_result
        else:
            error_msg = f"Analysis generation failed: {response}"
            logging.error(error_msg)
            return {"error": error_msg}
    
    def _generate_brief(self, ticker: str, industry_data: Dict, financial_data: Dict, analysis_data: Dict) -> str:
        """Generate final investment brief using OpenAI API"""
        
        logging.info("Generating final investment brief...")
        
        # Check if analysis was successful
        if not analysis_data.get("analysis"):
            error_msg = "Cannot generate brief: missing or failed investment analysis"
            logging.error(error_msg)
            return error_msg
        
        # Prepare data summaries
        industry_summary = self._extract_key_info(industry_data, "industry")
        financial_summary = self._extract_key_info(financial_data, "financial")
        
        # Get formatted prompts using universal manager
        formatted_prompts = self.prompt_manager.format_prompt(
            "brief",
            self.brief_prompt_version,
            ticker=ticker,
            industry_summary=industry_summary,
            financial_summary=financial_summary,
            analysis=analysis_data.get("analysis", "")
        )
        
        logging.info(f"Using brief prompt version: {self.brief_prompt_version}")
        
        # Call OpenAI API with formatted prompts
        brief = self._call_openai(formatted_prompts["system"], formatted_prompts["user"])
        
        if brief and not brief.startswith("Error:"):
            logging.info("✅ Investment brief generated successfully")
            return brief
        else:
            error_msg = f"Brief generation failed: {brief}"
            logging.error(error_msg)
            return error_msg
        
    def _extract_key_info(self, data: Dict, data_type: str) -> str:
        """Extract key information from data dictionaries"""
        
        if not data or "error" in data:
            return f"No {data_type} data available"
        
        if data_type == "industry":
            # Extract key industry insights
            summary = data.get("research_summary", "")
            if summary:
                return summary
            else:
                return json.dumps(data, indent=2)
                
        elif data_type == "financial":
            # Handle the correct financial data structure
            if isinstance(data, dict):
                if "financial_summary" in data:
                    return data["financial_summary"]
                elif "detailed_data" in data:
                    key_metrics = data["detailed_data"].get("key_metrics", {})
                    price_data = data["detailed_data"].get("price_data", {})
                    return f"Key Metrics: {json.dumps(key_metrics, indent=2)}\nPrice Data: {json.dumps(price_data, indent=2)}..."
                else:
                    return json.dumps(data, indent=2)
            else:
                return str(data)
        
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Test the simplified agent
    print("Testing Investor Agent...")
    
    agent = InvestorAgent()
    results = agent.create_investment_brief("AAPL")
    
    if "error" not in results and results.get("investment_brief"):
        print("✅ Brief created successfully!")
        print(f"Brief preview:\n{results['investment_brief'][:500]}...")
    else:
        print(f"❌ Brief creation failed")
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("Investment brief is empty")