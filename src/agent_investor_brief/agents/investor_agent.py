"""
Investor Agent - Main orchestrator for creating comprehensive investment briefs
"""

import json
import mlflow
import mlflow.langchain
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import settings, MLFLOW_CONFIG
from ..tools.industry_research import IndustryResearchTool
from ..tools.financial_data import FinancialDataTool
from ..tools.analysis_builder import AnalysisBuilderTool


class InvestorAgent:
    """
    Main agent for creating comprehensive investment briefs
    
    This agent orchestrates the entire process:
    1. Research industry and market conditions
    2. Fetch comprehensive financial data  
    3. Build detailed investment analysis
    4. Generate professional investor brief
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """Initialize the investor agent"""
        
        # Setup MLflow tracking
        self._setup_mlflow()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name or settings.default_model,
            temperature=temperature or settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        # Initialize tools
        self.tools = [
            IndustryResearchTool(),
            FinancialDataTool(), 
            AnalysisBuilderTool()
        ]
        
        # Setup memory for conversational capability
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=settings.verbose_logging,
            max_iterations=10,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        # Create analysis chain for final brief generation
        self.brief_chain = self._create_brief_chain()
        
        print("✅ Investor Agent initialized successfully")
        print(f"🤖 Model: {self.llm.model_name}")
        print(f"🔧 Tools: {[tool.name for tool in self.tools]}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiments"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
            
            # Set or create experiment
            mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
            
            # Enable autologging for LangChain
            if MLFLOW_CONFIG["auto_log"]:
                mlflow.langchain.autolog(
                    log_models=MLFLOW_CONFIG["log_models"],
                    log_input_examples=MLFLOW_CONFIG["log_input_examples"],
                    log_model_signatures=MLFLOW_CONFIG["log_model_signatures"]
                )
            
            print("✅ MLflow tracking initialized")
            
        except Exception as e:
            print(f"⚠️  MLflow setup warning: {e}")
    
    def _create_brief_chain(self):
        """Create chain for final brief generation and formatting"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior financial analyst creating professional investment briefs for institutional investors.

Your role is to synthesize research and analysis into a clear, actionable investment brief that helps investors make informed decisions.

Key requirements:
- Be objective and data-driven
- Highlight both opportunities and risks
- Provide clear recommendations with rationale
- Use professional financial language
- Structure information logically
- Include quantitative metrics where available

Create a comprehensive but concise brief that busy investors can quickly digest and act upon."""),
            
            ("human", """Based on the following analysis data, create a professional investment brief:

Company/Ticker: {ticker}
Industry Research: {industry_research}
Financial Analysis: {financial_data}
Investment Analysis: {investment_analysis}

Please create a well-structured investment brief that synthesizes this information into actionable insights.""")
        ])
        
        return prompt | self.llm | StrOutputParser()
    
    def create_investment_brief(self, company_or_ticker: str, include_conversation: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive investment brief for a company
        
        Args:
            company_or_ticker: Company name or stock ticker
            include_conversation: Whether to include conversational context
            
        Returns:
            Dictionary containing the investment brief and analysis data
        """
        
        with mlflow.start_run(run_name=f"investment_brief_{company_or_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            
            try:
                # Log input parameters
                mlflow.log_param("company_ticker", company_or_ticker)
                mlflow.log_param("model_name", self.llm.model_name)
                mlflow.log_param("include_conversation", include_conversation)
                
                print(f"\n🎯 Creating Investment Brief for: {company_or_ticker}")
                print("=" * 60)
                
                # Step 1: Industry Research
                print("📊 Step 1: Conducting Industry Research...")
                industry_research = self._conduct_industry_research(company_or_ticker)
                mlflow.log_text(json.dumps(industry_research, indent=2), "industry_research.json")
                
                # Step 2: Financial Data Collection
                print("💰 Step 2: Fetching Financial Data...")
                financial_data = self._fetch_financial_data(company_or_ticker)
                mlflow.log_text(json.dumps(financial_data, indent=2), "financial_data.json")
                
                # Step 3: Investment Analysis
                print("🔍 Step 3: Building Investment Analysis...")
                investment_analysis = self._build_investment_analysis(
                    financial_data, industry_research, company_or_ticker
                )
                mlflow.log_text(json.dumps(investment_analysis, indent=2), "investment_analysis.json")
                
                # Step 4: Generate Final Brief
                print("📝 Step 4: Generating Investment Brief...")
                final_brief = self._generate_final_brief(
                    company_or_ticker, industry_research, financial_data, investment_analysis
                )
                
                # Step 5: Create Conversational Summary (if requested)
                conversational_summary = ""
                if include_conversation:
                    print("💬 Step 5: Creating Conversational Summary...")
                    conversational_summary = self._create_conversational_summary(
                        company_or_ticker, final_brief
                    )
                
                # Compile final results
                results = {
                    "company_ticker": company_or_ticker,
                    "brief_generated_at": datetime.now().isoformat(),
                    "investment_brief": final_brief,
                    "conversational_summary": conversational_summary,
                    "supporting_data": {
                        "industry_research": industry_research,
                        "financial_data": financial_data,
                        "investment_analysis": investment_analysis
                    },
                    "agent_metadata": {
                        "model_used": self.llm.model_name,
                        "tools_used": [tool.name for tool in self.tools],
                        "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
                    }
                }
                
                # Log final results
                mlflow.log_text(final_brief, "investment_brief.md")
                if conversational_summary:
                    mlflow.log_text(conversational_summary, "conversational_summary.txt")
                
                mlflow.log_metric("analysis_completeness", self._calculate_completeness_score(results))
                mlflow.log_metric("brief_length_words", len(final_brief.split()))
                
                print("✅ Investment Brief Created Successfully!")
                print(f"📋 Brief length: {len(final_brief.split())} words")
                print(f"🆔 MLflow Run ID: {results['agent_metadata']['mlflow_run_id']}")
                
                return results
                
            except Exception as e:
                error_msg = f"Investment brief creation failed: {str(e)}"
                print(f"❌ {error_msg}")
                mlflow.log_param("error", error_msg)
                
                return {
                    "error": error_msg,
                    "company_ticker": company_or_ticker,
                    "timestamp": datetime.now().isoformat()
                }
    
    def _conduct_industry_research(self, company_or_ticker: str) -> Dict:
        """Conduct industry research using the research tool"""
        try:
            research_tool = IndustryResearchTool()
            result = research_tool._run(company_or_ticker)
            return json.loads(result)
        except Exception as e:
            return {"error": f"Industry research failed: {str(e)}"}
    
    def _fetch_financial_data(self, ticker: str) -> Dict:
        """Fetch financial data using the financial data tool"""
        try:
            financial_tool = FinancialDataTool()
            result = financial_tool._run(ticker)
            return json.loads(result)
        except Exception as e:
            return {"error": f"Financial data fetch failed: {str(e)}"}
    
    def _build_investment_analysis(self, financial_data: Dict, industry_research: Dict, ticker: str) -> Dict:
        """Build comprehensive investment analysis"""
        try:
            analysis_tool = AnalysisBuilderTool()
            
            analysis_input = json.dumps({
                "financial_data": financial_data,
                "industry_research": industry_research,
                "ticker": ticker
            })
            
            result = analysis_tool._run(analysis_input)
            return json.loads(result)
        except Exception as e:
            return {"error": f"Investment analysis failed: {str(e)}"}
    
    def _generate_final_brief(self, ticker: str, industry_research: Dict, 
                            financial_data: Dict, investment_analysis: Dict) -> str:
        """Generate the final investment brief using the brief chain"""
        try:
            # Format data for the chain
            brief_response = self.brief_chain.invoke({
                "ticker": ticker,
                "industry_research": json.dumps(industry_research, indent=2)[:2000] + "...",  # Truncate for context
                "financial_data": json.dumps(financial_data, indent=2)[:2000] + "...",
                "investment_analysis": json.dumps(investment_analysis, indent=2)[:3000] + "..."
            })
            
            return brief_response
            
        except Exception as e:
            return f"Brief generation failed: {str(e)}"
    
    def _create_conversational_summary(self, company_ticker: str, brief: str) -> str:
        """Create a conversational summary of the brief"""
        try:
            conversation_prompt = f"""
            Based on the investment brief below, create a conversational summary as if you're 
            explaining the investment opportunity to a colleague in 2-3 paragraphs. 
            Be natural, engaging, and highlight the key points an investor should know.
            
            Investment Brief:
            {brief[:2000]}...
            """
            
            summary = self.llm.invoke(conversation_prompt).content
            return summary
            
        except Exception as e:
            return f"Conversational summary generation failed: {str(e)}"
    
    def _calculate_completeness_score(self, results: Dict) -> float:
        """Calculate how complete the analysis is (0-100)"""
        try:
            supporting_data = results.get("supporting_data", {})
            
            completeness_checks = [
                "industry_research" in supporting_data and not "error" in str(supporting_data["industry_research"]),
                "financial_data" in supporting_data and not "error" in str(supporting_data["financial_data"]),
                "investment_analysis" in supporting_data and not "error" in str(supporting_data["investment_analysis"]),
                len(results.get("investment_brief", "")) > 1000,  # Substantial brief
                results.get("conversational_summary", "") != ""  # Has summary
            ]
            
            score = (sum(completeness_checks) / len(completeness_checks)) * 100
            return round(score, 1)
            
        except:
            return 0.0
    
    def ask_question(self, question: str) -> str:
        """
        Ask a follow-up question about the analysis or company
        
        Args:
            question: Question about the company or analysis
            
        Returns:
            Agent's response to the question
        """
        try:
            with mlflow.start_run(run_name=f"question_{datetime.now().strftime('%H%M%S')}", nested=True):
                mlflow.log_param("question", question)
                
                response = self.agent.run(question)
                
                mlflow.log_text(response, "agent_response.txt")
                mlflow.log_metric("response_length", len(response))
                
                return response
                
        except Exception as e:
            return f"Question processing failed: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for message in messages:
                history.append({
                    "type": message.__class__.__name__,
                    "content": message.content[:500] + "..." if len(message.content) > 500 else message.content
                })
            
            return history
            
        except Exception as e:
            return [{"error": f"Failed to retrieve history: {str(e)}"}]
    
    def clear_conversation(self):
        """Clear the conversation memory"""
        self.memory.clear()
        print("🧹 Conversation memory cleared")
    
    def export_brief(self, results: Dict, format: str = "markdown", output_path: str = None) -> str:
        """
        Export the investment brief to a file
        
        Args:
            results: Results from create_investment_brief
            format: Export format ("markdown", "text", "json")
            output_path: Optional custom output path
            
        Returns:
            Path to the exported file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company = results.get("company_ticker", "unknown").replace("/", "_")
            
            if not output_path:
                output_path = settings.output_dir / f"investment_brief_{company}_{timestamp}.{format}"
            
            if format == "markdown":
                content = results.get("investment_brief", "")
            elif format == "json":
                content = json.dumps(results, indent=2, default=str)
            else:  # text
                content = results.get("investment_brief", "")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"📁 Brief exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""


# Convenience functions for direct usage
def create_quick_brief(company_or_ticker: str, model_name: str = None) -> Dict[str, Any]:
    """
    Quick function to create an investment brief
    
    Args:
        company_or_ticker: Company name or ticker symbol
        model_name: Optional LLM model name to use
        
    Returns:
        Investment brief results
    """
    agent = InvestorAgent(model_name=model_name)
    return agent.create_investment_brief(company_or_ticker)


def analyze_multiple_companies(tickers: List[str], model_name: str = None) -> Dict[str, Dict]:
    """
    Analyze multiple companies in batch
    
    Args:
        tickers: List of ticker symbols
        model_name: Optional LLM model name to use
        
    Returns:
        Dictionary mapping tickers to their analysis results
    """
    agent = InvestorAgent(model_name=model_name)
    results = {}
    
    for ticker in tickers:
        print(f"\n🔄 Processing {ticker}...")
        results[ticker] = agent.create_investment_brief(ticker, include_conversation=False)
    
    return results


if __name__ == "__main__":
    # Test the agent
    print("🧪 Testing Investor Agent...")
    
    # Create agent instance
    agent = InvestorAgent()
    
    # Test with a sample ticker
    test_ticker = "AAPL"
    print(f"\n📋 Creating brief for {test_ticker}...")
    
    # Create brief
    results = agent.create_investment_brief(test_ticker)
    
    # Print summary
    if "error" not in results:
        print("\n✅ Brief created successfully!")
        print(f"Brief preview:\n{results['investment_brief'][:500]}...")
        
        # Test follow-up question
        question = f"What are the main risks for investing in {test_ticker}?"
        print(f"\n❓ Asking: {question}")
        response = agent.ask_question(question)
        print(f"🤖 Response: {response[:200]}...")
        
    else:
        print(f"❌ Brief creation failed: {results['error']}")