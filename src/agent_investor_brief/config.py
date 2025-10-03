"""
Configuration module for Agent Investor Brief
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field("sqlite:///mlflow.db", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("investor_brief_analysis", env="MLFLOW_EXPERIMENT_NAME")
    
    # LLM Configuration
    default_model: str = Field("gpt-4o-mini", env="DEFAULT_MODEL")
    llm_temperature: float = Field(0.1, env="LLM_TEMPERATURE")

    # Prompts version
    analysis_prompt_version: str = Field("v1", env="ANALYSIS_PROMPT_VERSION")
    brief_prompt_version: str = Field("v1", env="BRIEF_PROMPT_VERSION")
    
    # Tool Configuration
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")
    financial_data_period: str = Field("1y", env="FINANCIAL_DATA_PERIOD")  # 1y, 2y, 5y
    
    # Output Configuration
    output_dir: Path = Field(Path("outputs"), env="OUTPUT_DIR")
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")
    verbose_logging: bool = Field(False, env="VERBOSE_LOGGING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def setup_environment() -> Settings:
    """
    Initialize environment and return settings
    """
    # Load environment variables
    load_dotenv()
    
    # Create settings instance
    settings = Settings()
    
    # Create output directory
    settings.output_dir.mkdir(exist_ok=True)
    
    # Validate required API keys
    if not settings.openai_api_key:
        logging.warning("OPENAI_API_KEY not set. Some features may not work.")
    
    # Set OpenAI API key in environment (required by langchain)
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    
    logging.info("Environment setup completed")
    logging.info(f"MLflow tracking: {settings.mlflow_tracking_uri}")
    logging.info(f"Default model: {settings.default_model}")
    logging.info(f"Output directory: {settings.output_dir}")
    
    return settings


def get_settings() -> Settings:
    """Get application settings (cached)"""
    if not hasattr(get_settings, "_settings"):
        get_settings._settings = setup_environment()
    return get_settings._settings


# Global settings instance
settings = get_settings()


# Tool-specific configurations
TOOL_CONFIGS = {
    "industry_research": {
        "max_results": settings.max_search_results,
        "timeout": 30,
        "user_agent": "Mozilla/5.0 (compatible; InvestorBrief/1.0)"
    },
    "financial_data": {
        "period": settings.financial_data_period,
        "timeout": 20,
        "retry_attempts": 3
    },
    "analysis_builder": {
        "risk_free_rate": 0.02,  # 2% default risk-free rate
        "market_return": 0.10,   # 10% expected market return
        "confidence_interval": 0.95
    }
}


# MLflow experiment configuration
MLFLOW_CONFIG = {
    "experiment_name": settings.mlflow_experiment_name,
    "tracking_uri": settings.mlflow_tracking_uri,
    "auto_log": settings.enable_tracing,
}


# Validation functions
def validate_api_keys():
    """Validate that required API keys are present"""
    missing_keys = []
    
    if not settings.openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return True


def validate_environment():
    """Comprehensive environment validation"""
    try:
        # Check API keys
        validate_api_keys()
        
        # Check MLflow setup
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Check LangChain setup
        from langchain_openai import ChatOpenAI
        ChatOpenAI(model=settings.default_model, api_key=settings.openai_api_key)
        
        logging.info("✅ All environment validations passed")
        return True
        
    except Exception as e:
        logging.info(f"❌ Environment validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    settings = setup_environment()
    logging.info("\n🧪 Testing configuration...")
    
    if validate_environment():
        logging.info("🎉 Configuration is ready!")
    else:
        logging.info("🔧 Please check your configuration and API keys")