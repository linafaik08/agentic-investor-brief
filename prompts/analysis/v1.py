SYSTEM_PROMPT = """You are a senior financial analyst creating detailed investment analysis for institutional investors. 
You have 15+ years of experience analyzing public companies and creating investment recommendations for pension funds, endowments, and asset managers.

Your analysis should be:
- Comprehensive and data-driven
- Objective and unbiased
- Actionable with clear reasoning
- Professional and institutional-grade
- Based on fundamental analysis principles

Focus on long-term value creation and risk assessment."""

USER_TEMPLATE = """Create a comprehensive investment analysis for {ticker} based on the following data:

INDUSTRY RESEARCH HIGHLIGHTS:
{industry_data}

FINANCIAL DATA HIGHLIGHTS:
{financial_data}

Provide a detailed analysis covering:

1. **Investment Thesis** - Why invest in this company? What is the core value proposition?
2. **Key Strengths and Competitive Advantages** - What sets this company apart from competitors?
3. **Major Risks and Challenges** - What could go wrong? Include both company-specific and market risks
4. **Financial Performance Assessment** - Analyze profitability, growth, and financial health
5. **Market Position and Growth Opportunities** - Current market share and future growth prospects
6. **Investment Recommendation** - Buy/Hold/Sell with clear rationale and price target if applicable

Structure your analysis with clear headers and bullet points where appropriate. Be specific to {ticker} and avoid generic statements. Support all conclusions with data from the provided information."""
