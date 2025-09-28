"""
Financial Data Tool - Yahoo Finance integration for stock and financial data
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from pydantic import Field

from ..config import settings, TOOL_CONFIGS


class FinancialDataTool(BaseTool):
    """Tool for fetching comprehensive financial data from Yahoo Finance"""
    
    name: str = "financial_data"
    description: str = """
    Fetch comprehensive financial data for publicly traded companies using Yahoo Finance.
    Provides:
    - Stock price history and current metrics
    - Financial statements (income, balance sheet, cash flow)
    - Key financial ratios and performance indicators
    - Analyst recommendations and target prices
    - Dividend information and yield
    
    Input should be a stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT")
    """
    
    def _run(self, ticker: str) -> str:
        """Fetch comprehensive financial data for the given ticker"""
        try:
            config = TOOL_CONFIGS["financial_data"]
            ticker = ticker.upper().strip()
            
            # Initialize yfinance ticker
            stock = yf.Ticker(ticker)
            
            # Comprehensive data collection
            financial_data = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "company_info": self._get_company_info(stock),
                "price_data": self._get_price_data(stock, config),
                "financial_statements": self._get_financial_statements(stock),
                "key_metrics": self._get_key_metrics(stock),
                "analyst_data": self._get_analyst_data(stock),
                "technical_indicators": self._calculate_technical_indicators(stock, config)
            }
            
            # Create investor-focused summary
            summary = self._create_financial_summary(financial_data)
            
            return json.dumps({
                "financial_summary": summary,
                "detailed_data": financial_data,
                "data_quality": self._assess_data_quality(financial_data),
                "last_updated": financial_data["timestamp"]
            }, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Financial data fetch failed: {str(e)}",
                "ticker": ticker,
                "suggestion": "Verify ticker symbol is correct and company is publicly traded"
            })
    
    def _get_company_info(self, stock: yf.Ticker) -> Dict:
        """Get basic company information"""
        try:
            info = stock.info
            return {
                "company_name": info.get("longName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "employees": info.get("fullTimeEmployees", 0),
                "country": info.get("country", "Unknown"),
                "website": info.get("website", ""),
                "business_summary": info.get("longBusinessSummary", "")[:500] + "..." if info.get("longBusinessSummary") else ""
            }
        except Exception as e:
            return {"error": f"Company info unavailable: {str(e)}"}
    
    def _get_price_data(self, stock: yf.Ticker, config: Dict) -> Dict:
        """Get current and historical price data"""
        try:
            info = stock.info
            hist = stock.history(period=config["period"])
            
            if hist.empty:
                return {"error": "No price history available"}
            
            current_price = hist['Close'].iloc[-1]
            price_52w_high = hist['High'].max()
            price_52w_low = hist['Low'].min()
            
            # Calculate returns
            returns_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
            returns_1m = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100 if len(hist) > 21 else 0
            returns_3m = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100 if len(hist) > 63 else 0
            returns_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            return {
                "current_price": float(current_price),
                "previous_close": info.get("previousClose", current_price),
                "day_change": float(returns_1d),
                "volume": int(hist['Volume'].iloc[-1]),
                "avg_volume": int(hist['Volume'].mean()),
                "52_week_high": float(price_52w_high),
                "52_week_low": float(price_52w_low),
                "returns": {
                    "1_day": float(returns_1d),
                    "1_month": float(returns_1m),
                    "3_months": float(returns_3m),
                    "ytd": float(returns_ytd)
                },
                "volatility_30d": float(hist['Close'].pct_change().rolling(30).std() * np.sqrt(252) * 100),
                "beta": info.get("beta", 1.0)
            }
        except Exception as e:
            return {"error": f"Price data unavailable: {str(e)}"}
    
    def _get_financial_statements(self, stock: yf.Ticker) -> Dict:
        """Get key financial statement data"""
        try:
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            financial_statements = {}
            
            # Income Statement (most recent year)
            if not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]
                financial_statements["income_statement"] = {
                    "total_revenue": float(latest_income.get("Total Revenue", 0)),
                    "gross_profit": float(latest_income.get("Gross Profit", 0)),
                    "operating_income": float(latest_income.get("Operating Income", 0)),
                    "net_income": float(latest_income.get("Net Income", 0)),
                    "ebitda": float(latest_income.get("EBITDA", 0)),
                    "eps": float(latest_income.get("Basic EPS", 0))
                }
            
            # Balance Sheet (most recent quarter)
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                financial_statements["balance_sheet"] = {
                    "total_assets": float(latest_balance.get("Total Assets", 0)),
                    "total_debt": float(latest_balance.get("Total Debt", 0)),
                    "total_equity": float(latest_balance.get("Total Equity Gross Minority Interest", 0)),
                    "cash_and_equivalents": float(latest_balance.get("Cash And Cash Equivalents", 0)),
                    "working_capital": float(latest_balance.get("Working Capital", 0))
                }
            
            # Cash Flow (most recent year)
            if not cash_flow.empty:
                latest_cashflow = cash_flow.iloc[:, 0]
                financial_statements["cash_flow"] = {
                    "operating_cash_flow": float(latest_cashflow.get("Operating Cash Flow", 0)),
                    "investing_cash_flow": float(latest_cashflow.get("Investing Cash Flow", 0)),
                    "financing_cash_flow": float(latest_cashflow.get("Financing Cash Flow", 0)),
                    "free_cash_flow": float(latest_cashflow.get("Free Cash Flow", 0)),
                    "capital_expenditure": float(latest_cashflow.get("Capital Expenditure", 0))
                }
            
            return financial_statements
            
        except Exception as e:
            return {"error": f"Financial statements unavailable: {str(e)}"}
    
    def _get_key_metrics(self, stock: yf.Ticker) -> Dict:
        """Calculate key financial metrics and ratios"""
        try:
            info = stock.info
            
            # Valuation metrics
            pe_ratio = info.get("trailingPE", 0)
            pb_ratio = info.get("priceToBook", 0)
            ps_ratio = info.get("priceToSalesTrailing12Months", 0)
            peg_ratio = info.get("pegRatio", 0)
            
            # Profitability metrics
            profit_margin = info.get("profitMargins", 0) * 100
            operating_margin = info.get("operatingMargins", 0) * 100
            roe = info.get("returnOnEquity", 0) * 100
            roa = info.get("returnOnAssets", 0) * 100
            
            # Financial health
            debt_to_equity = info.get("debtToEquity", 0)
            current_ratio = info.get("currentRatio", 0)
            quick_ratio = info.get("quickRatio", 0)
            
            # Growth metrics
            revenue_growth = info.get("revenueGrowth", 0) * 100
            earnings_growth = info.get("earningsGrowth", 0) * 100
            
            # Dividend information
            dividend_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
            payout_ratio = info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0
            
            return {
                "valuation_metrics": {
                    "pe_ratio": float(pe_ratio),
                    "pb_ratio": float(pb_ratio),
                    "ps_ratio": float(ps_ratio),
                    "peg_ratio": float(peg_ratio),
                    "ev_ebitda": float(info.get("enterpriseToEbitda", 0))
                },
                "profitability_metrics": {
                    "profit_margin_pct": float(profit_margin),
                    "operating_margin_pct": float(operating_margin),
                    "roe_pct": float(roe),
                    "roa_pct": float(roa),
                    "roic_pct": float(info.get("returnOnCapital", 0) * 100)
                },
                "financial_health": {
                    "debt_to_equity": float(debt_to_equity),
                    "current_ratio": float(current_ratio),
                    "quick_ratio": float(quick_ratio),
                    "interest_coverage": float(info.get("interestCoverage", 0))
                },
                "growth_metrics": {
                    "revenue_growth_pct": float(revenue_growth),
                    "earnings_growth_pct": float(earnings_growth),
                    "book_value_growth_pct": float(info.get("bookValue", 0))
                },
                "dividend_info": {
                    "dividend_yield_pct": float(dividend_yield),
                    "payout_ratio_pct": float(payout_ratio),
                    "dividend_rate": float(info.get("dividendRate", 0))
                }
            }
            
        except Exception as e:
            return {"error": f"Key metrics calculation failed: {str(e)}"}
    
    def _get_analyst_data(self, stock: yf.Ticker) -> Dict:
        """Get analyst recommendations and price targets"""
        try:
            info = stock.info
            
            analyst_data = {
                "recommendation": info.get("recommendationKey", "none"),
                "target_price": float(info.get("targetMeanPrice", 0)),
                "target_high": float(info.get("targetHighPrice", 0)),
                "target_low": float(info.get("targetLowPrice", 0)),
                "analyst_count": int(info.get("numberOfAnalystOpinions", 0)),
                "recommendation_mean": float(info.get("recommendationMean", 0))
            }
            
            # Try to get detailed recommendations
            try:
                recommendations = stock.recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.tail(1)
                    analyst_data["latest_recommendation"] = {
                        "date": str(latest_rec.index[0]),
                        "firm": str(latest_rec["Firm"].iloc[0]),
                        "grade": str(latest_rec["To Grade"].iloc[0]),
                        "action": str(latest_rec["Action"].iloc[0])
                    }
            except:
                pass
            
            return analyst_data
            
        except Exception as e:
            return {"error": f"Analyst data unavailable: {str(e)}"}
    
    def _calculate_technical_indicators(self, stock: yf.Ticker, config: Dict) -> Dict:
        """Calculate basic technical indicators"""
        try:
            hist = stock.history(period=config["period"])
            
            if hist.empty:
                return {"error": "No price data for technical analysis"}
            
            # Simple moving averages
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
            
            current_price = hist['Close'].iloc[-1]
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal
            
            return {
                "moving_averages": {
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50),
                    "sma_200": float(sma_200) if sma_200 is not None else None,
                    "price_vs_sma20": float(((current_price - sma_20) / sma_20) * 100),
                    "price_vs_sma50": float(((current_price - sma_50) / sma_50) * 100)
                },
                "momentum_indicators": {
                    "rsi": float(rsi),
                    "macd": float(macd.iloc[-1]),
                    "macd_signal": float(signal.iloc[-1]),
                    "macd_histogram": float(macd_histogram.iloc[-1])
                },
                "trend_analysis": {
                    "trend_direction": "bullish" if current_price > sma_50 else "bearish",
                    "strength": "strong" if abs((current_price - sma_20) / sma_20) > 0.05 else "moderate"
                }
            }
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def _create_financial_summary(self, data: Dict) -> str:
        """Create a concise financial summary for investors"""
        summary_parts = []
        
        # Company overview
        company_info = data.get("company_info", {})
        summary_parts.append(f"## Financial Analysis: {data['ticker']}")
        summary_parts.append(f"**Company**: {company_info.get('company_name', 'Unknown')}")
        summary_parts.append(f"**Sector**: {company_info.get('sector', 'Unknown')} | **Industry**: {company_info.get('industry', 'Unknown')}")
        
        # Current price and performance
        price_data = data.get("price_data", {})
        if "current_price" in price_data:
            summary_parts.append(f"\n### Current Trading Data")
            summary_parts.append(f"- **Current Price**: ${price_data['current_price']:.2f}")
            summary_parts.append(f"- **Day Change**: {price_data['day_change']:.2f}%")
            summary_parts.append(f"- **YTD Return**: {price_data.get('returns', {}).get('ytd', 0):.2f}%")
            summary_parts.append(f"- **52W Range**: ${price_data['52_week_low']:.2f} - ${price_data['52_week_high']:.2f}")
        
        # Key metrics
        metrics = data.get("key_metrics", {})
        valuation = metrics.get("valuation_metrics", {})
        profitability = metrics.get("profitability_metrics", {})
        
        if valuation:
            summary_parts.append(f"\n### Valuation Metrics")
            summary_parts.append(f"- **P/E Ratio**: {valuation.get('pe_ratio', 0):.1f}")
            summary_parts.append(f"- **P/B Ratio**: {valuation.get('pb_ratio', 0):.1f}")
            summary_parts.append(f"- **PEG Ratio**: {valuation.get('peg_ratio', 0):.1f}")
        
        if profitability:
            summary_parts.append(f"\n### Profitability")
            summary_parts.append(f"- **Profit Margin**: {profitability.get('profit_margin_pct', 0):.1f}%")
            summary_parts.append(f"- **ROE**: {profitability.get('roe_pct', 0):.1f}%")
            summary_parts.append(f"- **ROA**: {profitability.get('roa_pct', 0):.1f}%")
        
        # Financial statements summary
        statements = data.get("financial_statements", {})
        income = statements.get("income_statement", {})
        if income and income.get("total_revenue", 0) > 0:
            summary_parts.append(f"\n### Financial Performance (Latest Year)")
            summary_parts.append(f"- **Revenue**: ${income['total_revenue']:,.0f}")
            summary_parts.append(f"- **Net Income**: ${income['net_income']:,.0f}")
            summary_parts.append(f"- **EPS**: ${income.get('eps', 0):.2f}")
        
        # Analyst sentiment
        analyst = data.get("analyst_data", {})
        if analyst.get("recommendation"):
            summary_parts.append(f"\n### Analyst Outlook")
            summary_parts.append(f"- **Recommendation**: {analyst['recommendation'].upper()}")
            if analyst.get("target_price", 0) > 0:
                summary_parts.append(f"- **Price Target**: ${analyst['target_price']:.2f}")
                upside = ((analyst['target_price'] - price_data.get('current_price', 0)) / price_data.get('current_price', 1)) * 100
                summary_parts.append(f"- **Upside Potential**: {upside:.1f}%")
        
        return "\n".join(summary_parts)
    
    def _assess_data_quality(self, data: Dict) -> Dict:
        """Assess the quality and completeness of fetched data"""
        quality_score = 0
        total_checks = 6
        
        # Check data completeness
        if data.get("company_info") and not "error" in str(data["company_info"]):
            quality_score += 1
        
        if data.get("price_data") and not "error" in str(data["price_data"]):
            quality_score += 1
        
        if data.get("financial_statements") and not "error" in str(data["financial_statements"]):
            quality_score += 1
        
        if data.get("key_metrics") and not "error" in str(data["key_metrics"]):
            quality_score += 1
        
        if data.get("analyst_data") and not "error" in str(data["analyst_data"]):
            quality_score += 1
        
        if data.get("technical_indicators") and not "error" in str(data["technical_indicators"]):
            quality_score += 1
        
        quality_percentage = (quality_score / total_checks) * 100
        
        return {
            "quality_score": quality_score,
            "total_checks": total_checks,
            "quality_percentage": quality_percentage,
            "completeness": "excellent" if quality_percentage >= 90 else 
                           "good" if quality_percentage >= 70 else
                           "fair" if quality_percentage >= 50 else "poor",
            "missing_data_areas": self._identify_missing_data(data)
        }
    
    def _identify_missing_data(self, data: Dict) -> List[str]:
        """Identify which data areas are missing or incomplete"""
        missing = []
        
        for key, section in data.items():
            if isinstance(section, dict) and "error" in str(section):
                missing.append(key.replace("_", " ").title())
        
        return missing
    
    def _arun(self, ticker: str) -> str:
        """Async version (not implemented)"""
        raise NotImplementedError("Async version not implemented")


# Convenience function for standalone usage
def get_financial_data(ticker: str) -> Dict[str, Any]:
    """
    Standalone function to get financial data
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        Dictionary containing financial data
    """
    tool = FinancialDataTool()
    result_str = tool._run(ticker)
    
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse financial data", "raw_result": result_str}


if __name__ == "__main__":
    # Test the tool
    tool = FinancialDataTool()
    
    test_tickers = ["AAPL", "TSLA", "MSFT"]
    
    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Testing Financial Data: {ticker}")
        print(f"{'='*50}")
        
        result = tool._run(ticker)
        print(result[:1000] + "..." if len(result) > 1000 else result)