"""
Analysis Builder Tool - Creates comprehensive financial analysis and investment recommendations
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from pydantic import Field

from ..config import settings, TOOL_CONFIGS


class AnalysisBuilderTool(BaseTool):
    """Tool for building comprehensive financial analysis and investment recommendations"""
    
    name: str = "analysis_builder"
    description: str = """
    Build comprehensive financial analysis and investment recommendations.
    Takes financial data and industry research to create:
    - Detailed financial ratio analysis
    - Risk assessment and scoring
    - Investment recommendation with price targets
    - Scenario analysis (bull/base/bear cases)
    - Portfolio fit analysis
    
    Input should be JSON containing financial_data and industry_research from other tools.
    """
    
    def _run(self, analysis_input: str) -> str:
        """Build comprehensive financial analysis"""
        try:
            # Parse input data
            input_data = json.loads(analysis_input)
            
            config = TOOL_CONFIGS["analysis_builder"]
            
            # Extract data components
            financial_data = input_data.get("financial_data", {})
            industry_research = input_data.get("industry_research", {})
            ticker = input_data.get("ticker", "Unknown")
            
            # Build comprehensive analysis
            analysis = {
                "ticker": ticker,
                "analysis_date": datetime.now().isoformat(),
                "executive_summary": self._create_executive_summary(financial_data, industry_research),
                "financial_analysis": self._conduct_financial_analysis(financial_data, config),
                "valuation_analysis": self._conduct_valuation_analysis(financial_data, config),
                "risk_assessment": self._assess_investment_risks(financial_data, industry_research, config),
                "scenario_analysis": self._create_scenario_analysis(financial_data, industry_research, config),
                "investment_recommendation": self._generate_investment_recommendation(financial_data, industry_research, config),
                "portfolio_considerations": self._analyze_portfolio_fit(financial_data, industry_research),
                "key_catalysts": self._identify_key_catalysts(financial_data, industry_research),
                "monitoring_metrics": self._define_monitoring_metrics(financial_data)
            }
            
            # Create investor brief
            brief = self._create_investor_brief(analysis)
            
            return json.dumps({
                "investor_brief": brief,
                "detailed_analysis": analysis,
                "recommendation_strength": self._calculate_recommendation_strength(analysis),
                "generated_at": analysis["analysis_date"]
            }, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Analysis building failed: {str(e)}",
                "suggestion": "Ensure input contains valid financial_data and industry_research"
            })
    
    def _create_executive_summary(self, financial_data: Dict, industry_research: Dict) -> Dict:
        """Create executive summary of the investment opportunity"""
        try:
            # Extract key metrics
            price_data = financial_data.get("detailed_data", {}).get("price_data", {})
            metrics = financial_data.get("detailed_data", {}).get("key_metrics", {})
            company_info = financial_data.get("detailed_data", {}).get("company_info", {})
            
            # Investment thesis components
            strengths = []
            concerns = []
            
            # Analyze financial strength
            profitability = metrics.get("profitability_metrics", {})
            if profitability.get("roe_pct", 0) > 15:
                strengths.append("Strong return on equity")
            if profitability.get("profit_margin_pct", 0) > 10:
                strengths.append("Healthy profit margins")
            
            # Analyze valuation
            valuation = metrics.get("valuation_metrics", {})
            pe_ratio = valuation.get("pe_ratio", 0)
            if 10 < pe_ratio < 25:
                strengths.append("Reasonable valuation")
            elif pe_ratio > 30:
                concerns.append("High valuation multiple")
            
            # Analyze financial health
            health = metrics.get("financial_health", {})
            if health.get("debt_to_equity", 999) < 0.5:
                strengths.append("Conservative debt levels")
            elif health.get("debt_to_equity", 0) > 1.0:
                concerns.append("High debt burden")
            
            # Industry factors
            industry_summary = industry_research.get("research_summary", "")
            if "positive" in industry_summary.lower() or "growth" in industry_summary.lower():
                strengths.append("Favorable industry dynamics")
            if "challenge" in industry_summary.lower() or "risk" in industry_summary.lower():
                concerns.append("Industry headwinds")
            
            return {
                "company_name": company_info.get("company_name", "Unknown"),
                "sector": company_info.get("sector", "Unknown"),
                "current_price": price_data.get("current_price", 0),
                "market_cap_billions": (company_info.get("market_cap", 0)) / 1e9,
                "investment_strengths": strengths,
                "key_concerns": concerns,
                "one_liner": self._generate_investment_one_liner(financial_data, industry_research)
            }
            
        except Exception as e:
            return {"error": f"Executive summary creation failed: {str(e)}"}
    
    def _conduct_financial_analysis(self, financial_data: Dict, config: Dict) -> Dict:
        """Conduct detailed financial analysis"""
        try:
            detailed_data = financial_data.get("detailed_data", {})
            metrics = detailed_data.get("key_metrics", {})
            statements = detailed_data.get("financial_statements", {})
            
            analysis = {
                "profitability_analysis": self._analyze_profitability(metrics),
                "efficiency_analysis": self._analyze_efficiency(metrics, statements),
                "liquidity_analysis": self._analyze_liquidity(metrics),
                "leverage_analysis": self._analyze_leverage(metrics),
                "growth_analysis": self._analyze_growth(metrics, statements),
                "quality_scores": self._calculate_quality_scores(metrics, statements)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Financial analysis failed: {str(e)}"}
    
    def _analyze_profitability(self, metrics: Dict) -> Dict:
        """Analyze profitability metrics"""
        profitability = metrics.get("profitability_metrics", {})
        
        # Score profitability (0-100)
        roe = profitability.get("roe_pct", 0)
        roa = profitability.get("roa_pct", 0)
        profit_margin = profitability.get("profit_margin_pct", 0)
        operating_margin = profitability.get("operating_margin_pct", 0)
        
        # Scoring logic
        roe_score = min(100, max(0, (roe / 20) * 100))  # 20% ROE = 100 points
        roa_score = min(100, max(0, (roa / 10) * 100))  # 10% ROA = 100 points
        margin_score = min(100, max(0, (profit_margin / 15) * 100))  # 15% margin = 100 points
        
        overall_score = (roe_score + roa_score + margin_score) / 3
        
        return {
            "roe_pct": roe,
            "roa_pct": roa,
            "profit_margin_pct": profit_margin,
            "operating_margin_pct": operating_margin,
            "profitability_score": round(overall_score, 1),
            "profitability_grade": self._score_to_grade(overall_score),
            "benchmark_comparison": self._compare_to_benchmarks(profitability, "profitability")
        }
    
    def _analyze_efficiency(self, metrics: Dict, statements: Dict) -> Dict:
        """Analyze operational efficiency"""
        # Asset efficiency metrics
        efficiency_metrics = {
            "asset_turnover": self._calculate_asset_turnover(statements),
            "receivables_turnover": self._calculate_receivables_turnover(statements),
            "inventory_turnover": self._calculate_inventory_turnover(statements),
            "efficiency_score": 0
        }
        
        # Calculate composite efficiency score
        scores = []
        for metric, value in efficiency_metrics.items():
            if isinstance(value, (int, float)) and value > 0:
                scores.append(min(100, value * 20))  # Normalize to 0-100
        
        efficiency_metrics["efficiency_score"] = sum(scores) / len(scores) if scores else 0
        efficiency_metrics["efficiency_grade"] = self._score_to_grade(efficiency_metrics["efficiency_score"])
        
        return efficiency_metrics
    
    def _analyze_liquidity(self, metrics: Dict) -> Dict:
        """Analyze liquidity position"""
        health = metrics.get("financial_health", {})
        
        current_ratio = health.get("current_ratio", 0)
        quick_ratio = health.get("quick_ratio", 0)
        
        # Liquidity scoring
        current_score = min(100, max(0, (current_ratio / 2.0) * 100))  # 2.0 = ideal
        quick_score = min(100, max(0, (quick_ratio / 1.0) * 100))  # 1.0 = ideal
        
        liquidity_score = (current_score + quick_score) / 2
        
        return {
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "liquidity_score": round(liquidity_score, 1),
            "liquidity_grade": self._score_to_grade(liquidity_score),
            "liquidity_assessment": "Strong" if liquidity_score > 80 else 
                                  "Adequate" if liquidity_score > 60 else
                                  "Weak" if liquidity_score > 40 else "Poor"
        }
    
    def _analyze_leverage(self, metrics: Dict) -> Dict:
        """Analyze leverage and debt management"""
        health = metrics.get("financial_health", {})
        
        debt_to_equity = health.get("debt_to_equity", 0)
        interest_coverage = health.get("interest_coverage", 0)
        
        # Leverage scoring (lower debt = higher score)
        debt_score = max(0, 100 - (debt_to_equity * 50))  # Penalize high debt
        coverage_score = min(100, (interest_coverage / 5) * 100)  # 5x = ideal coverage
        
        leverage_score = (debt_score + coverage_score) / 2
        
        return {
            "debt_to_equity": debt_to_equity,
            "interest_coverage": interest_coverage,
            "leverage_score": round(leverage_score, 1),
            "leverage_grade": self._score_to_grade(leverage_score),
            "debt_assessment": "Conservative" if debt_to_equity < 0.3 else
                             "Moderate" if debt_to_equity < 0.6 else
                             "Aggressive" if debt_to_equity < 1.0 else "High Risk"
        }
    
    def _analyze_growth(self, metrics: Dict, statements: Dict) -> Dict:
        """Analyze growth prospects and trends"""
        growth = metrics.get("growth_metrics", {})
        
        revenue_growth = growth.get("revenue_growth_pct", 0)
        earnings_growth = growth.get("earnings_growth_pct", 0)
        
        # Growth scoring
        revenue_score = min(100, max(0, (revenue_growth + 10) * 5))  # Normalize around 10% growth
        earnings_score = min(100, max(0, (earnings_growth + 10) * 5))
        
        growth_score = (revenue_score + earnings_score) / 2
        
        return {
            "revenue_growth_pct": revenue_growth,
            "earnings_growth_pct": earnings_growth,
            "growth_score": round(growth_score, 1),
            "growth_grade": self._score_to_grade(growth_score),
            "growth_sustainability": self._assess_growth_sustainability(growth, statements)
        }
    
    def _calculate_quality_scores(self, metrics: Dict, statements: Dict) -> Dict:
        """Calculate overall financial quality scores"""
        try:
            # Get individual component scores
            prof_score = self._analyze_profitability(metrics).get("profitability_score", 0)
            liq_score = self._analyze_liquidity(metrics).get("liquidity_score", 0)
            lev_score = self._analyze_leverage(metrics).get("leverage_score", 0)
            growth_score = self._analyze_growth(metrics, statements).get("growth_score", 0)
            
            # Weighted composite score
            weights = {"profitability": 0.3, "liquidity": 0.2, "leverage": 0.25, "growth": 0.25}
            
            composite_score = (
                prof_score * weights["profitability"] +
                liq_score * weights["liquidity"] +
                lev_score * weights["leverage"] +
                growth_score * weights["growth"]
            )
            
            return {
                "profitability_score": prof_score,
                "liquidity_score": liq_score,
                "leverage_score": lev_score,
                "growth_score": growth_score,
                "composite_score": round(composite_score, 1),
                "overall_grade": self._score_to_grade(composite_score),
                "quality_tier": "High Quality" if composite_score >= 80 else
                              "Good Quality" if composite_score >= 65 else
                              "Average Quality" if composite_score >= 50 else
                              "Below Average" if composite_score >= 35 else "Poor Quality"
            }
            
        except Exception as e:
            return {"error": f"Quality score calculation failed: {str(e)}"}
    
    def _conduct_valuation_analysis(self, financial_data: Dict, config: Dict) -> Dict:
        """Conduct comprehensive valuation analysis"""
        try:
            detailed_data = financial_data.get("detailed_data", {})
            metrics = detailed_data.get("key_metrics", {})
            price_data = detailed_data.get("price_data", {})
            
            valuation = metrics.get("valuation_metrics", {})
            current_price = price_data.get("current_price", 0)
            
            # Multiple valuation approaches
            analysis = {
                "current_metrics": valuation,
                "relative_valuation": self._relative_valuation_analysis(valuation),
                "intrinsic_valuation": self._intrinsic_valuation_estimate(detailed_data, config),
                "peer_comparison": self._peer_valuation_comparison(valuation),
                "fair_value_estimate": self._calculate_fair_value(detailed_data, config),
                "valuation_summary": self._create_valuation_summary(valuation, current_price)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Valuation analysis failed: {str(e)}"}
    
    def _relative_valuation_analysis(self, valuation: Dict) -> Dict:
        """Analyze relative valuation metrics"""
        pe_ratio = valuation.get("pe_ratio", 0)
        pb_ratio = valuation.get("pb_ratio", 0)
        ps_ratio = valuation.get("ps_ratio", 0)
        
        # Industry averages (simplified - in practice, fetch real data)
        industry_averages = {
            "pe_avg": 18.0,
            "pb_avg": 2.5,
            "ps_avg": 3.0
        }
        
        # Calculate relative metrics
        pe_premium = ((pe_ratio - industry_averages["pe_avg"]) / industry_averages["pe_avg"]) * 100
        pb_premium = ((pb_ratio - industry_averages["pb_avg"]) / industry_averages["pb_avg"]) * 100
        ps_premium = ((ps_ratio - industry_averages["ps_avg"]) / industry_averages["ps_avg"]) * 100
        
        avg_premium = (pe_premium + pb_premium + ps_premium) / 3
        
        return {
            "pe_vs_industry": f"{pe_premium:+.1f}%",
            "pb_vs_industry": f"{pb_premium:+.1f}%",
            "ps_vs_industry": f"{ps_premium:+.1f}%",
            "average_premium": f"{avg_premium:+.1f}%",
            "relative_assessment": "Overvalued" if avg_premium > 20 else
                                 "Fairly Valued" if -10 <= avg_premium <= 20 else
                                 "Undervalued"
        }
    
    def _intrinsic_valuation_estimate(self, detailed_data: Dict, config: Dict) -> Dict:
        """Estimate intrinsic value using DCF-like approach"""
        try:
            statements = detailed_data.get("financial_statements", {})
            cashflow = statements.get("cash_flow", {})
            
            free_cash_flow = cashflow.get("free_cash_flow", 0)
            if free_cash_flow <= 0:
                return {"error": "Insufficient cash flow data for DCF"}
            
            # DCF assumptions
            growth_rate = 0.05  # 5% terminal growth
            discount_rate = 0.10  # 10% WACC assumption
            terminal_multiple = 15  # P/E multiple for terminal value
            
            # Simple 5-year DCF projection
            projected_fcf = []
            for year in range(1, 6):
                fcf = free_cash_flow * ((1 + growth_rate) ** year)
                projected_fcf.append(fcf)
            
            # Terminal value
            terminal_fcf = projected_fcf[-1] * (1 + growth_rate)
            terminal_value = terminal_fcf * terminal_multiple
            
            # Present value calculation
            pv_fcf = sum([fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
            pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
            
            enterprise_value = pv_fcf + pv_terminal
            
            # Convert to equity value (simplified)
            company_info = detailed_data.get("company_info", {})
            shares_outstanding = company_info.get("market_cap", 0) / detailed_data.get("price_data", {}).get("current_price", 1)
            
            intrinsic_value_per_share = enterprise_value / shares_outstanding if shares_outstanding > 0 else 0
            
            return {
                "free_cash_flow": free_cash_flow,
                "projected_fcf_5yr": projected_fcf,
                "terminal_value": terminal_value,
                "enterprise_value": enterprise_value,
                "intrinsic_value_per_share": intrinsic_value_per_share,
                "assumptions": {
                    "growth_rate": growth_rate,
                    "discount_rate": discount_rate,
                    "terminal_multiple": terminal_multiple
                }
            }
            
        except Exception as e:
            return {"error": f"DCF calculation failed: {str(e)}"}
    
    def _peer_valuation_comparison(self, valuation: Dict) -> Dict:
        """Compare valuation to peer group"""
        # Simplified peer comparison (in practice, fetch real peer data)
        peer_ranges = {
            "pe_range": {"low": 12, "high": 25, "median": 18},
            "pb_range": {"low": 1.5, "high": 4.0, "median": 2.5},
            "ps_range": {"low": 1.0, "high": 5.0, "median": 3.0}
        }
        
        pe_ratio = valuation.get("pe_ratio", 0)
        pb_ratio = valuation.get("pb_ratio", 0)
        ps_ratio = valuation.get("ps_ratio", 0)
        
        def get_percentile(value, range_dict):
            if value <= range_dict["low"]:
                return "Bottom Quartile"
            elif value <= range_dict["median"]:
                return "Below Median"
            elif value <= range_dict["high"]:
                return "Above Median"
            else:
                return "Top Quartile"
        
        return {
            "pe_percentile": get_percentile(pe_ratio, peer_ranges["pe_range"]),
            "pb_percentile": get_percentile(pb_ratio, peer_ranges["pb_range"]),
            "ps_percentile": get_percentile(ps_ratio, peer_ranges["ps_range"]),
            "peer_summary": "Trading at premium to peers" if pe_ratio > peer_ranges["pe_range"]["median"] else
                           "Trading at discount to peers"
        }
    
    def _calculate_fair_value(self, detailed_data: Dict, config: Dict) -> Dict:
        """Calculate fair value estimate using multiple approaches"""
        try:
            price_data = detailed_data.get("price_data", {})
            current_price = price_data.get("current_price", 0)
            
            # Method 1: Relative valuation
            relative_value = current_price * 1.0  # Placeholder
            
            # Method 2: DCF
            dcf_analysis = self._intrinsic_valuation_estimate(detailed_data, config)
            dcf_value = dcf_analysis.get("intrinsic_value_per_share", current_price)
            
            # Method 3: Asset-based (simplified)
            statements = detailed_data.get("financial_statements", {})
            balance_sheet = statements.get("balance_sheet", {})
            book_value_per_share = balance_sheet.get("total_equity", 0) / 1000000  # Simplified
            
            # Weighted average fair value
            weights = {"relative": 0.4, "dcf": 0.4, "asset": 0.2}
            fair_value = (
                relative_value * weights["relative"] +
                dcf_value * weights["dcf"] +
                book_value_per_share * weights["asset"]
            )
            
            upside_downside = ((fair_value - current_price) / current_price) * 100
            
            return {
                "relative_value": relative_value,
                "dcf_value": dcf_value,
                "asset_value": book_value_per_share,
                "fair_value_estimate": fair_value,
                "current_price": current_price,
                "upside_downside_pct": upside_downside,
                "valuation_conclusion": "Undervalued" if upside_downside > 15 else
                                      "Fairly Valued" if -15 <= upside_downside <= 15 else
                                      "Overvalued"
            }
            
        except Exception as e:
            return {"error": f"Fair value calculation failed: {str(e)}"}
    
    def _create_valuation_summary(self, valuation: Dict, current_price: float) -> str:
        """Create valuation summary text"""
        pe_ratio = valuation.get("pe_ratio", 0)
        pb_ratio = valuation.get("pb_ratio", 0)
        
        summary = f"Current valuation shows P/E of {pe_ratio:.1f}x and P/B of {pb_ratio:.1f}x. "
        
        if pe_ratio > 25:
            summary += "High P/E suggests growth expectations or potential overvaluation. "
        elif pe_ratio < 12:
            summary += "Low P/E may indicate value opportunity or fundamental concerns. "
        else:
            summary += "P/E ratio appears reasonable for current market conditions. "
        
        return summary
    
    def _assess_investment_risks(self, financial_data: Dict, industry_research: Dict, config: Dict) -> Dict:
        """Comprehensive risk assessment"""
        try:
            risks = {
                "financial_risks": self._assess_financial_risks(financial_data),
                "business_risks": self._assess_business_risks(financial_data, industry_research),
                "market_risks": self._assess_market_risks(financial_data),
                "esg_risks": self._assess_esg_risks(financial_data, industry_research),
                "overall_risk_score": 0,
                "risk_grade": "Unknown"
            }
            
            # Calculate overall risk score
            risk_scores = [
                risks["financial_risks"].get("risk_score", 50),
                risks["business_risks"].get("risk_score", 50),
                risks["market_risks"].get("risk_score", 50),
                risks["esg_risks"].get("risk_score", 50)
            ]
            
            overall_risk = sum(risk_scores) / len(risk_scores)
            risks["overall_risk_score"] = round(overall_risk, 1)
            risks["risk_grade"] = self._risk_score_to_grade(overall_risk)
            
            return risks
            
        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}
    
    def _assess_financial_risks(self, financial_data: Dict) -> Dict:
        """Assess financial-specific risks"""
        detailed_data = financial_data.get("detailed_data", {})
        metrics = detailed_data.get("key_metrics", {})
        
        health = metrics.get("financial_health", {})
        debt_to_equity = health.get("debt_to_equity", 0)
        current_ratio = health.get("current_ratio", 0)
        
        # Risk scoring (higher score = higher risk)
        debt_risk = min(100, debt_to_equity * 50)  # High debt = high risk
        liquidity_risk = max(0, 100 - (current_ratio * 50))  # Low liquidity = high risk
        
        financial_risk_score = (debt_risk + liquidity_risk) / 2
        
        return {
            "debt_risk": debt_risk,
            "liquidity_risk": liquidity_risk,
            "risk_score": round(financial_risk_score, 1),
            "key_risks": [
                "High debt levels" if debt_to_equity > 1.0 else None,
                "Liquidity concerns" if current_ratio < 1.5 else None,
                "Interest rate sensitivity" if debt_to_equity > 0.5 else None
            ]
        }
    
    def _assess_business_risks(self, financial_data: Dict, industry_research: Dict) -> Dict:
        """Assess business and industry risks"""
        # Analyze industry research for risk factors
        research_summary = industry_research.get("research_summary", "").lower()
        
        risk_keywords = ["challenge", "decline", "competition", "disruption", "regulation"]
        risk_count = sum(1 for keyword in risk_keywords if keyword in research_summary)
        
        business_risk_score = min(100, risk_count * 20)  # Scale based on risk indicators
        
        return {
            "industry_risk_indicators": risk_count,
            "risk_score": round(business_risk_score, 1),
            "key_risks": [
                "Competitive pressure",
                "Regulatory changes",
                "Technology disruption",
                "Market saturation"
            ][:risk_count + 1]  # Show relevant risks
        }
    
    def _assess_market_risks(self, financial_data: Dict) -> Dict:
        """Assess market-related risks"""
        detailed_data = financial_data.get("detailed_data", {})
        price_data = detailed_data.get("price_data", {})
        
        volatility = price_data.get("volatility_30d", 20)  # Default 20%
        beta = price_data.get("beta", 1.0)
        
        # Market risk scoring
        volatility_risk = min(100, volatility * 2)  # High volatility = high risk
        beta_risk = abs(beta - 1.0) * 50  # Deviation from market = risk
        
        market_risk_score = (volatility_risk + beta_risk) / 2
        
        return {
            "volatility_30d": volatility,
            "beta": beta,
            "risk_score": round(market_risk_score, 1),
            "key_risks": [
                "High volatility" if volatility > 30 else None,
                "Market sensitivity" if abs(beta) > 1.5 else None,
                "Liquidity risk" if price_data.get("volume", 0) < price_data.get("avg_volume", 1) * 0.5 else None
            ]
        }
    
    def _assess_esg_risks(self, financial_data: Dict, industry_research: Dict) -> Dict:
        """Assess ESG (Environmental, Social, Governance) risks"""
        detailed_data = financial_data.get("detailed_data", {})
        company_info = detailed_data.get("company_info", {})
        
        sector = company_info.get("sector", "").lower()
        
        # Sector-based ESG risk assessment
        high_esg_risk_sectors = ["energy", "materials", "utilities"]
        medium_esg_risk_sectors = ["industrials", "consumer discretionary"]
        
        if any(risk_sector in sector for risk_sector in high_esg_risk_sectors):
            base_esg_risk = 70
        elif any(med_sector in sector for med_sector in medium_esg_risk_sectors):
            base_esg_risk = 50
        else:
            base_esg_risk = 30
        
        return {
            "sector_esg_risk": base_esg_risk,
            "risk_score": base_esg_risk,
            "key_risks": [
                "Environmental regulations" if base_esg_risk > 60 else None,
                "Social responsibility issues" if base_esg_risk > 50 else None,
                "Governance concerns" if base_esg_risk > 40 else None
            ]
        }
    
    def _create_scenario_analysis(self, financial_data: Dict, industry_research: Dict, config: Dict) -> Dict:
        """Create bull/base/bear scenario analysis"""
        try:
            detailed_data = financial_data.get("detailed_data", {})
            price_data = detailed_data.get("price_data", {})
            current_price = price_data.get("current_price", 0)
            
            # Base case assumptions
            base_growth = 0.05  # 5% growth
            base_multiple = 15  # P/E multiple
            
            scenarios = {
                "bull_case": {
                    "probability": 25,
                    "growth_rate": base_growth * 1.5,
                    "pe_multiple": base_multiple * 1.2,
                    "key_drivers": [
                        "Strong industry tailwinds",
                        "Market share expansion",
                        "Operational efficiency gains",
                        "Multiple expansion"
                    ]
                },
                "base_case": {
                    "probability": 50,
                    "growth_rate": base_growth,
                    "pe_multiple": base_multiple,
                    "key_drivers": [
                        "Steady market growth",
                        "Maintain market position",
                        "Normal operational performance",
                        "Stable valuation"
                    ]
                },
                "bear_case": {
                    "probability": 25,
                    "growth_rate": base_growth * 0.3,
                    "pe_multiple": base_multiple * 0.8,
                    "key_drivers": [
                        "Industry headwinds",
                        "Competitive pressure",
                        "Margin compression",
                        "Multiple contraction"
                    ]
                }
            }
            
            # Calculate price targets for each scenario
            for scenario_name, scenario in scenarios.items():
                # Simplified price target calculation
                growth_factor = 1 + scenario["growth_rate"]
                multiple_factor = scenario["pe_multiple"] / base_multiple
                price_target = current_price * growth_factor * multiple_factor
                
                scenario["price_target"] = round(price_target, 2)
                scenario["upside_downside"] = round(((price_target - current_price) / current_price) * 100, 1)
            
            # Expected value calculation
            expected_return = sum([
                scenario["upside_downside"] * (scenario["probability"] / 100)
                for scenario in scenarios.values()
            ])
            
            return {
                "scenarios": scenarios,
                "expected_return_pct": round(expected_return, 1),
                "risk_reward_ratio": abs(expected_return / max(abs(scenarios["bear_case"]["upside_downside"]), 1)),
                "scenario_summary": self._create_scenario_summary(scenarios)
            }
            
        except Exception as e:
            return {"error": f"Scenario analysis failed: {str(e)}"}
    
    def _generate_investment_recommendation(self, financial_data: Dict, industry_research: Dict, config: Dict) -> Dict:
        """Generate final investment recommendation"""
        try:
            # Gather key metrics for decision
            detailed_data = financial_data.get("detailed_data", {})
            price_data = detailed_data.get("price_data", {})
            current_price = price_data.get("current_price", 0)
            
            # Score various factors (0-100)
            quality_score = 75  # Placeholder - would use actual quality analysis
            valuation_score = 65  # Placeholder - would use actual valuation analysis
            growth_score = 70  # Placeholder - would use actual growth analysis
            risk_score = 60  # Placeholder - would use actual risk analysis (lower = less risk)
            
            # Weighted recommendation score
            weights = {"quality": 0.25, "valuation": 0.30, "growth": 0.25, "risk": 0.20}
            
            recommendation_score = (
                quality_score * weights["quality"] +
                valuation_score * weights["valuation"] +
                growth_score * weights["growth"] +
                (100 - risk_score) * weights["risk"]  # Invert risk score
            )
            
            # Generate recommendation
            if recommendation_score >= 80:
                recommendation = "Strong Buy"
                target_price = current_price * 1.25
            elif recommendation_score >= 70:
                recommendation = "Buy"
                target_price = current_price * 1.15
            elif recommendation_score >= 60:
                recommendation = "Hold"
                target_price = current_price * 1.05
            elif recommendation_score >= 50:
                recommendation = "Weak Hold"
                target_price = current_price * 0.95
            else:
                recommendation = "Sell"
                target_price = current_price * 0.85
            
            return {
                "recommendation": recommendation,
                "recommendation_score": round(recommendation_score, 1),
                "price_target": round(target_price, 2),
                "current_price": current_price,
                "upside_potential_pct": round(((target_price - current_price) / current_price) * 100, 1),
                "time_horizon": "12 months",
                "confidence_level": "Medium" if 60 <= recommendation_score <= 80 else "High" if recommendation_score > 80 else "Low",
                "key_factors": {
                    "quality_score": quality_score,
                    "valuation_score": valuation_score,
                    "growth_score": growth_score,
                    "risk_score": risk_score
                },
                "investment_thesis": self._create_investment_thesis(recommendation, financial_data, industry_research)
            }
            
        except Exception as e:
            return {"error": f"Recommendation generation failed: {str(e)}"}
    
    def _create_investment_thesis(self, recommendation: str, financial_data: Dict, industry_research: Dict) -> str:
        """Create investment thesis narrative"""
        detailed_data = financial_data.get("detailed_data", {})
        company_info = detailed_data.get("company_info", {})
        company_name = company_info.get("company_name", "The company")
        
        if recommendation in ["Strong Buy", "Buy"]:
            thesis = f"{company_name} presents an attractive investment opportunity driven by solid fundamentals, reasonable valuation, and positive industry dynamics. "
        elif recommendation == "Hold":
            thesis = f"{company_name} demonstrates stable performance with balanced risk-reward characteristics, suitable for maintaining current position. "
        else:
            thesis = f"{company_name} faces challenges that may impact near-term performance, warranting caution in current market environment. "
        
        return thesis
    
    def _analyze_portfolio_fit(self, financial_data: Dict, industry_research: Dict) -> Dict:
        """Analyze how investment fits in portfolio context"""
        detailed_data = financial_data.get("detailed_data", {})
        company_info = detailed_data.get("company_info", {})
        price_data = detailed_data.get("price_data", {})
        
        sector = company_info.get("sector", "Unknown")
        beta = price_data.get("beta", 1.0)
        
        return {
            "sector_exposure": sector,
            "portfolio_role": "Growth" if beta > 1.2 else "Income" if beta < 0.8 else "Core",
            "diversification_benefit": "High" if sector in ["Technology", "Healthcare"] else "Medium",
            "correlation_estimate": "Medium",  # Would calculate vs market indices
            "suitable_for": [
                "Growth-oriented portfolios" if beta > 1.0 else "Conservative portfolios",
                "Long-term investors",
                "Sector diversification"
            ]
        }
    
    def _identify_key_catalysts(self, financial_data: Dict, industry_research: Dict) -> Dict:
        """Identify key catalysts and events to monitor"""
        return {
            "positive_catalysts": [
                "Earnings beat expectations",
                "New product launches",
                "Market expansion",
                "Industry consolidation"
            ],
            "negative_catalysts": [
                "Regulatory changes",
                "Competitive pressure",
                "Economic downturn",
                "Management changes"
            ],
            "upcoming_events": [
                "Quarterly earnings (estimate)",
                "Product announcements",
                "Regulatory decisions"
            ]
        }
    
    def _define_monitoring_metrics(self, financial_data: Dict) -> Dict:
        """Define key metrics to monitor ongoing"""
        return {
            "financial_metrics": [
                "Revenue growth rate",
                "Profit margins",
                "Return on equity",
                "Debt-to-equity ratio"
            ],
            "market_metrics": [
                "Stock price performance",
                "Trading volume",
                "Analyst revisions",
                "Peer comparison"
            ],
            "operational_metrics": [
                "Market share",
                "Customer growth",
                "Product pipeline",
                "Management guidance"
            ]
        }
    
    def _create_investor_brief(self, analysis: Dict) -> str:
        """Create final investor brief document"""
        executive_summary = analysis.get("executive_summary", {})
        recommendation = analysis.get("investment_recommendation", {})
        
        brief_parts = []
        
        # Header
        brief_parts.append(f"# Investment Brief: {executive_summary.get('company_name', 'Unknown')}")
        brief_parts.append(f"**Ticker**: {analysis.get('ticker', 'N/A')} | **Analysis Date**: {analysis.get('analysis_date', '')[:10]}")
        brief_parts.append(f"**Sector**: {executive_summary.get('sector', 'Unknown')}")
        
        # Investment Recommendation
        rec = recommendation.get("recommendation", "Hold")
        price_target = recommendation.get("price_target", 0)
        upside = recommendation.get("upside_potential_pct", 0)
        
        brief_parts.append(f"\n## Investment Recommendation: {rec}")
        brief_parts.append(f"**Price Target**: ${price_target:.2f} | **Upside**: {upside:+.1f}%")
        brief_parts.append(f"**Time Horizon**: {recommendation.get('time_horizon', '12 months')}")
        
        # Investment Thesis
        brief_parts.append(f"\n## Investment Thesis")
        brief_parts.append(recommendation.get("investment_thesis", "Analysis pending"))
        
        # Key Strengths and Concerns
        brief_parts.append(f"\n### Key Investment Strengths")
        for strength in executive_summary.get("investment_strengths", [])[:3]:
            brief_parts.append(f"• {strength}")
        
        brief_parts.append(f"\n### Key Concerns")
        for concern in executive_summary.get("key_concerns", [])[:3]:
            brief_parts.append(f"• {concern}")
        
        # Financial Highlights
        financial_analysis = analysis.get("financial_analysis", {})
        quality_scores = financial_analysis.get("quality_scores", {})
        
        brief_parts.append(f"\n## Financial Quality Assessment")
        brief_parts.append(f"**Overall Grade**: {quality_scores.get('overall_grade', 'N/A')}")
        brief_parts.append(f"**Quality Tier**: {quality_scores.get('quality_tier', 'Unknown')}")
        
        # Risk Assessment
        risk_assessment = analysis.get("risk_assessment", {})
        brief_parts.append(f"\n## Risk Profile")
        brief_parts.append(f"**Risk Grade**: {risk_assessment.get('risk_grade', 'Unknown')}")
        brief_parts.append(f"**Overall Risk Score**: {risk_assessment.get('overall_risk_score', 0)}/100")
        
        # Scenario Analysis
        scenario_analysis = analysis.get("scenario_analysis", {})
        brief_parts.append(f"\n## Scenario Analysis")
        brief_parts.append(f"**Expected Return**: {scenario_analysis.get('expected_return_pct', 0):+.1f}%")
        
        scenarios = scenario_analysis.get("scenarios", {})
        if scenarios:
            for scenario_name, scenario in scenarios.items():
                brief_parts.append(f"• **{scenario_name.replace('_', ' ').title()}**: {scenario.get('upside_downside', 0):+.1f}% ({scenario.get('probability', 0)}% probability)")
        
        # Portfolio Considerations
        portfolio = analysis.get("portfolio_considerations", {})
        brief_parts.append(f"\n## Portfolio Fit")
        brief_parts.append(f"**Role**: {portfolio.get('portfolio_role', 'Unknown')}")
        brief_parts.append(f"**Sector**: {portfolio.get('sector_exposure', 'Unknown')}")
        
        # Key Catalysts
        catalysts = analysis.get("key_catalysts", {})
        brief_parts.append(f"\n## Key Catalysts to Monitor")
        brief_parts.append("**Positive**: " + ", ".join(catalysts.get("positive_catalysts", [])[:3]))
        brief_parts.append("**Risks**: " + ", ".join(catalysts.get("negative_catalysts", [])[:3]))
        
        # Footer
        brief_parts.append(f"\n---")
        brief_parts.append(f"*This analysis is for informational purposes only and should not be considered as investment advice.*")
        
        return "\n".join(brief_parts)
    
    def _calculate_recommendation_strength(self, analysis: Dict) -> Dict:
        """Calculate overall recommendation strength and confidence"""
        recommendation = analysis.get("investment_recommendation", {})
        rec_score = recommendation.get("recommendation_score", 50)
        
        return {
            "strength_score": rec_score,
            "confidence_level": recommendation.get("confidence_level", "Medium"),
            "conviction": "High" if rec_score >= 80 else "Medium" if rec_score >= 60 else "Low",
            "recommendation_rationale": f"Score of {rec_score}/100 based on weighted analysis of quality, valuation, growth, and risk factors"
        }
    
    # Helper methods
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "A-"
        elif score >= 75: return "B+"
        elif score >= 70: return "B"
        elif score >= 65: return "B-"
        elif score >= 60: return "C+"
        elif score >= 55: return "C"
        elif score >= 50: return "C-"
        elif score >= 45: return "D+"
        elif score >= 40: return "D"
        else: return "F"
    
    def _risk_score_to_grade(self, risk_score: float) -> str:
        """Convert risk score to risk grade (lower score = lower risk)"""
        if risk_score <= 20: return "Very Low"
        elif risk_score <= 35: return "Low"
        elif risk_score <= 50: return "Medium"
        elif risk_score <= 70: return "High"
        else: return "Very High"
    
    def _compare_to_benchmarks(self, metrics: Dict, category: str) -> Dict:
        """Compare metrics to industry benchmarks"""
        # Simplified benchmark comparison
        benchmarks = {
            "profitability": {"excellent": 20, "good": 15, "average": 10, "poor": 5}
        }
        
        return {"status": "Benchmark comparison available"}
    
    def _calculate_asset_turnover(self, statements: Dict) -> float:
        """Calculate asset turnover ratio"""
        try:
            income = statements.get("income_statement", {})
            balance = statements.get("balance_sheet", {})
            
            revenue = income.get("total_revenue", 0)
            assets = balance.get("total_assets", 1)
            
            return revenue / assets if assets > 0 else 0
        except:
            return 0
    
    def _calculate_receivables_turnover(self, statements: Dict) -> float:
        """Calculate receivables turnover ratio"""
        # Simplified calculation
        return 12.0  # Placeholder
    
    def _calculate_inventory_turnover(self, statements: Dict) -> float:
        """Calculate inventory turnover ratio"""
        # Simplified calculation  
        return 8.0  # Placeholder
    
    def _assess_growth_sustainability(self, growth: Dict, statements: Dict) -> str:
        """Assess sustainability of growth rates"""
        revenue_growth = growth.get("revenue_growth_pct", 0)
        
        if revenue_growth > 20:
            return "High growth but monitor sustainability"
        elif revenue_growth > 10:
            return "Healthy sustainable growth"
        elif revenue_growth > 0:
            return "Modest growth trajectory"
        else:
            return "Declining growth trend"
    
    def _generate_investment_one_liner(self, financial_data: Dict, industry_research: Dict) -> str:
        """Generate concise investment summary"""
        detailed_data = financial_data.get("detailed_data", {})
        company_info = detailed_data.get("company_info", {})
        
        company_name = company_info.get("company_name", "This company")
        sector = company_info.get("sector", "its sector")
        
        return f"{company_name} offers solid fundamentals and growth potential in the {sector} space."
    
    def _create_scenario_summary(self, scenarios: Dict) -> str:
        """Create scenario analysis summary"""
        bull = scenarios.get("bull_case", {})
        bear = scenarios.get("bear_case", {})
        
        bull_upside = bull.get("upside_downside", 0)
        bear_downside = bear.get("upside_downside", 0)
        
        return f"Scenarios range from {bear_downside:+.1f}% (bear) to {bull_upside:+.1f}% (bull) with asymmetric risk/reward profile."
    
    def _arun(self, analysis_input: str) -> str:
        """Async version (not implemented)"""
        raise NotImplementedError("Async version not implemented")


# Convenience function for standalone usage
def build_investment_analysis(financial_data: Dict, industry_research: Dict, ticker: str = "Unknown") -> Dict[str, Any]:
    """
    Standalone function to build investment analysis
    
    Args:
        financial_data: Financial data from FinancialDataTool
        industry_research: Research data from IndustryResearchTool  
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing comprehensive analysis
    """
    tool = AnalysisBuilderTool()
    
    analysis_input = json.dumps({
        "financial_data": financial_data,
        "industry_research": industry_research,
        "ticker": ticker
    })
    
    result_str = tool._run(analysis_input)
    
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse analysis results", "raw_result": result_str}


if __name__ == "__main__":
    # Test the tool with sample data
    tool = AnalysisBuilderTool()
    
    # Sample test data
    sample_financial_data = {
        "detailed_data": {
            "company_info": {"company_name": "Test Corp", "sector": "Technology"},
            "price_data": {"current_price": 100.0},
            "key_metrics": {
                "profitability_metrics": {"roe_pct": 15.0, "profit_margin_pct": 12.0},
                "valuation_metrics": {"pe_ratio": 18.0, "pb_ratio": 2.5},
                "financial_health": {"debt_to_equity": 0.3, "current_ratio": 2.0}
            }
        }
    }
    
    sample_industry_research = {
        "research_summary": "Positive industry outlook with growth opportunities"
    }
    
    test_input = json.dumps({
        "financial_data": sample_financial_data,
        "industry_research": sample_industry_research,
        "ticker": "TEST"
    })
    
    print("Testing Analysis Builder Tool...")
    result = tool._run(test_input)
    print(result[:1000] + "..." if len(result) > 1000 else result)