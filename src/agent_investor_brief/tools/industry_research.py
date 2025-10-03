"""
Industry Research Tool - Real web search using Tavily API
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime

from tavily import TavilyClient
from pydantic import Field


class IndustryResearchTool:
    """Tool for researching industry trends, news, and competitive landscape using Tavily API
    
    Research industry trends, news, and competitive landscape for a company.
    Use this tool to gather:
    - Recent industry news and developments
    - Market trends and outlook
    - Competitive analysis
    - Regulatory changes
    - Industry growth prospects
    
    Input should be a company name or industry sector (e.g., "Apple" or "Electric Vehicles")
    """
    
    def __init__(self):
        # Initialize Tavily client
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        self._tavily_client = TavilyClient(api_key=api_key)
    
    def _run(self, query: str) -> str:
        """Execute industry research for the given query"""
        try:
            results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "research_data": []
            }
            
            # 1. Recent industry news
            news_data = self._search_recent_news(query)
            results["research_data"].extend(news_data)
            
            # 2. Market trends and analysis
            trends_data = self._search_market_trends(query)
            results["research_data"].extend(trends_data)
            
            # 3. Competitive landscape
            competitive_data = self._search_competitive_info(query)
            results["research_data"].extend(competitive_data)
            
            # Format results for LLM consumption
            summary = self._format_research_summary(results)
            
            return json.dumps({
                "research_summary": summary,
                "total_sources": len(results["research_data"]),
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Industry research failed: {str(e)}",
                "query": query,
                "suggestion": "Check your Tavily API key and try a more specific company name or industry term"
            })
    
    def _search_recent_news(self, query: str) -> List[Dict]:
        """Search for recent industry news"""
        try:
            # Search for recent news
            search_query = f"{query} industry news recent developments"
            response = self._tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=5,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "techcrunch.com", "wsj.com"]
            )
            
            news_results = []
            for result in response.get("results", []):
                news_results.append({
                    "type": "news",
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0),
                    "published_date": result.get("published_date", "")
                })
            
            return news_results
            
        except Exception as e:
            return [{"type": "news_error", "error": str(e)}]
    
    def _search_market_trends(self, query: str) -> List[Dict]:
        """Search for market trends and analysis"""
        try:
            # Search for market trends
            search_query = f"{query} market trends analysis growth outlook 2024 2025"
            response = self._tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=4,
                include_domains=["marketwatch.com", "seekingalpha.com", "fool.com", "barrons.com", "morningstar.com"]
            )
            
            trends_results = []
            for result in response.get("results", []):
                trends_results.append({
                    "type": "market_trends",
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })
            
            return trends_results
            
        except Exception as e:
            return [{"type": "trends_error", "error": str(e)}]
    
    def _search_competitive_info(self, query: str) -> List[Dict]:
        """Search for competitive landscape information"""
        try:
            # Search for competitive analysis
            search_query = f"{query} competitors competitive analysis market share"
            response = self._tavily_client.search(
                query=search_query,
                search_depth="basic",
                max_results=3
            )
            
            competitive_results = []
            for result in response.get("results", []):
                competitive_results.append({
                    "type": "competitive",
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })
            
            return competitive_results
            
        except Exception as e:
            return [{"type": "competitive_error", "error": str(e)}]
    
    def _format_research_summary(self, results: Dict) -> str:
        """Format research results into a comprehensive summary"""
        summary_parts = []
        
        # Header
        summary_parts.append(f"# Industry Research Report: {results['query']}")
        summary_parts.append(f"**Research Date:** {results['timestamp']}")
        summary_parts.append(f"**Total Sources:** {len(results['research_data'])}")
        
        # Group results by type
        news_items = [r for r in results['research_data'] if r.get('type') == 'news']
        trends_items = [r for r in results['research_data'] if r.get('type') == 'market_trends']
        competitive_items = [r for r in results['research_data'] if r.get('type') == 'competitive']
        
        # Recent News Section
        if news_items:
            summary_parts.append("\n## Recent Industry News")
            for item in news_items:  # Top news items
                summary_parts.append(f"**{item.get('title', 'N/A')}**")
                summary_parts.append(f"{item.get('content', '')}...")
                summary_parts.append(f"*Source: {item.get('url', 'N/A')}*\n")
        
        # Market Trends Section
        if trends_items:
            summary_parts.append("## Market Trends & Analysis")
            for item in trends_items:  # Top trend items
                summary_parts.append(f"**{item.get('title', 'N/A')}**")
                summary_parts.append(f"{item.get('content', '')}...")
                summary_parts.append(f"*Source: {item.get('url', 'N/A')}*\n")
        
        # Competitive Landscape Section
        if competitive_items:
            summary_parts.append("## Competitive Landscape")
            for item in competitive_items:  # Top competitive items
                summary_parts.append(f"**{item.get('title', 'N/A')}**")
                summary_parts.append(f"{item.get('content', '')}...")
                summary_parts.append(f"*Source: {item.get('url', 'N/A')}*\n")
        
        # Add any errors found
        error_items = [r for r in results['research_data'] if 'error' in r.get('type', '')]
        if error_items:
            summary_parts.append("## Research Notes")
            for error in error_items:
                summary_parts.append(f"- {error.get('type', 'Error')}: {error.get('error', 'Unknown error')}")
        
        return "\n".join(summary_parts)

if __name__ == "__main__":
    # Test the tool
    # Make sure to set TAVILY_API_KEY environment variable
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Please set TAVILY_API_KEY environment variable")
        print("Get your API key from: https://app.tavily.com/")
        exit(1)
    
    tool = IndustryResearchTool()
    
    test_queries = ["Tesla", "Cloud Computing", "Renewable Energy"]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing Industry Research: {query}")
        print(f"{'='*60}")
        
        result = tool._run(query)
        print(result)
        print("\n" + "="*60)