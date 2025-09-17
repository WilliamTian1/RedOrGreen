"""Real-time news and earnings data ingestion from multiple sources.

This module provides functions to fetch actual earnings news and financial headlines
from various free APIs and RSS feeds.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsItem:
    ticker: str
    headline: str
    summary: str
    source: str
    timestamp: datetime
    url: Optional[str] = None


def fetch_yahoo_finance_news(tickers: List[str], max_articles: int = 200) -> List[NewsItem]:
    """Fetch recent news from Yahoo Finance for given tickers."""
    news_items = []
    
    for ticker in tickers[:20]:  # Increased from 10 to 20 tickers
        try:
            # Yahoo Finance news endpoint (unofficial)
            url = f"https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                'q': ticker,
                'lang': 'en-US',
                'region': 'US',
                'quotesCount': 1,
                'newsCount': 10  # Increased from 5 to 10 per ticker
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'news' in data:
                    for item in data['news'][:8]:  # Increased from 3 to 8 per ticker
                        news_items.append(NewsItem(
                            ticker=ticker,
                            headline=item.get('title', ''),
                            summary=item.get('summary', item.get('title', '')),
                            source='Yahoo Finance',
                            timestamp=datetime.fromtimestamp(item.get('providerPublishTime', time.time())),
                            url=item.get('link')
                        ))
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance news for {ticker}: {e}")
            continue
    
    return news_items


def fetch_marketwatch_earnings() -> List[NewsItem]:
    """Scrape recent earnings headlines from MarketWatch."""
    news_items = []
    
    try:
        url = "https://www.marketwatch.com/investing/earnings-calendar"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for earnings-related headlines
            headlines = soup.find_all(['h3', 'h4', 'a'], class_=re.compile(r'headline|title'))
            
            for headline in headlines[:50]:  # Increased from 20 to 50
                text = headline.get_text(strip=True)
                if any(keyword in text.lower() for keyword in ['earnings', 'revenue', 'profit', 'beat', 'miss', 'guidance']):
                    # Try to extract ticker from headline
                    ticker_match = re.search(r'\b([A-Z]{2,5})\b', text)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        news_items.append(NewsItem(
                            ticker=ticker,
                            headline=text,
                            summary=text,
                            source='MarketWatch',
                            timestamp=datetime.now()
                        ))
    
    except Exception as e:
        logger.warning(f"Failed to fetch MarketWatch earnings: {e}")
    
    return news_items


def fetch_sec_filings_rss() -> List[NewsItem]:
    """Fetch recent SEC filings from RSS feed."""
    news_items = []
    
    try:
        # SEC RSS feed for recent filings
        url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom"
        
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            
            entries = soup.find_all('entry')[:30]  # Increased from 15 to 30
            for entry in entries:
                title = entry.find('title')
                summary = entry.find('summary') 
                updated = entry.find('updated')
                
                if title and summary:
                    title_text = title.get_text(strip=True)
                    summary_text = summary.get_text(strip=True)
                    
                    # Extract ticker from filing
                    ticker_match = re.search(r'\(([A-Z]{2,5})\)', title_text)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        
                        # Filter for earnings-related 8-Ks
                        if any(keyword in summary_text.lower() for keyword in ['earnings', 'results', 'financial']):
                            timestamp = datetime.now()
                            if updated:
                                try:
                                    timestamp = datetime.fromisoformat(updated.get_text().replace('Z', '+00:00'))
                                except:
                                    pass
                            
                            news_items.append(NewsItem(
                                ticker=ticker,
                                headline=title_text,
                                summary=summary_text[:200],
                                source='SEC Filings',
                                timestamp=timestamp
                            ))
    
    except Exception as e:
        logger.warning(f"Failed to fetch SEC filings: {e}")
    
    return news_items


def fetch_free_news_api(api_key: Optional[str] = None) -> List[NewsItem]:
    """Fetch financial news using NewsAPI (requires free API key)."""
    news_items = []
    
    if not api_key:
        logger.info("No NewsAPI key provided, skipping NewsAPI fetch")
        return news_items
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'earnings OR "quarterly results" OR "financial results"',
            'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            
            for article in data.get('articles', []):
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Try to extract ticker from title or description
                text = f"{title} {description}"
                ticker_matches = re.findall(r'\b([A-Z]{2,5})\b', text)
                
                for ticker in ticker_matches[:1]:  # Take first match
                    if len(ticker) >= 2:  # Valid ticker length
                        news_items.append(NewsItem(
                            ticker=ticker,
                            headline=title,
                            summary=description or title,
                            source=article.get('source', {}).get('name', 'NewsAPI'),
                            timestamp=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                            url=article.get('url')
                        ))
                        break
    
    except Exception as e:
        logger.warning(f"Failed to fetch NewsAPI data: {e}")
    
    return news_items


def aggregate_live_news(
    tickers: Optional[List[str]] = None,
    max_articles: int = 300,  # Increased from 100 to 300
    newsapi_key: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate news from multiple sources and return as DataFrame."""
    
    if tickers is None:
        # Default to popular earnings-active tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'XOM']
    
    logger.info(f"Fetching live news for {len(tickers)} tickers...")
    
    all_news = []
    
    # Fetch from multiple sources
    try:
        all_news.extend(fetch_yahoo_finance_news(tickers, max_articles//3))
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed: {e}")
    
    try:
        all_news.extend(fetch_marketwatch_earnings())
    except Exception as e:
        logger.warning(f"MarketWatch fetch failed: {e}")
    
    try:
        all_news.extend(fetch_sec_filings_rss())
    except Exception as e:
        logger.warning(f"SEC RSS fetch failed: {e}")
    
    if newsapi_key:
        try:
            all_news.extend(fetch_free_news_api(newsapi_key))
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
    
    # Convert to DataFrame
    if not all_news:
        logger.warning("No news items fetched, falling back to sample data")
        return pd.DataFrame()
    
    # Deduplicate and clean
    seen_headlines = set()
    unique_news = []
    
    for item in all_news:
        headline_key = (item.ticker, item.headline[:50])  # Dedupe key
        if headline_key not in seen_headlines and len(item.headline.strip()) > 10:
            seen_headlines.add(headline_key)
            unique_news.append(item)
    
    # Convert to DataFrame format expected by pipeline
    df_data = []
    for item in unique_news[:max_articles]:
        # Create pipeline-compatible text
        text = f"{item.ticker}: {item.headline}"
        if item.summary and item.summary != item.headline:
            text += f" - {item.summary[:100]}"
        
        df_data.append({
            'ticker': item.ticker,
            'text': text,
            'source': item.source,
            'timestamp': item.timestamp.isoformat(),
            'url': item.url
        })
    
    df = pd.DataFrame(df_data)
    logger.info(f"Collected {len(df)} unique news items from live sources")
    
    return df


def save_live_news_cache(df: pd.DataFrame, cache_path: str = "app/data/live_news_cache.jsonl"):
    """Save fetched news to cache file."""
    if not df.empty:
        df.to_json(cache_path, orient='records', lines=True)
        logger.info(f"Saved {len(df)} news items to {cache_path}")


def load_live_news_cache(cache_path: str = "app/data/live_news_cache.jsonl") -> pd.DataFrame:
    """Load cached news data."""
    try:
        df = pd.read_json(cache_path, lines=True)
        logger.info(f"Loaded {len(df)} cached news items")
        return df
    except Exception as e:
        logger.warning(f"Failed to load news cache: {e}")
        return pd.DataFrame()
