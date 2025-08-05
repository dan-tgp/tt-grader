#!/usr/bin/env python3
"""
Real-Time SEO Domain Scraper + LLM Grader
Features:
- Scrapes domains with 48 concurrent workers
- Real-time LLM grading as results come in
- GPT-4o Mini with STRUCTURED JSON output for reliability
- Live streaming results to CSV
- Simple rubric: FAIL (1-3), NEEDS IMPROVEMENT (4-7), GOOD (8-10)
- Clean simplified output: Just core statistics, quick reference, and character lengths
"""

import asyncio
import aiohttp
import requests
import json
import csv
import time
import os
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from urllib.parse import urlparse
import openai
from openai import AsyncOpenAI

# Import the clean output formatter
from output_formatter import display_enhanced_statistics

# Read API keys from .env file
def load_env():
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Support both = and : formats
                    if '=' in line:
                        key, value = line.split('=', 1)
                    elif ':' in line:
                        key, value = line.split(':', 1)
                    else:
                        continue
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("⚠️ .env file not found. Please create one with your API keys.")
    return env_vars

# Load environment variables
env_vars = load_env()
SCRAPINGBEE_API_KEY = env_vars.get('SCRAPINGBEE_API_KEY')
FIRECRAWL_API_KEY = env_vars.get('FIRECRAWL_KEY')
OPENAI_API_KEY = env_vars.get('OPENAI_KEY')

if not SCRAPINGBEE_API_KEY:
    print("❌ SCRAPINGBEE_API_KEY not found in .env file")
    exit(1)

if not FIRECRAWL_API_KEY:
    print("❌ FIRECRAWL_KEY not found in .env file")
    exit(1)

if not OPENAI_API_KEY:
    print("❌ OPENAI_KEY not found in .env file")
    exit(1)

# Configuration
MAX_CONCURRENT = 48  # ScrapingBee concurrent requests
MAX_LLM_CONCURRENT = 200  # OpenAI concurrent requests (optimized for batching)
BATCH_SIZE = 25  # Items per batch for maximum speed
MAX_URLS_TO_MAP = 10000  # Maximum URLs to discover via Firecrawl
SCRAPINGBEE_BASE_URL = "https://app.scrapingbee.com/api/v1/"
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"

@dataclass
class PageResult:
    """Data structure for page scraping and grading results"""
    url: str = ""
    title: str = ""
    meta_description: str = ""
    h1_text: str = ""
    title_score: int = 0
    description_score: int = 0
    h1_score: int = 0
    overall_score: int = 0
    explanation: str = ""
    status: str = "pending"
    error: str = ""
    graded: bool = False

class RealTimeCSVWriter:
    """Write results to CSV in real-time as they come in"""
    def __init__(self, filename: str):
        self.filename = filename
        self.headers_written = False
    
    def write_result(self, result: PageResult):
        """Write a single result to CSV"""
        file_exists = os.path.exists(self.filename)
        
        with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['url', 'title', 'meta_description', 'h1_text', 'title_score', 
                         'description_score', 'h1_score', 'overall_score', 'explanation', 'status', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists or not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            
            writer.writerow({
                'url': result.url,
                'title': result.title,
                'meta_description': result.meta_description,
                'h1_text': result.h1_text,
                'title_score': result.title_score,
                'description_score': result.description_score,
                'h1_score': result.h1_score,
                'overall_score': result.overall_score,
                'explanation': result.explanation,
                'status': result.status,
                'error': result.error
            })

def create_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')

def get_domain_urls(domain: str, target_count: int, search_param: str = None) -> List[str]:
    """Get URLs from domain using Firecrawl /map endpoint with optional search filtering"""
    
    # Credit-efficient mapping limits based on target count
    if target_count <= 10:
        map_limit = 300
    elif target_count <= 50:
        map_limit = 800  
    elif target_count <= 100:
        map_limit = 1500
    elif target_count <= 500:
        map_limit = 3000
    else:
        map_limit = MAX_URLS_TO_MAP
    
    print(f"📊 Map limit: {map_limit} URLs (💰 CREDIT-OPTIMIZED for target: {target_count})")
    print(f"🔧 Max possible: {MAX_URLS_TO_MAP} URLs (increase MAX_URLS_TO_MAP if needed)")
    
    if search_param:
        print(f"🔍 Search filter: '{search_param}'")
    
    return get_urls_with_map(domain, map_limit, search_param)

def get_urls_with_map(domain: str, map_limit: int, search_param: str = None) -> List[str]:
    """Use Firecrawl /map endpoint for URL discovery with optional search filtering"""
    clean_domain = domain.replace('http://', '').replace('https://', '').replace('www.', '').strip('/')
    
    final_urls = []
    best_count = 0
    
    # Try different URL variations for better discovery
    url_variations = [
        f"https://www.{clean_domain}",
        f"https://{clean_domain}",
        f"https://shop.{clean_domain}",  # For e-commerce sites
        f"https://store.{clean_domain}"  # Alternative for e-commerce
    ]
    
    headers = {
        'Authorization': f'Bearer {FIRECRAWL_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    for base_url in url_variations:
        try:
            print(f"🔍 Mapping: {base_url}")
            
            data = {
                'url': base_url,
                'limit': map_limit,
                'includeSubdomains': True,
                'ignoreSitemap': False  # Use sitemap for comprehensive discovery
            }
            
            # Add search parameter if provided
            if search_param:
                data['search'] = search_param
            
            response = requests.post(f'{FIRECRAWL_BASE_URL}/map', headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                urls = result.get('links', [])
                
                print(f"  ✅ Found {len(urls)} URLs from {base_url}")
                if len(urls) > 0:
                    sample_urls = urls[:5] if len(urls) >= 5 else urls
                    print(f"  📋 Sample URLs: {sample_urls}")
                
                if len(urls) > best_count:
                    final_urls = urls
                    best_count = len(urls)
                    print(f"  🎯 New best result: {len(urls)} URLs")
                    
                    if len(urls) >= map_limit * 0.8:  # Got close to limit
                        print(f"  🚀 Excellent discovery! Got {len(urls)} URLs (near limit)")
                        break
                        
            else:
                print(f"  ❌ Failed: HTTP {response.status_code}")
                if response.status_code == 402:
                    print(f"  💰 Credits exhausted for {base_url}")
                
        except Exception as e:
            print(f"  ❌ Error mapping {base_url}: {e}")
            continue
    
    # If still no good results, try mapping without sitemap
    if len(final_urls) < 50:
        print(f"🔄 Trying mapping without sitemap...")
        try:
            retry_data = {
                'url': f"https://www.{clean_domain}",
                'limit': map_limit,
                'includeSubdomains': True,
                'ignoreSitemap': True  # Try without sitemap
            }
            
            # Add search parameter if provided
            if search_param:
                retry_data['search'] = search_param
                
            response = requests.post(f'{FIRECRAWL_BASE_URL}/map', headers=headers, json=retry_data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                retry_urls = result.get('links', [])
                if len(retry_urls) > len(final_urls):
                    final_urls = retry_urls
                    print(f"  ✅ Better result without sitemap: {len(retry_urls)} URLs")
        except:
            pass
    
    print(f"✅ Total unique URLs discovered: {len(final_urls)}")
    
    if len(final_urls) >= map_limit * 0.8:
        print(f"🎯 Excellent discovery! Found {len(final_urls)} URLs (near mapping limit)")
        print(f"   You can increase MAX_URLS_TO_MAP if you need more URLs")
    elif len(final_urls) < 100:
        print(f"⚠️  Found {len(final_urls)} URLs - may be limited by domain structure")
        print(f"   Some domains have fewer public pages or restricted sitemaps")
    
    return final_urls

# Sanitize function for reuse in batch processing
def sanitize_text(text: str) -> str:
    """Remove or escape problematic characters that break JSON"""
    if not text:
        return ""
    text = text.replace('"', "'")  # Replace double quotes
    text = text.replace('\\', '')  # Remove backslashes
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')  # Replace newlines/tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char.isspace())  # Remove control chars
    return text.strip()[:150]  # Shorter for batch efficiency

async def batch_grade_with_llm(results: List[PageResult]) -> List[PageResult]:
    """Grade multiple pages in a single API call for MAXIMUM SPEED"""
    # Filter results that have content to grade
    gradeable_results = [r for r in results if r.title or r.meta_description]
    if not gradeable_results:
        for result in results:
            result.graded = True
            result.title_score = result.description_score = result.h1_score = result.overall_score = 0
            result.explanation = "No content to grade"
        return results
    
    print(f"⚡ Batch grading {len(gradeable_results)} pages in single API call...")
    
    try:
        # Create ultra-efficient batch prompt
        batch_items = []
        for i, result in enumerate(gradeable_results):
            safe_title = sanitize_text(result.title or "N/A")
            safe_desc = sanitize_text(result.meta_description or "N/A") 
            safe_h1 = sanitize_text(result.h1_text or "N/A")
            batch_items.append(f"{i}|{safe_title}|{safe_desc}|{safe_h1}")
        
        prompt = f"""Grade {len(gradeable_results)} pages for SEO (1-10, 10=excellent). Format: INDEX|TITLE|DESCRIPTION|H1

{chr(10).join(batch_items)}

Return JSON: {{"results":[{{"i":0,"t":8,"d":7,"h":6,"o":7,"e":"brief reason"}}, ...]}}
Keys: i=index, t=title_score, d=description_score, h=h1_score, o=overall_score, e=explanation(max 60 chars)"""
        
        # Ultra-compact JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "t": {"type": "integer", "minimum": 1, "maximum": 10},
                            "d": {"type": "integer", "minimum": 1, "maximum": 10},
                            "h": {"type": "integer", "minimum": 1, "maximum": 10},
                            "o": {"type": "integer", "minimum": 1, "maximum": 10},
                            "e": {"type": "string", "maxLength": 60}
                        },
                        "required": ["i", "t", "d", "h", "o", "e"]
                    }
                }
            },
            "required": ["results"]
        }
        
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema", 
                "json_schema": {"name": "batch_seo", "schema": json_schema}
            },
            max_tokens=4000,  # Enough for large batches
            temperature=0.1   # Consistent scoring
        )
        
        # Parse batch response
        try:
            batch_data = json.loads(response.choices[0].message.content)
            score_map = {score['i']: score for score in batch_data.get('results', [])}
            
            # Apply scores
            for i, result in enumerate(gradeable_results):
                if i in score_map:
                    s = score_map[i]
                    result.title_score = max(1, min(10, s.get('t', 5)))
                    result.description_score = max(1, min(10, s.get('d', 5)))
                    result.h1_score = max(1, min(10, s.get('h', 5)))
                    result.overall_score = max(1, min(10, s.get('o', 5)))
                    result.explanation = s.get('e', 'No explanation')[:80]
                else:
                    # Fallback for missing items
                    result.title_score = result.description_score = result.h1_score = result.overall_score = 5
                    result.explanation = "Batch fallback"
                
                result.graded = True
                
            print(f"✅ Batch complete: {len(gradeable_results)} pages graded")
            
        except json.JSONDecodeError as e:
            print(f"❌ Batch JSON parsing failed: {e}")
            # Apply fallback scores to all
            for result in gradeable_results:
                result.title_score = result.description_score = result.h1_score = result.overall_score = 5
                result.explanation = "Batch parse error"
                result.graded = True
                
    except Exception as e:
        print(f"❌ Batch grading failed: {e}")
        # Apply fallback scores
        for result in gradeable_results:
            result.title_score = result.description_score = result.h1_score = result.overall_score = 5
            result.explanation = f"Batch failed: {str(e)[:30]}"
            result.graded = True
    
    return results

async def scrape_single_url(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> PageResult:
    """Scrape a single URL using ScrapingBee with optimized settings"""
    async with semaphore:
        result = PageResult(url=url)
        
        try:
            params = {
                'api_key': SCRAPINGBEE_API_KEY,
                'url': url,
                'json_response': 'true',
                'render_js': 'false',  # Speed optimization
                'extract_rules': json.dumps({
                    'title': 'title',
                    'meta_description': 'meta[name="description"]@content',
                    'og_title': 'meta[property="og:title"]@content',
                    'og_description': 'meta[property="og:description"]@content',
                    'h1_text': 'h1'
                }),
                'premium_proxy': 'false',
                'stealth_proxy': 'false',
                'block_ads': 'true',
                'block_resources': 'true'
            }
            
            async with session.get(SCRAPINGBEE_BASE_URL, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    extracted = data.get('body', {})
                    
                    result.title = extracted.get('title', '').strip()
                    result.meta_description = extracted.get('meta_description', '').strip()
                    result.h1_text = extracted.get('h1_text', '').strip()
                    
                    # Handle missing H1s
                    if not result.h1_text:
                        result.h1_text = "[NO H1 TAG]"  # Explicit marker for missing H1
                    
                    # Fallback to OG tags
                    if not result.title and extracted.get('og_title'):
                        result.title = extracted.get('og_title', '').strip()
                    if not result.meta_description and extracted.get('og_description'):
                        result.meta_description = extracted.get('og_description', '').strip()
                    
                    result.status = "success"
                    
                    # Enhanced logging with H1 info and debugging
                    h1_info = f"H1({len(result.h1_text)})" if result.h1_text != "[NO H1 TAG]" else "NO-H1"
                    title_info = f"T:{len(result.title)}" if result.title else "NO-TITLE"
                    desc_info = f"D:{len(result.meta_description)}" if result.meta_description else "NO-DESC"
                    print(f"✅ Scraped: {url[:40]}... | {title_info} {desc_info} {h1_info}")
                    
                    # DEBUG: Check if content was actually extracted
                    if not result.title and not result.meta_description and result.h1_text in ["[NO H1 TAG]", "[EMPTY H1]"]:
                        print(f"⚠️ DEBUG: No content extracted from {url[:40]}...")
                        print(f"   Raw extracted keys: {list(extracted.keys()) if extracted else 'None'}")
                        if extracted:
                            print(f"   Sample values: {dict(list(extracted.items())[:3]) if extracted else 'None'}")
                    
                else:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text[:100]}"
                    result.status = "error"
                    print(f"🚨 ScrapingBee ERROR for {url[:40]}...")
                    print(f"   HTTP Status: {response.status}")
                    print(f"   Error: {error_text[:200]}")
                    
                    # Special handling for common issues
                    if response.status == 401:
                        print("   💡 This looks like an API key issue")
                    elif response.status == 402:
                        print("   💡 This looks like a credits/billing issue")
                    elif response.status == 429:
                        print("   💡 This looks like rate limiting")
                    elif response.status == 403:
                        print("   💡 This might be a blocked domain or forbidden access")
                    
        except Exception as e:
            result.error = str(e)[:100]
            result.status = "error"
        
        return result

async def process_with_real_time_grading(urls: List[str], csv_writer: RealTimeCSVWriter, target_count: int):
    """Process URLs with real-time LLM grading and CSV writing"""
    print(f"🚀 Starting ULTRA-FAST batch scraping + grading with {MAX_CONCURRENT} scrape workers...")
    print(f"⚡ Batch processing: {BATCH_SIZE} pages per API call for MAXIMUM SPEED")
    print(f"🎯 Target: {target_count} graded results")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=15)
    
    successful_count = 0
    processed_count = 0
    
    # Track scores for statistics
    all_scores = {
        'title_scores': [],
        'description_scores': [],
        'h1_scores': [],
        'overall_scores': []
    }
    
    # Track score distributions (0-10: 0=failed grading, 1-10=structured JSON scores)
    score_distributions = {
        'title_distribution': {i: 0 for i in range(0, 11)},
        'description_distribution': {i: 0 for i in range(0, 11)},
        'h1_distribution': {i: 0 for i in range(0, 11)},
        'overall_distribution': {i: 0 for i in range(0, 11)}
    }
    
    # Track character lengths for analysis
    char_lengths = {
        'title_lengths': [],
        'description_lengths': [],
        'h1_lengths': []
    }
    
    # Store all results for final analysis
    all_results = []
    
    # Dynamic batch sizing based on target count
    if target_count <= 10:
        batch_size = 20
    elif target_count <= 50:
        batch_size = 50
    elif target_count <= 200:
        batch_size = 100
    else:
        batch_size = 200
        
    print(f"📦 Using batch size: {batch_size} (optimized for target: {target_count})")
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        url_index = 0
        
        while successful_count < target_count and url_index < len(urls):
            # Calculate batch size for this iteration
            remaining_needed = target_count - successful_count
            urls_needed = min(remaining_needed * 3, batch_size)  # 3x buffer for failures
            
            batch_urls = urls[url_index:url_index + urls_needed]
            if not batch_urls:
                break
                
            print(f"📊 Processing batch {(url_index // batch_size) + 1} ({len(batch_urls)} URLs) - {successful_count}/{target_count} graded...")
            
            # Scrape batch of URLs concurrently
            scrape_tasks = [scrape_single_url(session, url, semaphore) for url in batch_urls]
            scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
            # Separate successful and failed scrapes for concurrent processing
            successful_scrapes = []
            failed_scrapes = []
            
            for scraped_result in scraped_results:
                if isinstance(scraped_result, Exception):
                    continue
                
                processed_count += 1
                
                if scraped_result.status == "success" and (scraped_result.title or scraped_result.meta_description):
                    successful_scrapes.append(scraped_result)
                    print(f"📝 Ready for grading: {scraped_result.url[:40]}... (T:{bool(scraped_result.title)} D:{bool(scraped_result.meta_description)})")
                else:
                    failed_scrapes.append(scraped_result)
                    print(f"❌ Failed/no content: {scraped_result.url[:40]}... Status:{scraped_result.status} T:{bool(scraped_result.title)} D:{bool(scraped_result.meta_description)}")
            
            # Write failed attempts immediately for transparency
            for failed_result in failed_scrapes:
                csv_writer.write_result(failed_result)
            
            # Grade successful scrapes using BATCH processing for MAXIMUM SPEED
            if successful_scrapes:
                # Process in batches for ultra-fast grading
                batch_start = 0
                while batch_start < len(successful_scrapes) and successful_count < target_count:
                    batch_end = min(batch_start + BATCH_SIZE, len(successful_scrapes))
                    batch_to_grade = successful_scrapes[batch_start:batch_end]
                    
                    # Grade entire batch in single API call
                    graded_batch = await batch_grade_with_llm(batch_to_grade)
                    
                    # Process graded results
                    for graded_result in graded_batch:
                        if graded_result.graded:
                            # Write to CSV immediately
                            csv_writer.write_result(graded_result)
                            
                            # Track scores for statistics
                            all_scores['title_scores'].append(graded_result.title_score)
                            all_scores['description_scores'].append(graded_result.description_score)
                            all_scores['h1_scores'].append(graded_result.h1_score)
                            all_scores['overall_scores'].append(graded_result.overall_score)
                            
                            # Track score distributions
                            score_distributions['title_distribution'][graded_result.title_score] += 1
                            score_distributions['description_distribution'][graded_result.description_score] += 1
                            score_distributions['h1_distribution'][graded_result.h1_score] += 1
                            score_distributions['overall_distribution'][graded_result.overall_score] += 1
                            
                            # Track character lengths (skip special markers for missing content)
                            if graded_result.title:
                                char_lengths['title_lengths'].append(len(graded_result.title))
                            if graded_result.meta_description:
                                char_lengths['description_lengths'].append(len(graded_result.meta_description))
                            if graded_result.h1_text and not graded_result.h1_text.startswith('['):
                                char_lengths['h1_lengths'].append(len(graded_result.h1_text))
                            
                            # Store full result for final analysis
                            all_results.append(graded_result)
                            
                            successful_count += 1
                            
                            if successful_count >= target_count:
                                break
                    
                    batch_start = batch_end
                    
                    if successful_count >= target_count:
                        break
            
            print(f"✅ Batch complete: {successful_count} graded, {processed_count} total processed")
            url_index += len(batch_urls)
            
            if successful_count >= target_count:
                break
    
    return all_scores, score_distributions, char_lengths, all_results

def get_user_input():
    """Get domain, target count, and search parameters from user"""
    print("\n🎯 Real-Time SEO Domain Scraper + LLM Grader")
    print("=" * 60)
    print("🤖 Powered by GPT-4o Mini for fast, cost-effective SEO scoring")
    print("📊 Real-time streaming results to CSV")
    print("Examples: peerspace.com, giggster.com, airbnb.com")
    
    # Get domain
    while True:
        domain = input("\nEnter domain to scrape and grade: ").strip()
        if domain:
            break
        print("Please enter a valid domain.")
    
    # Ask about search filtering  
    print("\n🔍 SEARCH FILTERING OPTIONS:")
    print("   Filter URLs by content/path keywords (fast Firecrawl /map endpoint)")
    print("   Examples:")
    print("   - 'blog' → finds /blog/, /company-blog/, etc.")
    print("   - 'api' → finds /api/, /rapid-api-guide/, etc.")
    print("   - 'product' → finds /products/, /product-categories/, etc.")
    print("   - 'doc' → finds /docs/, /documentation/, etc.")
    
    # Ask for search parameter
    search_param = input("\nSearch for URLs containing (leave blank for all URLs): ").strip()
    if search_param:
        print(f"🔍 Will search for URLs containing: '{search_param}'")
    else:
        print("📊 Will map all available URLs")
    
    # Display optimization info
    print("\n💡 Tip: Start with 5 to test LLM grading, then use 1000 for full analysis")
    print("💰 Credit-optimized: Mapping limits automatically reduced to save Firecrawl credits")
    print("🎯 Token-optimized: Structured JSON output eliminates parsing errors and reduces costs")
    print("📊 Simple rubric: FAIL (1-3), NEEDS IMPROVEMENT (4-7), GOOD (8-10)")
    print("📈 Includes ASCII distribution charts showing score patterns visually")
    print("⚡ Using fast /map endpoint for URL discovery")
    
    # Get target count
    while True:
        try:
            target = input("\nHow many results to grade? (recommended: 5 or 1000): ").strip()
            target_count = int(target)
            if target_count > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    return domain, target_count, search_param

async def main():
    """Main execution function"""
    create_output_folder()
    
    # Get domain, target count, and search parameter from user
    domain, target_count, search_param = get_user_input()
    
    # Create output filename
    clean_domain = domain.replace('https://', '').replace('http://', '').replace('www.', '').replace('/', '').replace('.', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"output/{clean_domain}_{timestamp}_seo_graded.csv"
    
    print(f"\n🎯 Starting Real-Time SEO Analysis: {domain}")
    print("=" * 70)
    print(f"⚡ Configuration: {MAX_CONCURRENT} scrape workers, {BATCH_SIZE} pages/batch (ULTRA-FAST BATCH MODE)")
    print(f"🎯 Target: {target_count} graded results")
    print(f"🤖 LLM Model: GPT-4o Mini ($0.15/1M input + $0.60/1M output)")
    print(f"💾 Real-time output: {csv_filename}")
    print()
    
    # Step 1: Get URLs with optional search filtering
    all_urls = get_domain_urls(domain, target_count, search_param)
    if not all_urls:
        print("❌ No URLs found, exiting...")
        return
    
    # Filter content URLs
    content_urls = [url for url in all_urls if not any(skip in url.lower() for skip in [
        'api.', 'sitemap', '.xml', '.json', '/api/', 'favicon', '.css', '.js', '.pdf', '.jpg', '.png'
    ])]
    
    print(f"📊 Content URLs available: {len(content_urls)}")
    
    # Check if domain has enough URLs
    if len(content_urls) < target_count:
        print(f"⚠️  Domain only has {len(content_urls)} content URLs (requested {target_count})")
        print(f"🎯 Will process all {len(content_urls)} available URLs")
        actual_target = len(content_urls)
    else:
        print(f"🎯 Will process URLs until we get {target_count} graded results")
        print(f"⚡ Smart processing: Starting from top URLs, will only scrape ~{target_count * 3} URLs (with failure buffer)")
        actual_target = target_count
    
    print(f"🔴 LIVE: Starting real-time processing...")
    print()
    
    # Step 2: Initialize CSV writer
    csv_writer = RealTimeCSVWriter(csv_filename)
    
    # Step 3: Process with real-time grading
    start_time = time.time()
    
    all_scores, score_distributions, char_lengths, all_results = await process_with_real_time_grading(content_urls, csv_writer, actual_target)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    successful_count = len(all_results)
    total_processed = sum(len(scores) for scores in all_scores.values()) // 4  # Divide by 4 since we have 4 score types
    
    print("\n" + "=" * 70)
    print("🎉 REAL-TIME SEO ANALYSIS COMPLETED!")
    print(f"🗺️  Domain: {domain}")
    print(f"📊 URLs discovered: {len(all_urls)}")
    print(f"📊 Content URLs available: {len(content_urls)}")
    print(f"🔄 URLs processed: {total_processed}")
    print(f"✅ Successfully graded: {successful_count}")
    print(f"🎯 Target reached: {successful_count >= actual_target}")
    print(f"⚡ Workers: {MAX_CONCURRENT} scrape, {BATCH_SIZE} pages/batch (ULTRA-FAST MODE)")
    print(f"🤖 LLM Model: GPT-4o Mini")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    if total_processed > 0:
        print(f"🚀 Processing speed: {total_processed/total_time:.1f} URLs/second")
        print(f"📈 Success rate: {successful_count/total_processed*100:.1f}%")
    print(f"💾 Results saved to: {csv_filename}")
    print("🎯 Ready for analysis and optimization!")
    print("=" * 70)
    
    # Display simplified statistics using the clean formatter
    display_enhanced_statistics(all_scores, score_distributions, char_lengths, all_results)

if __name__ == "__main__":
    asyncio.run(main())