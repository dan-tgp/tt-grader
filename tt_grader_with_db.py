#!/usr/bin/env python3
"""
TT-GRADER WITH DATABASE - SEO grader with persistent storage
Stores all runs and results in SQLite database for analysis

UPDATED: Now using GPT-5 Nano via OpenAI Responses API
- Model: gpt-5-nano (80% cheaper than GPT-5 mini, 67% cheaper than GPT-4o mini)  
- API: OpenAI Responses API (not Chat Completions)
- Parameters: reasoning effort="minimal", verbosity="low"
- Better accuracy than GPT-4o with ~45% fewer errors
- Ultra-fast response times for high-volume processing
"""

import asyncio
import aiohttp
import requests
import json
import csv
import time
import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI

# Load environment variables
def load_env():
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                    elif ':' in line:
                        key, value = line.split(':', 1)
                    else:
                        continue
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("ERROR: .env file not found.")
        exit(1)
    return env_vars

env_vars = load_env()
SCRAPINGBEE_API_KEY = env_vars.get('SCRAPINGBEE_API_KEY')
FIRECRAWL_API_KEY = env_vars.get('FIRECRAWL_KEY')
OPENAI_API_KEY = env_vars.get('OPENAI_KEY')

# Configuration
MAX_CONCURRENT_SCRAPING = 50
MAX_CONCURRENT_LLM = 100
BATCH_SIZE = 25
SCRAPINGBEE_TIMEOUT = 10
SCRAPINGBEE_BASE_URL = "https://app.scrapingbee.com/api/v1/"
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"
DATABASE_PATH = "tt_grader.db"

@dataclass
class PageResult:
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

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                path_filter TEXT,
                target_count INTEGER,
                successful_count INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_duration_seconds REAL,
                discovery_time_seconds REAL,
                scraping_time_seconds REAL,
                grading_time_seconds REAL,
                pages_per_second REAL,
                csv_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                meta_description TEXT,
                h1_text TEXT,
                title_content TEXT,
                description_content TEXT,
                h1_content TEXT,
                title_score INTEGER,
                description_score INTEGER,
                h1_score INTEGER,
                overall_score INTEGER,
                explanation TEXT,
                status TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs (id)
            )
        ''')
        
        # Add new columns if they don't exist (migration)
        cursor.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'title_content' not in columns:
            cursor.execute('ALTER TABLE results ADD COLUMN title_content TEXT')
            cursor.execute('ALTER TABLE results ADD COLUMN description_content TEXT')
            cursor.execute('ALTER TABLE results ADD COLUMN h1_content TEXT')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_overall ON results(overall_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_url ON results(url)')
        
        conn.commit()
        conn.close()
    
    def create_run(self, domain: str, path_filter: Optional[str], 
                   target_count: int, csv_filename: str) -> int:
        """Create a new run record and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO runs (domain, path_filter, target_count, 
                            start_time, csv_filename)
            VALUES (?, ?, ?, ?, ?)
        ''', (domain, path_filter, target_count, 
              datetime.now(), csv_filename))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id
    
    def update_run(self, run_id: int, successful_count: int, discovery_time: float,
                   scraping_time: float, grading_time: float, total_time: float):
        """Update run with completion statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pages_per_second = successful_count / total_time if total_time > 0 else 0
        
        cursor.execute('''
            UPDATE runs 
            SET successful_count = ?, 
                end_time = ?,
                total_duration_seconds = ?,
                discovery_time_seconds = ?,
                scraping_time_seconds = ?,
                grading_time_seconds = ?,
                pages_per_second = ?
            WHERE id = ?
        ''', (successful_count, datetime.now(), total_time, discovery_time,
              scraping_time, grading_time, pages_per_second, run_id))
        
        conn.commit()
        conn.close()
    
    def save_result(self, run_id: int, result: PageResult):
        """Save a single result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO results (run_id, url, title, meta_description, h1_text,
                               title_content, description_content, h1_content,
                               title_score, description_score, h1_score, overall_score,
                               explanation, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, result.url, result.title, result.meta_description, result.h1_text,
              result.title, result.meta_description, result.h1_text,
              result.title_score, result.description_score, result.h1_score, 
              result.overall_score, result.explanation, result.status, result.error))
        
        conn.commit()
        conn.close()
    
    def save_results_batch(self, run_id: int, results: List[PageResult]):
        """Save multiple results in a batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = [(run_id, r.url, r.title, r.meta_description, r.h1_text,
                r.title, r.meta_description, r.h1_text,
                r.title_score, r.description_score, r.h1_score, r.overall_score,
                r.explanation, r.status, r.error) for r in results]
        
        cursor.executemany('''
            INSERT INTO results (run_id, url, title, meta_description, h1_text,
                               title_content, description_content, h1_content,
                               title_score, description_score, h1_score, overall_score,
                               explanation, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        conn.close()
    
    def get_run_summary(self, run_id: int) -> Dict[str, Any]:
        """Get summary statistics for a run"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get run details
        cursor.execute('SELECT * FROM runs WHERE id = ?', (run_id,))
        run = dict(cursor.fetchone())
        
        # Get result statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_results,
                AVG(overall_score) as avg_overall,
                AVG(title_score) as avg_title,
                AVG(description_score) as avg_description,
                AVG(h1_score) as avg_h1,
                COUNT(CASE WHEN overall_score >= 8 THEN 1 END) as excellent_count,
                COUNT(CASE WHEN overall_score < 6 THEN 1 END) as needs_work_count
            FROM results
            WHERE run_id = ? AND status = 'success'
        ''', (run_id,))
        
        stats = dict(cursor.fetchone())
        run['statistics'] = stats
        
        conn.close()
        return run
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent runs with statistics"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.*, 
                   COUNT(res.id) as result_count,
                   AVG(res.overall_score) as avg_score
            FROM runs r
            LEFT JOIN results res ON r.id = res.run_id AND res.status = 'success'
            GROUP BY r.id
            ORDER BY r.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return runs

class RealTimeCSVWriter:
    def __init__(self, filename: str, db_manager: DatabaseManager, run_id: int):
        self.filename = filename
        self.headers_written = False
        self.lock = asyncio.Lock()
        self.db_manager = db_manager
        self.run_id = run_id
        self.result_buffer = []
        self.buffer_size = 10  # Save to DB every 10 results
    
    async def write_result(self, result: PageResult):
        async with self.lock:
            # Write to CSV
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
            
            # Buffer for database
            self.result_buffer.append(result)
            
            # Save to database when buffer is full
            if len(self.result_buffer) >= self.buffer_size:
                self.db_manager.save_results_batch(self.run_id, self.result_buffer)
                self.result_buffer = []
    
    async def flush_buffer(self):
        """Save any remaining buffered results to database"""
        async with self.lock:
            if self.result_buffer:
                self.db_manager.save_results_batch(self.run_id, self.result_buffer)
                self.result_buffer = []

def map_and_filter_urls(domain: str, map_limit: int = 5000, path_filter: Optional[str] = None) -> List[str]:
    """
    Step 1: Map large number of URLs
    Step 2: Optionally filter URLs by path
    """
    clean_domain = domain.replace('http://', '').replace('https://', '').replace('www.', '').strip('/')
    
    print(f"Mapping up to {map_limit} URLs from {domain}...")
    if path_filter:
        print(f"Will filter for URLs containing '{path_filter}'")
    
    headers = {
        'Authorization': f'Bearer {FIRECRAWL_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        data = {
            'url': f"https://www.{clean_domain}",
            'limit': map_limit,
            'includeSubdomains': True,
            'ignoreSitemap': False
        }
        
        print("Requesting URL map from Firecrawl...")
        response = requests.post(f'{FIRECRAWL_BASE_URL}/map', headers=headers, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            all_urls = result.get('links', [])
            print(f"Retrieved {len(all_urls)} total URLs")
            
            # Apply filters
            filtered_urls = all_urls
            
            if path_filter:
                filtered_urls = [url for url in filtered_urls if path_filter.lower() in url.lower()]
                print(f"After path filter: {len(filtered_urls)} URLs")
            
            
            return filtered_urls
        else:
            print(f"Firecrawl mapping failed: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error during mapping: {e}")
        return []

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('"', "'")
    text = text.replace('\\', '')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = ''.join(char for char in text if ord(char) >= 32 or char.isspace())
    return text.strip()[:150]

async def batch_grade(results: List[PageResult], batch_id: int = 0) -> List[PageResult]:
    """Grade pages using the optimized TopTitle-focused prompt"""
    gradeable_results = [r for r in results if r.title or r.meta_description]
    if not gradeable_results:
        for result in results:
            result.graded = True
            result.title_score = result.description_score = result.h1_score = result.overall_score = 0
            result.explanation = "No content to grade"
        return results
    
    print(f"Batch {batch_id}: Grading {len(gradeable_results)} pages...")
    
    try:
        batch_items = []
        for i, result in enumerate(gradeable_results):
            safe_title = sanitize_text(result.title or "N/A")
            safe_desc = sanitize_text(result.meta_description or "N/A") 
            safe_h1 = sanitize_text(result.h1_text or "N/A")
            safe_url = result.url.split('/')[-1] if '/' in result.url else result.url
            safe_url = safe_url.replace('-', ' ').replace('_', ' ')[:50]
            batch_items.append(f"{i}|{safe_title}|{safe_desc}|{safe_h1}|{safe_url}")
        
        system_message = """You are a professional SEO consultant analyzing pages for optimization opportunities. Score each element separately, then provide detailed explanations that SEO professionals will respect.

Input Format: INDEX|TITLE|DESCRIPTION|H1|URL_SLUG

First, infer the most likely TARGET SEARCH QUERY (keyphrase) for this page - what 2-5 word phrase would users type into Google to find this content? Consider the URL structure and content context to determine the realistic search query this page should be optimized for. Examples: "email marketing software", "how to grow tomatoes", "best running shoes women", "chicago wedding photographer". Use this target keyphrase to evaluate optimization effectiveness (but never mention this simulated keyphrase to the user - it's just for internal scoring).

TITLE SCORING (t) - 1-10 scale:
• Length: 50-60 chars = 10, 60-70 = 9, 70-80 = 8, 80-90 = 7, <50 = 6, >90 = 5 or less
• Target keyphrase placement: Front-loaded target query = +2, middle = +1, missing = -3
• Click-worthiness: Compelling/specific = +2, generic = -2
• SERP truncation risk: Will truncate badly = -2

META DESCRIPTION SCORING (d) - 1-10 scale:
• Length: 150-160 = 10, 160-200 = 9, 200-250 = 8, 250-275 = 7, <120 = 5, >275 = 4
• Call-to-action presence: Strong CTA = +2, weak = +1, none = -2
• Value proposition clarity: Clear benefit = +2, vague = -1
• Target query usage: Natural keyphrase inclusion = +1, stuffed = -1, missing = -2

H1 SCORING (h) - 1-10 scale:
• Search intent match: Perfect match to target search query = 10, related = 7, generic = 4
• User question answered: Directly addresses search intent = +2
• Specificity: Specific to the keyphrase topic = +2, generic = -2

COHERENCE SCORING (o) - How elements work together:
• All three elements target same search query/intent = 10
• Elements reinforce same value proposition = +2
• Conflicting focus between elements = -4
• Elements match page purpose and target keyphrase = +2

PROFESSIONAL EXPLANATION (e):
Explain WHY the score is low/high in SEO terms a professional would respect. Be specific about the issue and its impact on search performance. Max 120 chars.

Good examples:
"Strong: Title targets 'email marketing automation' perfectly & front-loads the full keyphrase"
"Weak: Generic H1 'Our Services' - should target actual search query like 'web design services chicago'"
"Title missing target query - users searching 'project management tools' won't find this"
"Meta description lacks the search query and CTA - both needed for SERP click-through"
"Elements misaligned: Title suggests 'CRM software' but H1 says 'sales tools' - mixed intent signals"
"Perfect coherence: All elements focus on 'organic garden fertilizer' with clear value props"

Return JSON: {"results":[{"i":0,"t":6,"d":4,"h":7,"o":5,"e":"Title strong but meta weak - no CTA and vague value prop reduces click-through potential"}, ...]}"""
        
        user_message = f"""Grade these {len(gradeable_results)} pages:

{chr(10).join(batch_items)}"""
        
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
                            "e": {"type": "string", "maxLength": 120}
                        },
                        "required": ["i", "t", "d", "h", "o", "e"]
                    }
                }
            },
            "required": ["results"]
        }
        
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # GPT-5 nano with Chat Completions API (minimal parameters for compatibility)
        response = await openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={
                "type": "json_schema", 
                "json_schema": {"name": "batch_seo", "schema": json_schema}
            },
            max_completion_tokens=4000  # GPT-5 nano only supports default temperature (1.0)
        )
        
        # Standard Chat Completions response format
        batch_data = json.loads(response.choices[0].message.content)
        score_map = {score['i']: score for score in batch_data.get('results', [])}
        
        for i, result in enumerate(gradeable_results):
            if i in score_map:
                s = score_map[i]
                result.title_score = max(1, min(10, s.get('t', 5)))
                result.description_score = max(1, min(10, s.get('d', 5)))
                result.h1_score = max(1, min(10, s.get('h', 5)))
                result.overall_score = max(1, min(10, s.get('o', 5)))
                result.explanation = s.get('e', 'No explanation')[:120]
            else:
                result.title_score = result.description_score = result.h1_score = result.overall_score = 5
                result.explanation = "Batch fallback"
            
            result.graded = True
            
        print(f"Batch {batch_id}: COMPLETE ({len(gradeable_results)} pages)")
        return results
            
    except Exception as e:
        print(f"Batch {batch_id}: FAILED ({e})")
        for result in gradeable_results:
            result.title_score = result.description_score = result.h1_score = result.overall_score = 5
            result.explanation = f"Batch failed: {str(e)[:30]}"
            result.graded = True
        return results

async def scrape_url(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> PageResult:
    """Scrape individual URLs with optimized extraction"""
    async with semaphore:
        result = PageResult(url=url)
        
        try:
            params = {
                'api_key': SCRAPINGBEE_API_KEY,
                'url': url,
                'json_response': 'true',
                'render_js': 'false',
                'extract_rules': json.dumps({
                    'title': 'title',
                    'meta_description': 'meta[name="description"]@content',
                    'h1_text': 'h1:first-of-type'
                }),
                'premium_proxy': 'false',
                'stealth_proxy': 'false',
                'block_ads': 'true',
                'block_resources': 'true',
                'timeout': SCRAPINGBEE_TIMEOUT * 1000
            }
            
            async with session.get(SCRAPINGBEE_BASE_URL, params=params, timeout=SCRAPINGBEE_TIMEOUT + 5) as response:
                if response.status == 200:
                    data = await response.json()
                    extracted = data.get('body', {})
                    
                    result.title = extracted.get('title', '').strip()
                    result.meta_description = extracted.get('meta_description', '').strip()
                    result.h1_text = extracted.get('h1_text', '').strip()
                    
                    if not result.h1_text:
                        result.h1_text = "[NO H1 TAG]"
                    
                    result.status = "success"
                    
                else:
                    result.error = f"HTTP {response.status}"
                    result.status = "error"
                    
        except Exception as e:
            result.error = str(e)[:100]
            result.status = "error"
        
        return result

async def process_urls(urls: List[str], csv_writer: RealTimeCSVWriter, target_count: int):
    """Process URLs with full optimization pipeline"""
    print("=" * 70)
    print("PROCESSING")
    print("=" * 70)
    print(f"Scraping workers: {MAX_CONCURRENT_SCRAPING}")
    print(f"LLM batch workers: {MAX_CONCURRENT_LLM}")
    print(f"Target: {target_count} graded results")
    print(f"Available URLs: {len(urls)}")
    print()
    
    # Semaphores for rate limiting
    scrape_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPING)
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    
    timeout = aiohttp.ClientTimeout(total=SCRAPINGBEE_TIMEOUT + 5)
    
    successful_count = 0
    
    # Use only the URLs we need (with some buffer for failures)
    urls_to_process = urls[:min(target_count * 2, len(urls))]
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Step 1: Scrape URLs
        print(f"Step 1: Scraping {len(urls_to_process)} URLs...")
        scrape_start = time.time()
        
        scrape_tasks = [scrape_url(session, url, scrape_semaphore) for url in urls_to_process]
        scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        scrape_time = time.time() - scrape_start
        print(f"Scraping complete: {scrape_time:.1f}s")
        
        # Step 2: Filter successful scrapes
        successful_scrapes = []
        failed_count = 0
        
        for scraped_result in scraped_results:
            if isinstance(scraped_result, Exception):
                failed_count += 1
                continue
                
            if scraped_result.status == "success" and (scraped_result.title or scraped_result.meta_description):
                successful_scrapes.append(scraped_result)
            else:
                await csv_writer.write_result(scraped_result)
                failed_count += 1
        
        print(f"Successful scrapes: {len(successful_scrapes)}")
        print(f"Failed scrapes: {failed_count}")
        
        # Step 3: Grade results
        if successful_scrapes:
            print(f"\nStep 2: Grading pages...")
            grade_start = time.time()
            
            to_grade = successful_scrapes[:target_count]
            
            async def grade_batch_with_limit(batch_data, batch_num):
                async with llm_semaphore:
                    return await batch_grade(batch_data, batch_num)
            
            grade_tasks = []
            for i in range(0, len(to_grade), BATCH_SIZE):
                batch = to_grade[i:i + BATCH_SIZE]
                batch_id = i // BATCH_SIZE + 1
                grade_tasks.append(grade_batch_with_limit(batch, batch_id))
            
            print(f"Launching {len(grade_tasks)} concurrent batch calls...")
            
            graded_batches = await asyncio.gather(*grade_tasks, return_exceptions=True)
            
            grade_time = time.time() - grade_start
            print(f"LLM grading complete: {grade_time:.1f}s")
            
            # Step 4: Write results
            write_start = time.time()
            write_tasks = []
            
            for batch_results in graded_batches:
                if isinstance(batch_results, Exception):
                    print(f"Batch exception: {batch_results}")
                    continue
                    
                for result in batch_results:
                    if result.graded:
                        write_tasks.append(csv_writer.write_result(result))
                        successful_count += 1
            
            await asyncio.gather(*write_tasks)
            await csv_writer.flush_buffer()  # Ensure all results are saved to DB
            
            write_time = time.time() - write_start
            print(f"CSV writing complete: {write_time:.1f}s")
    
    return successful_count, scrape_time, grade_time

async def run_grader(
    domain: str,
    target_count: int = 500,
    map_limit: int = 5000,
    path_filter: Optional[str] = None,
    output_name: Optional[str] = None
):
    """
    Main grader function with database storage
    
    Args:
        domain: Domain to analyze (e.g., "example.com")
        target_count: Number of pages to grade
        map_limit: Maximum URLs to map initially
        path_filter: Optional path filter (e.g., "/product/")
        output_name: Optional custom name for output file
    """
    print("=" * 70)
    print("TT-GRADER WITH DATABASE")
    print("=" * 70)
    print(f"Domain: {domain}")
    if path_filter:
        print(f"Path filter: '{path_filter}'")
    print(f"Map limit: {map_limit} URLs")
    print(f"Target: {target_count} graded pages")
    print()
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_name:
        csv_filename = f"output/{output_name}_{timestamp}.csv"
    else:
        clean_domain = domain.replace('.', '_')
        csv_filename = f"output/{clean_domain}_{timestamp}.csv"
    
    # Create run in database
    run_id = db_manager.create_run(domain, path_filter, target_count, csv_filename)
    print(f"Database run ID: {run_id}")
    
    csv_writer = RealTimeCSVWriter(csv_filename, db_manager, run_id)
    
    # Step 1 & 2: Map and filter URLs
    discovery_start = time.time()
    filtered_urls = map_and_filter_urls(domain, map_limit, path_filter)
    discovery_time = time.time() - discovery_start
    
    if not filtered_urls:
        print("No URLs found matching criteria!")
        db_manager.update_run(run_id, 0, discovery_time, 0, 0, discovery_time)
        return
    
    print(f"Discovery complete: {discovery_time:.1f}s")
    print(f"Found {len(filtered_urls)} URLs to process")
    
    if len(filtered_urls) < target_count:
        print(f"WARNING: Only found {len(filtered_urls)} URLs, less than target of {target_count}")
        target_count = len(filtered_urls)
    
    # Step 3: Process URLs
    process_start = time.time()
    successful_count, scrape_time, grade_time = await process_urls(filtered_urls, csv_writer, target_count)
    total_process_time = time.time() - process_start
    
    total_time = discovery_time + total_process_time
    
    # Update run in database
    db_manager.update_run(run_id, successful_count, discovery_time, scrape_time, grade_time, total_time)
    
    # Final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Target: {target_count}")
    print(f"Successfully graded: {successful_count}")
    print(f"Success rate: {successful_count/target_count*100:.1f}%")
    print()
    print("TIMING:")
    print(f"  URL Discovery: {discovery_time:.1f}s")
    print(f"  Scraping: {scrape_time:.1f}s")
    print(f"  LLM Analysis: {grade_time:.1f}s") 
    print(f"  Total Processing: {total_process_time:.1f}s")
    print(f"  TOTAL TIME: {total_time:.1f}s")
    print()
    print("PERFORMANCE:")
    if successful_count > 0:
        overall_speed = successful_count/total_time
        processing_speed = successful_count/total_process_time
        
        print(f"  Analysis Speed: {overall_speed:.1f} pages/second")
        print(f"  Processing Speed: {processing_speed:.1f} pages/second")
    
    # Get run summary from database
    summary = db_manager.get_run_summary(run_id)
    if summary['statistics']['total_results'] > 0:
        print()
        print("DATABASE STATISTICS:")
        print(f"  Average Overall Score: {summary['statistics']['avg_overall']:.1f}")
        print(f"  Average Title Score: {summary['statistics']['avg_title']:.1f}")
        print(f"  Average Description Score: {summary['statistics']['avg_description']:.1f}")
        print(f"  Excellent Pages (8-10): {summary['statistics']['excellent_count']}")
        print(f"  Pages Needing Work (<6): {summary['statistics']['needs_work_count']}")
            
    print(f"\nOutput saved to: {csv_filename}")
    print(f"Results stored in database with Run ID: {run_id}")
    print("=" * 70)
    
    return run_id

def view_recent_runs(limit: int = 10):
    """View recent runs from database"""
    db_manager = DatabaseManager()
    runs = db_manager.get_recent_runs(limit)
    
    print("=" * 70)
    print("RECENT RUNS")
    print("=" * 70)
    
    if not runs:
        print("No runs found in database")
        return
    
    print(f"{'ID':<5} {'Domain':<20} {'Path Filter':<15} {'Results':<10} {'Avg Score':<10} {'Date':<20}")
    print("-" * 80)
    
    for run in runs:
        domain = run['domain'][:18] + '..' if len(run['domain']) > 20 else run['domain']
        path = (run['path_filter'] or 'None')[:13] + '..' if run['path_filter'] and len(run['path_filter']) > 15 else (run['path_filter'] or 'None')
        avg_score = f"{run['avg_score']:.1f}" if run['avg_score'] else "N/A"
        date = run['created_at'][:19] if run['created_at'] else "Unknown"
        
        print(f"{run['id']:<5} {domain:<20} {path:<15} {run['result_count']:<10} {avg_score:<10} {date:<20}")
    
    print("=" * 70)

def export_run_to_csv(run_id: int, output_file: Optional[str] = None):
    """Export a specific run from database to CSV"""
    db_manager = DatabaseManager()
    conn = sqlite3.connect(db_manager.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM results WHERE run_id = ?', (run_id,))
    results = cursor.fetchall()
    
    if not results:
        print(f"No results found for run ID {run_id}")
        return
    
    if not output_file:
        output_file = f"export_run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'meta_description', 'h1_text', 'title_score', 
                     'description_score', 'h1_score', 'overall_score', 'explanation', 'status', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            writer.writerow({
                'url': row['url'],
                'title': row['title'],
                'meta_description': row['meta_description'],
                'h1_text': row['h1_text'],
                'title_score': row['title_score'],
                'description_score': row['description_score'],
                'h1_score': row['h1_score'],
                'overall_score': row['overall_score'],
                'explanation': row['explanation'],
                'status': row['status'],
                'error': row['error']
            })
    
    print(f"Exported run {run_id} to {output_file}")
    conn.close()

if __name__ == "__main__":
    # Example: Run a new analysis
    asyncio.run(run_grader(
        domain="example.com",
        target_count=500,
        map_limit=5000,
        path_filter="/product/",
        output_name="example_products"
    ))
    
    # Example: View recent runs
    # view_recent_runs(10)
    
    # Example: Export a specific run
    # export_run_to_csv(1, "run_1_export.csv")