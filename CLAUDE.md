# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based SEO analysis tool that scrapes websites and provides LLM-powered grading of SEO elements. The system uses:

- ScrapingBee API for web scraping
- Firecrawl API for URL discovery via domain mapping  
- OpenAI GPT-4o Mini for SEO scoring and analysis
- Async/concurrent processing for high-performance scraping

## Core Architecture

### Main Components

- `real_time_seo_grader.py` - Main application script with interactive CLI interface
- `output_formatter.py` - Results display and statistics formatting module
- `output/` - Directory containing CSV results and analysis files
- `.env` - API keys configuration (ScrapingBee, Firecrawl, OpenAI)

### Key Classes and Functions

- `PageResult` dataclass - Structured data for page scraping and grading results
- `RealTimeCSVWriter` - Streams results to CSV as they are processed
- `batch_grade_with_llm()` - Batches multiple pages for efficient LLM grading
- `scrape_single_url()` - Handles individual URL scraping with ScrapingBee
- `get_domain_urls()` - URL discovery using Firecrawl mapping endpoint

### Processing Flow

1. Domain URL discovery via Firecrawl `/map` endpoint with optional search filtering
2. Concurrent scraping using aiohttp and ScrapingBee API (configurable workers)
3. Batch LLM grading using structured JSON responses from GPT-4o Mini
4. Real-time CSV output with comprehensive statistics and visualizations

## Common Commands

### Running the Application

```bash
python real_time_seo_grader.py
```

The script runs interactively and prompts for:
- Domain to analyze
- Optional search filter for URL discovery  
- Target number of results to grade

### Environment Setup

Ensure `.env` file contains required API keys:
```
SCRAPINGBEE_API_KEY=your_key_here
FIRECRAWL_KEY=your_key_here  
OPENAI_KEY=your_key_here
```

### Performance Configuration

Key constants in `real_time_seo_grader.py`:
- `MAX_CONCURRENT = 48` - ScrapingBee concurrent workers
- `MAX_LLM_CONCURRENT = 200` - OpenAI concurrent requests
- `BATCH_SIZE = 25` - Pages per LLM API call for batch processing
- `MAX_URLS_TO_MAP = 10000` - Maximum URLs to discover via Firecrawl

## Output Files

Results are saved to timestamped files in `output/`:
- `{domain}_{timestamp}_seo_graded.csv` - Main results with scores
- `{domain}_{timestamp}_domain_path_analysis.json` - URL analysis (if generated)

## Dependencies

The application requires Python 3.7+ with these key packages:
- `aiohttp` - Async HTTP client for concurrent scraping
- `openai` - OpenAI API client for LLM grading
- `requests` - HTTP client for Firecrawl API calls

## Known Issues

- Windows console encoding may cause issues with emoji characters in output
- Rate limiting can occur with high concurrent settings depending on API plans
- Some domains may have limited URL discovery due to sitemap restrictions