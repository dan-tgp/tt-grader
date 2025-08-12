# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based SEO analysis tool that scrapes websites and provides LLM-powered grading of SEO elements. The system uses:

- ScrapingBee API for web scraping
- Firecrawl API for URL discovery via domain mapping  
- OpenAI GPT-5 Nano for SEO scoring and analysis (ultra cost-effective, 67% cheaper than GPT-4o Mini)
- Async/concurrent processing for high-performance scraping

## Core Architecture

### Main Components

- `web_app.py` - Flask web application with interactive dashboard
- `tt_grader_with_db.py` - Core SEO grader with database storage
- `templates/` - HTML templates for web interface
- `static/` - Static assets (images, CSS)
- `output/` - Directory containing CSV results and analysis files
- `.env` - API keys configuration (ScrapingBee, Firecrawl, OpenAI)

### Key Classes and Functions

- `PageResult` dataclass - Structured data for page scraping and grading results
- `RealTimeCSVWriter` - Streams results to CSV as they are processed
- `batch_grade()` - Batches multiple pages for efficient GPT-5 Nano grading
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
python web_app.py
```

Access the web interface at http://localhost:5000

The application provides:
- Interactive web dashboard for SEO audits
- Real-time progress tracking
- Shareable report URLs
- CSV export functionality

### Environment Setup

Ensure `.env` file contains required API keys:
```
SCRAPINGBEE_API_KEY=your_key_here
FIRECRAWL_KEY=your_key_here  
OPENAI_KEY=your_key_here
```

### Performance Configuration

Key constants in `tt_grader_with_db.py`:
- `MAX_CONCURRENT_SCRAPING = 50` - ScrapingBee concurrent workers
- `MAX_CONCURRENT_LLM = 100` - OpenAI concurrent requests
- `BATCH_SIZE = 25` - Pages per LLM API call for batch processing

### Cost Optimization with GPT-5 Nano

The application uses GPT-5 Nano for optimal cost-performance:
- **Pricing**: $0.05/1M input tokens, $0.40/1M output tokens
- **Cost per 500-page audit**: ~$0.19 (vs $0.96 with GPT-5 Mini)
- **Performance**: Better than GPT-4o Mini with ~45% fewer errors
- **Speed**: Ultra-fast response times ideal for high-volume processing

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