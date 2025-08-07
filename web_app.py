#!/usr/bin/env python3
"""
TT-Grader Web Frontend - Share SEO analysis results via unique URLs
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import sqlite3
import hashlib
import json
import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import secrets

# Import the grader functionality
from tt_grader_with_db import run_grader, DatabaseManager

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

DATABASE_PATH = "tt_grader.db"

def generate_hash_for_run(run_id: int) -> str:
    """Generate a unique hash for a run ID"""
    # Create a hash from run_id + secret salt
    salt = "tt-grader-2024"
    hash_input = f"{run_id}-{salt}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

def get_run_by_hash(hash_id: str) -> Optional[Dict[str, Any]]:
    """Get run data by hash"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Try to find run with this hash
    cursor.execute('SELECT id FROM runs')
    runs = cursor.fetchall()
    
    for run in runs:
        if generate_hash_for_run(run['id']) == hash_id:
            # Get full run data
            cursor.execute('''
                SELECT r.*, 
                       COUNT(res.id) as result_count,
                       AVG(res.overall_score) as avg_overall,
                       AVG(res.title_score) as avg_title,
                       AVG(res.description_score) as avg_description,
                       AVG(res.h1_score) as avg_h1,
                       COUNT(CASE WHEN res.overall_score >= 8 THEN 1 END) as excellent_count,
                       COUNT(CASE WHEN res.overall_score >= 5 AND res.overall_score <= 7 THEN 1 END) as good_count,
                       COUNT(CASE WHEN res.overall_score <= 4 THEN 1 END) as needs_work_count
                FROM runs r
                LEFT JOIN results res ON r.id = res.run_id AND res.status = 'success'
                WHERE r.id = ?
                GROUP BY r.id
            ''', (run['id'],))
            
            run_data = dict(cursor.fetchone())
            run_data['hash'] = hash_id
            
            # Get score distributions for all elements
            distributions = {}
            
            for element in ['title_score', 'description_score', 'h1_score', 'overall_score']:
                cursor.execute(f'''
                    SELECT {element}, COUNT(*) as count
                    FROM results 
                    WHERE run_id = ? AND status = 'success' AND {element} IS NOT NULL
                    GROUP BY {element}
                    ORDER BY {element}
                ''', (run['id'],))
                
                element_dist = {}
                for row in cursor.fetchall():
                    element_dist[row[0]] = row[1]
                
                distributions[element] = element_dist
            
            run_data['score_distributions'] = distributions
            
            conn.close()
            return run_data
    
    conn.close()
    return None

def get_run_results(run_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    """Get results for a specific run"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM results 
        WHERE run_id = ? AND status = 'success'
        ORDER BY overall_score ASC
        LIMIT ?
    ''', (run_id, limit))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent runs with hashes and all score averages"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT r.*, 
               COUNT(res.id) as result_count,
               AVG(res.overall_score) as avg_overall,
               AVG(res.title_score) as avg_title,
               AVG(res.description_score) as avg_description,
               AVG(res.h1_score) as avg_h1
        FROM runs r
        LEFT JOIN results res ON r.id = res.run_id AND res.status = 'success'
        GROUP BY r.id
        ORDER BY r.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    runs = []
    for row in cursor.fetchall():
        run = dict(row)
        run['hash'] = generate_hash_for_run(run['id'])
        runs.append(run)
    
    conn.close()
    return runs

def get_overall_statistics():
    """Get overall statistics across all successful results"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get overall averages
    cursor.execute('''
        SELECT 
            COUNT(*) as total_pages,
            AVG(title_score) as avg_title,
            AVG(description_score) as avg_description,
            AVG(h1_score) as avg_h1,
            AVG(overall_score) as avg_overall
        FROM results 
        WHERE status = 'success'
    ''')
    
    stats = dict(cursor.fetchone())
    
    # Get score distributions
    cursor.execute('''
        SELECT 
            title_score,
            description_score,
            h1_score,
            overall_score
        FROM results 
        WHERE status = 'success'
    ''')
    
    scores = cursor.fetchall()
    
    # Calculate distributions
    distributions = {
        'title': {'1-3': 0, '4-6': 0, '7-10': 0},
        'description': {'1-3': 0, '4-6': 0, '7-10': 0},
        'h1': {'1-3': 0, '4-6': 0, '7-10': 0},
    }
    
    for score in scores:
        for score_type in ['title', 'description', 'h1']:
            score_val = score[f'{score_type}_score']
            if score_val and 1 <= score_val <= 10:
                if score_val <= 3:
                    distributions[score_type]['1-3'] += 1
                elif score_val <= 6:
                    distributions[score_type]['4-6'] += 1
                else:
                    distributions[score_type]['7-10'] += 1
    
    conn.close()
    return stats, distributions

def get_best_worst_examples():
    """Get top 10 best and worst examples for each SEO element with content"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    examples = {}
    
    for element in ['title', 'description', 'h1']:
        # Get top 10 best - use content if available, fallback to original field
        cursor.execute(f'''
            SELECT url, {element}_score as score, 
                   COALESCE({element}_content, {element}) as content
            FROM results 
            WHERE status = 'success' AND {element}_score IS NOT NULL
            ORDER BY {element}_score DESC, url ASC
            LIMIT 10
        ''')
        best = [dict(row) for row in cursor.fetchall()]
        
        # Get top 10 worst - use content if available, fallback to original field  
        cursor.execute(f'''
            SELECT url, {element}_score as score,
                   COALESCE({element}_content, {element}) as content
            FROM results 
            WHERE status = 'success' AND {element}_score IS NOT NULL
            ORDER BY {element}_score ASC, url ASC
            LIMIT 10
        ''')
        worst = [dict(row) for row in cursor.fetchall()]
        
        examples[element] = {'best': best, 'worst': worst}
    
    conn.close()
    return examples

def get_optimization_recommendations():
    """Get count of pages that could benefit from TopTitle optimization"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) as total_pages,
               COUNT(CASE WHEN title_score <= 7 OR description_score <= 7 OR h1_score <= 7 THEN 1 END) as optimization_candidates
        FROM results 
        WHERE status = 'success'
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    return {
        'total_pages': result[0] if result else 0,
        'optimization_candidates': result[1] if result else 0
    }

@app.route('/')
def index():
    """Homepage with recent runs and overall statistics"""
    runs = get_recent_runs(20)
    
    # Get overall statistics if we have data
    stats = None
    distributions = None
    examples = None
    recommendations = None
    
    if runs:  # Only calculate if we have runs
        try:
            stats, distributions = get_overall_statistics()
            examples = get_best_worst_examples()
            recommendations = get_optimization_recommendations()
        except Exception as e:
            print(f"Error getting statistics: {e}")
            stats = distributions = examples = recommendations = None
    
    return render_template('index.html', 
                         runs=runs, 
                         stats=stats, 
                         distributions=distributions, 
                         examples=examples,
                         recommendations=recommendations)

@app.route('/grader/<hash_id>')
def view_results(hash_id):
    """View results for a specific run via hash"""
    run_data = get_run_by_hash(hash_id)
    
    if not run_data:
        return "Run not found", 404
    
    # Get detailed results
    results = get_run_results(run_data['id'])
    
    # Calculate additional statistics
    if results:
        run_data['excellent_pct'] = (run_data['excellent_count'] / len(results) * 100) if results else 0
        run_data['good_pct'] = (run_data['good_count'] / len(results) * 100) if results else 0
        run_data['needs_work_pct'] = (run_data['needs_work_count'] / len(results) * 100) if results else 0
        
        # Get pages by tier
        run_data['critical_pages'] = [r for r in results if r['overall_score'] <= 4][:10]  # Critical (1-4)
        run_data['needs_improvement_pages'] = [r for r in results if 5 <= r['overall_score'] <= 7][:10]  # Needs Improvement (5-7)
        run_data['excellent_pages'] = sorted([r for r in results if r['overall_score'] >= 8], key=lambda x: x['overall_score'], reverse=True)[:5]  # Excellent (8-10)
        
        # Keep backwards compatibility
        run_data['worst_pages'] = run_data['critical_pages']
        run_data['best_pages'] = run_data['excellent_pages']
    
    return render_template('results.html', run=run_data, results=results)

@app.route('/api/run', methods=['POST'])
def api_run_analysis():
    """API endpoint to run new analysis"""
    data = request.json
    
    domain = data.get('domain')
    target_count = data.get('target_count', 100)
    path_filter = data.get('path_filter')
    
    if not domain:
        return jsonify({'error': 'Domain is required'}), 400
    
    # Run analysis asynchronously
    try:
        run_id = asyncio.run(run_grader(
            domain=domain,
            target_count=target_count,
            path_filter=path_filter
        ))
        
        # Generate hash for the new run
        hash_id = generate_hash_for_run(run_id)
        
        return jsonify({
            'success': True,
            'run_id': run_id,
            'hash': hash_id,
            'url': url_for('view_results', hash_id=hash_id, _external=True)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<hash_id>')
def api_get_results(hash_id):
    """API endpoint to get results as JSON"""
    run_data = get_run_by_hash(hash_id)
    
    if not run_data:
        return jsonify({'error': 'Run not found'}), 404
    
    results = get_run_results(run_data['id'])
    
    return jsonify({
        'run': run_data,
        'results': results
    })


@app.route('/new')
def new_analysis():
    """Form to create new analysis"""
    return render_template('new_analysis.html')

if __name__ == '__main__':
    # Ensure database exists
    db_manager = DatabaseManager()
    
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting TT-Grader Web Server...")
    print("Access at: http://localhost:5000")
    
    app.run(debug=True, port=5000, host='0.0.0.0')