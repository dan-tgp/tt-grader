#!/usr/bin/env python3
"""
SEO Results Output Formatter
Handles the display of SEO analysis results in a clean, organized format.
"""

import statistics
from typing import Dict, List
from dataclasses import dataclass

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

def display_enhanced_statistics(all_scores: Dict[str, List[int]], score_distributions: Dict, char_lengths: Dict, all_results: List[PageResult]):
    """Display simplified statistics panel with only the 3 requested sections"""
    print()
    print("📊 ENHANCED SEO STATISTICS PANEL")
    print("=" * 100)
    
    def calc_stats(scores):
        """Calculate min, max, avg, median for a list of scores"""
        if not scores:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        return {
            'min': min(scores),
            'max': max(scores),
            'avg': statistics.mean(scores),
            'median': statistics.median(scores)
        }
    
    def calc_length_stats(lengths):
        """Calculate length statistics"""
        if not lengths:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        return {
            'min': min(lengths),
            'max': max(lengths),
            'avg': statistics.mean(lengths),
            'median': statistics.median(lengths)
        }
    
    # 1. BASIC SCORE STATISTICS
    print("🎯 SCORE OVERVIEW")
    print("-" * 80)
    print(f"{'Score Type':<20} {'Min':<5} {'Max':<5} {'Avg':<6} {'Median':<7} {'Count'}")
    print("-" * 60)
    
    score_stats = {
        'Title Scores': calc_stats(all_scores['title_scores']),
        'Description Scores': calc_stats(all_scores['description_scores']), 
        'H1 Scores': calc_stats(all_scores['h1_scores']),
        'Overall Scores': calc_stats(all_scores['overall_scores'])
    }
    
    for score_type, stat_values in score_stats.items():
        count = len(all_scores[score_type.lower().replace(' scores', '_scores')])
        if count > 0:
            print(f"{score_type:<20} {stat_values['min']:<5.0f} {stat_values['max']:<5.0f} {stat_values['avg']:<6.1f} {stat_values['median']:<7.1f} {count:<6}")
        else:
            print(f"{score_type:<20} {'N/A':<5} {'N/A':<5} {'N/A':<6} {'N/A':<7} {count:<6}")
    
    # 2. SCORE DISTRIBUTION VISUALIZATIONS
    print("\n📈 SCORE DISTRIBUTION VISUALIZATIONS")
    print("    Rubric: 🔴 FAIL (1-3) | 🟡 NEEDS IMPROVEMENT (4-7) | 🟢 GOOD (8-10)")
    print("=" * 80)
    
    def create_distribution_chart(distribution, element_name, max_width=40):
        """Create ASCII bar chart for score distribution"""
        # Skip score 0 (failed grading) for cleaner visualization
        scores_1_to_10 = {k: v for k, v in distribution.items() if k >= 1}
        
        if not any(scores_1_to_10.values()):
            print(f"\n{element_name}: No data to display\n")
            return
        
        max_count = max(scores_1_to_10.values()) if scores_1_to_10.values() else 1
        scale_factor = max_width / max_count if max_count > 0 else 1
        
        print(f"\n📊 {element_name.upper()} SCORE DISTRIBUTION")
        print(f"{'Score':<8} {'Count':<6} {'Bar':<{max_width+2}} {'%':<6}")
        print("-" * (8 + 6 + max_width + 8))
        
        total_scores = sum(scores_1_to_10.values())
        
        for score in range(1, 11):
            count = scores_1_to_10.get(score, 0)
            bar_length = int(count * scale_factor) if count > 0 else 0
            percentage = round(count / total_scores * 100, 1) if total_scores > 0 else 0
            
            # Color coding based on rubric
            if score <= 3:
                score_label = f"{score} 🔴"
            elif score <= 7:
                score_label = f"{score} 🟡"
            else:
                score_label = f"{score} 🟢"
            
            # Create bar with block characters
            bar = "█" * bar_length if bar_length > 0 else ""
            print(f"{score_label:<8} {count:<6} {bar:<{max_width}} {percentage}%")
    
    # Create visualizations for each element type
    for element_type, distribution in [
        ("Titles", score_distributions['title_distribution']),
        ("Descriptions", score_distributions['description_distribution']), 
        ("H1 Tags", score_distributions['h1_distribution'])
    ]:
        create_distribution_chart(distribution, element_type)
    
    # 3. QUICK REFERENCE TABLE
    print("\n📋 QUICK REFERENCE TABLE")
    print("-" * 66)
    print(f"{'Element':<15} ", end="")
    for i in range(0, 11):
        print(f"{'S' + str(i):<4}", end="")
    print()
    print("-" * 66)
    
    distribution_mapping = {
        'Titles': score_distributions['title_distribution'],
        'Descriptions': score_distributions['description_distribution'],
        'H1 Tags': score_distributions['h1_distribution'],
        'Overall': score_distributions['overall_distribution']
    }
    
    for element_name, distribution in distribution_mapping.items():
        print(f"{element_name:<15} ", end="")
        for score in range(0, 11):
            count = distribution.get(score, 0)
            print(f"{count:<4}", end="")
        print()
    
    # 4. CHARACTER LENGTH ANALYSIS
    print("\n📏 CHARACTER LENGTH ANALYSIS")
    print("-" * 80)
    
    length_stats = {
        'Title Lengths': calc_length_stats(char_lengths['title_lengths']),
        'Description Lengths': calc_length_stats(char_lengths['description_lengths']),
        'H1 Lengths': calc_length_stats(char_lengths['h1_lengths'])
    }
    
    ideal_ranges = {
        'Title Lengths': '50-60 chars',
        'Description Lengths': '150-160 chars', 
        'H1 Lengths': '20-70 chars'
    }
    
    print(f"{'Element':<20} {'Min':<5} {'Max':<5} {'Avg':<6} {'Median':<7} {'Ideal Range':<12}")
    print("-" * 68)
    
    for element_type, length_stat in length_stats.items():
        if char_lengths[element_type.lower().replace(' lengths', '_lengths')]:
            ideal = ideal_ranges[element_type]
            print(f"{element_type:<20} {length_stat['min']:<5.0f} {length_stat['max']:<5.0f} {length_stat['avg']:<6.1f} {length_stat['median']:<7.1f} {ideal:<12}")
        else:
            print(f"{element_type:<20} {'N/A':<5} {'N/A':<5} {'N/A':<6} {'N/A':<7} {'N/A':<12}")

    print("=" * 100)