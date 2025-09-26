import os
import re
import time
import math
from collections import Counter
from difflib import SequenceMatcher
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

 
def remove_bracketed_segments(text):
    """Remove content enclosed in square brackets like [speaker_tag] from text.
    Preserves line breaks and trims extra spaces on each line."""
    # Remove [ ... ] segments
    without_tags = re.sub(r"\[[^\]]*\]", "", text)
    # Collapse horizontal spaces but keep newlines intact
    without_tags = re.sub(r"[ \t]+", " ", without_tags)
    # Trim spaces per line to avoid leading/trailing blanks
    cleaned_lines = [line.strip() for line in without_tags.splitlines()]
    return "\n".join(cleaned_lines)

def normalize_text(text):
    """Clean and normalize text for comparison"""
    # Remove extra whitespace, punctuation, and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.strip().lower())
    text = re.sub(r'\s+', ' ', text)
    words = [word for word in text.split() if word]
    return words, text

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two word frequency vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    # Create word frequency vectors
    all_words = set(vec1.keys()).union(set(vec2.keys()))
    vec1_norm = sum(vec1.get(word, 0) ** 2 for word in all_words) ** 0.5
    vec2_norm = sum(vec2.get(word, 0) ** 2 for word in all_words) ** 0.5
    
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0
    
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
    return dot_product / (vec1_norm * vec2_norm)

def word_order_similarity(words1, words2):
    """Calculate word order similarity between two word lists"""
    if not words1 or not words2:
        return 0.0
    
    # Find common words and their positions
    common_words = set(words1).intersection(set(words2))
    if not common_words:
        return 0.0
    
    # Get positions of common words
    pos1 = {word: [i for i, w in enumerate(words1) if w == word] for word in common_words}
    pos2 = {word: [i for i, w in enumerate(words2) if w == word] for word in common_words}
    
    # Calculate position differences
    total_diff = 0
    total_pairs = 0
    
    for word in common_words:
        positions1 = pos1[word]
        positions2 = pos2[word]
        
        # Use minimum number of occurrences
        min_count = min(len(positions1), len(positions2))
        for i in range(min_count):
            diff = abs(positions1[i] - positions2[i])
            total_diff += diff
            total_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    # Normalize by text length
    max_len = max(len(words1), len(words2))
    avg_diff = total_diff / total_pairs
    order_similarity = max(0, 1 - (avg_diff / max_len))
    
    return order_similarity

def frequency_similarity(words1, words2):
    """Calculate frequency-based similarity between two word lists"""
    if not words1 and not words2:
        return 1.0
    
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Calculate intersection and union
    intersection = sum((freq1 & freq2).values())
    union = sum((freq1 | freq2).values())
    
    return intersection / union if union > 0 else 0.0

def fast_edit_distance_similarity(text1, text2):
    """Fast edit distance similarity using character-level comparison"""
    if not text1 and not text2:
        return 1.0
    
    # Use SequenceMatcher for faster edit distance approximation
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity

def fast_semantic_similarity(words1, words2):
    """Fast semantic similarity based on word overlap and length weighting"""
    if not words1 or not words2:
        return 0.0
    
    # Create word frequency vectors
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Calculate weighted overlap (simplified)
    common_words = set(freq1.keys()).intersection(set(freq2.keys()))
    if not common_words:
        return 0.0
    
    # Simple weighted calculation
    total_weight = sum(freq1.values()) + sum(freq2.values())
    overlap_weight = sum(min(freq1[word], freq2[word]) for word in common_words)
    
    return overlap_weight / total_weight if total_weight > 0 else 0.0

def fast_structure_similarity(text1, text2):
    """Fast structure similarity based on text length and word count"""
    if not text1 and not text2:
        return 1.0
    
    # Simple length-based similarity
    len1, len2 = len(text1), len(text2)
    words1, words2 = len(text1.split()), len(text2.split())
    
    # Calculate similarities
    length_sim = 1 - abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    word_sim = 1 - abs(words1 - words2) / max(words1, words2) if max(words1, words2) > 0 else 0
    
    return (length_sim + word_sim) / 2

def fast_sequence_similarity(text1, text2):
    """Fast sequence similarity using word-level comparison"""
    words1, clean_text1 = normalize_text(text1)
    words2, clean_text2 = normalize_text(text2)
    
    # Use word-level similarity for speed
    return SequenceMatcher(None, words1, words2).ratio()

def positional_word_similarity(text1, text2, tolerance=None):
    """Calculate similarity based on words at similar positions with improved algorithm"""
    words1, clean_text1 = normalize_text(text1)
    words2, clean_text2 = normalize_text(text2)
    
    if not words1 or not words2:
        return 0.0
    
    # Find the minimum length to compare
    min_length = min(len(words1), len(words2))
    if min_length == 0:
        return 0.0
    
    # Calculate dynamic tolerance based on average sentence length
    if tolerance is None:
        # Use average words per sentence (roughly 15-20 words)
        # For shorter texts, use smaller tolerance; for longer texts, use larger tolerance
        avg_words_per_sentence = 15
        dynamic_tolerance = max(2, min(6, min_length // avg_words_per_sentence + 1))
        tolerance = dynamic_tolerance
    
    # Improved algorithm: Use sliding window with weighted scoring
    total_score = 0.0
    matched_positions = set()  # Track used positions in text2
    
    for i, word1 in enumerate(words1[:min_length]):
        best_match_score = 0.0
        best_match_pos = -1
        
        # Look for word1 in text2 within tolerance range
        start_pos = max(0, i - tolerance)
        end_pos = min(len(words2), i + tolerance + 1)
        
        # Check each position in the tolerance range
        for j in range(start_pos, end_pos):
            if j not in matched_positions and j < len(words2) and words2[j] == word1:
                # Calculate position-based score (closer positions get higher scores)
                distance = abs(i - j)
                position_score = 1.0 - (distance / (tolerance + 1))
                
                # Additional bonus for exact position match
                if distance == 0:
                    position_score += 0.2
                
                if position_score > best_match_score:
                    best_match_score = position_score
                    best_match_pos = j
        
        # If we found a match, add it to our score and mark position as used
        if best_match_pos != -1:
            total_score += best_match_score
            matched_positions.add(best_match_pos)
    
    # Calculate final positional similarity
    positional_score = total_score / min_length if min_length > 0 else 0.0
    
    # Consider length similarity (penalty for different lengths)
    length_ratio = min_length / max(len(words1), len(words2))
    
    # Additional bonus for overall word order preservation
    order_bonus = 0.0
    if len(matched_positions) > 1:
        # Check if matched positions maintain relative order
        sorted_positions = sorted(matched_positions)
        order_consistency = 0
        for k in range(len(sorted_positions) - 1):
            if sorted_positions[k] < sorted_positions[k + 1]:
                order_consistency += 1
        order_bonus = (order_consistency / (len(sorted_positions) - 1)) * 0.1
    
    # Combine all factors
    final_score = (positional_score + order_bonus) * length_ratio
    
    return min(1.0, final_score)

def compare_text_similarity_advanced(text1, text2, detailed=False):
    """
    Advanced text similarity comparison using multiple algorithms.
    Improved for speech-to-text evaluation.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        float or dict: Similarity score between 0.0 and 1.0, or detailed analysis if detailed=True
    """
    # Clean and normalize texts
    words1, clean_text1 = normalize_text(text1)
    words2, clean_text2 = normalize_text(text2)
    
    # Calculate individual similarity scores
    word_set1 = set(words1)
    word_set2 = set(words2)
    
    # Fast similarity metrics
    jaccard_score = jaccard_similarity(word_set1, word_set2)
    word_freq1 = Counter(words1)
    word_freq2 = Counter(words2)
    cosine_score = cosine_similarity(word_freq1, word_freq2)
    
    # Fast additional metrics
    frequency_score = frequency_similarity(words1, words2)
    semantic_score = fast_semantic_similarity(words1, words2)
    edit_score = fast_edit_distance_similarity(clean_text1, clean_text2)
    structure_score = fast_structure_similarity(text1, text2)
    seq_score = fast_sequence_similarity(text1, text2)
    positional_score = positional_word_similarity(text1, text2)
    
    # Fast weighted combined score for speech-to-text
    combined_score = (
        cosine_score * 0.25 +        # Word frequency similarity (most important)
        jaccard_score * 0.20 +       # Set-based similarity
        semantic_score * 0.20 +      # Fast semantic similarity
        positional_score * 0.15 +    # Positional word similarity (new!)
        seq_score * 0.10 +           # Fast sequence similarity
        frequency_score * 0.10       # Frequency similarity
    )
    
    # Ensure the score is between 0.0 and 1.0
    combined_score = max(0.0, min(1.0, combined_score))
    
    if detailed:
        analysis = {
            'combined_score': round(combined_score, 4),
            'individual_scores': {
                'cosine_similarity': round(cosine_score, 4),
                'jaccard_similarity': round(jaccard_score, 4),
                'fast_semantic_similarity': round(semantic_score, 4),
                'positional_word_similarity': round(positional_score, 4),
                'fast_sequence_similarity': round(seq_score, 4),
                'frequency_similarity': round(frequency_score, 4)
            },
            'text_analysis': {
                'text1_length': len(words1),
                'text2_length': len(words2),
                'common_words': len(word_set1.intersection(word_set2)),
                'unique_words_text1': len(word_set1),
                'unique_words_text2': len(word_set2),
                'word_overlap_ratio': len(word_set1.intersection(word_set2)) / max(len(word_set1), len(word_set2)) if max(len(word_set1), len(word_set2)) > 0 else 0
            }
        }
        return analysis
    
    return round(combined_score, 4)

def compare_text_similarity(text1, text2):
    """
    Simple text similarity comparison - returns combined score only
    """
    return compare_text_similarity_advanced(text1, text2, detailed=False)

def compare_files(file1, file2, detailed=False):
    """
    Compare two text files and return similarity score
    
    Args:
        file1 (str): Path to first text file
        file2 (str): Path to second text file
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        float or dict: Similarity score or detailed analysis
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            text1 = f.read().strip()
            # Skip [ ... ] speaker/stage tags in the original script file
            text1 = remove_bracketed_segments(text1)
        
        with open(file2, 'r', encoding='utf-8') as f:
            text2 = f.read().strip()
        
        return compare_text_similarity_advanced(text1, text2, detailed=detailed)
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

def get_similarity_quality(score):
    """Get quality assessment based on similarity score"""
    if score >= 0.9:
        return "EXCELLENT"
    elif score >= 0.8:
        return "GOOD"
    elif score >= 0.6:
        return "FAIR"
    else:
        return "POOR"

def explain_similarity_score(score):
    """Get easy-to-understand explanation of similarity score"""
    if score >= 0.8:
        return "🎯 Almost perfect! The transcription captures almost everything correctly."
    elif score >= 0.7:
        return "✅ Very good! Most words and meaning are preserved accurately."
    else:
        return "❌ Poor quality. Many words are different or missing, making it hard to understand."

def explain_individual_metrics():
    """Return explanations for each similarity metric"""
    return {
        'cosine_similarity': "📊 Word Frequency Match: How well the same words appear with similar frequency",
        'jaccard_similarity': "🔗 Word Set Overlap: How many unique words are shared between texts",
        'fast_semantic_similarity': "🧠 Meaning Similarity: How well the overall meaning is preserved",
        'positional_word_similarity': "📍 Word Position: How well words appear in the same order (with flexibility)",
        'fast_sequence_similarity': "📝 Word Sequence: How well the word-by-word flow matches",
        'frequency_similarity': "📈 Word Count Match: How similar the word frequency patterns are"
    }

def main():
    """Main function for standalone similarity comparison"""
    print("Text Similarity Analysis Tool")
    print("=" * 50)
    
    # Check if 1.story.txt and story.str.txt exist for comparison
    original_file = "../input/1.story.txt"
    transcribed_file = "../input/2.story.str.txt"
    
    if os.path.exists(original_file) and os.path.exists(transcribed_file):
        print(f"Found files to compare:")
        print(f"  • {original_file}")
        print(f"  • {transcribed_file}")
        print()
        
        start_time = time.time()
        
        # Perform detailed comparison
        result = compare_files(original_file, transcribed_file, detailed=True)
        
        if result:
            print("📊 DETAILED SIMILARITY ANALYSIS")
            print("=" * 50)
            
            # Combined score with explanation

            
            # Individual scores with explanations
            print("📈 Individual Algorithm Scores:")
            metric_explanations = explain_individual_metrics()
            for algorithm, score in result['individual_scores'].items():
                explanation = metric_explanations.get(algorithm, "Unknown metric")
                print(f"  • {algorithm.replace('_', ' ').title()}: {score:.3f}")
                print(f"    {explanation}")
            print()
            
            # Text analysis
            analysis = result['text_analysis']
            print("📝 Text Analysis:")
            print(f"  • Original text words: {analysis['text1_length']}")
            print(f"  • Transcribed text words: {analysis['text2_length']}")
            print(f"  • Common words: {analysis['common_words']}")
            print(f"  • Unique words (original): {analysis['unique_words_text1']}")
            print(f"  • Unique words (transcribed): {analysis['unique_words_text2']}")
            print(f"  • Word overlap ratio: {analysis['word_overlap_ratio']:.3f}")
            print()
            
            # Quality assessment (now redundant since we have explanation above)
            combined_score = result['combined_score']
            quality = get_similarity_quality(combined_score)
            explanation = explain_similarity_score(combined_score)
            print(f"🎯 Combined Similarity Score: {combined_score:.3f} ({quality})")
            print(f"💡 {explanation}")
            print()
            
            end_time = time.time()
            print(f"\n⏱️  Analysis completed in {end_time - start_time:.3f} seconds")
        
    else:
        print("No files found for comparison.")
        print("Expected files:")
        print(f"  • {original_file}")
        print(f"  • {transcribed_file}")
        print()
        print("You can also use the functions directly:")
        print("  • compare_text_similarity(text1, text2)")
        print("  • compare_text_similarity_advanced(text1, text2, detailed=True)")
        print("  • compare_files(file1, file2, detailed=True)")

if __name__ == "__main__":
    main()
