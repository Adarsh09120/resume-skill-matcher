"""
Resume Skill Matcher
Built with Python + scikit-learn
Uses TF-IDF and cosine similarity to match resumes with job descriptions
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# STEP 1: Sample data (replace with your actual resume)
# ============================================

# Your resume skills (what you actually know)
YOUR_RESUME = """
Python programming, loops, functions, lists, dictionaries
Basic understanding of Machine Learning
Pandas, NumPy (learning)
Team coordination, event volunteering, peer learning
"""

# Alovra AI job description (from their JD)
JOB_DESCRIPTION = """
Python & AI Developer
Required skills:
- Strong proficiency in Python
- Solid understanding of machine learning and model training
- Ability to lead and manage junior team members
- Working English fluency
Preferred:
- TensorFlow or PyTorch
- Startup experience
- Marketing or client outreach
"""

# ============================================
# STEP 2: Function to calculate match percentage
# ============================================

def calculate_match_percentage(resume_text, job_text):
    """
    Calculates how well a resume matches a job description
    Returns match percentage (0-100)
    """
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Combine both texts and transform to vectors
    all_texts = [resume_text, job_text]
    vectors = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:2])
    match_percentage = round(similarity_matrix[0][0] * 100, 2)
    
    return match_percentage

# ============================================
# STEP 3: Function to find missing skills
# ============================================

def find_missing_skills(resume_text, job_text):
    """
    Identifies skills mentioned in job description but missing from resume
    """
    # Simple word comparison
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    
    # Remove common words
    common_words = {'and', 'or', 'the', 'a', 'an', 'to', 'of', 'for', 'in', 'on', 'at', 'with'}
    
    resume_words = resume_words - common_words
    job_words = job_words - common_words
    
    missing = job_words - resume_words
    
    # Filter to only relevant terms (at least 3 characters)
    missing = [word for word in missing if len(word) > 3]
    
    return list(missing)[:10]  # Return top 10 missing skills

# ============================================
# STEP 4: Function to process multiple resumes (batch mode)
# ============================================

def process_multiple_resumes(resumes_dict, job_description):
    """
    Processes multiple resumes in batch mode
    resumes_dict: {'candidate_name': 'resume_text', ...}
    """
    results = []
    
    for candidate, resume in resumes_dict.items():
        match = calculate_match_percentage(resume, job_description)
        missing = find_missing_skills(resume, job_description)
        
        results.append({
            'Candidate': candidate,
            'Match %': match,
            'Missing Skills': ', '.join(missing[:5])
        })
    
    return pd.DataFrame(results)

# ============================================
# STEP 5: Main program
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📊 RESUME SKILL MATCHER")
    print("=" * 60)
    
    # Calculate your match
    print("\n📄 Analyzing YOUR resume against Alovra AI job...\n")
    
    match = calculate_match_percentage(YOUR_RESUME, JOB_DESCRIPTION)
    missing = find_missing_skills(YOUR_RESUME, JOB_DESCRIPTION)
    
    print(f"🎯 Match Percentage: {match}%")
    print(f"\n📋 Missing Skills (work on these):")
    for skill in missing[:7]:
        print(f"   • {skill}")
    
    # Batch mode example
    print("\n" + "=" * 60)
    print("🔄 BATCH MODE - Processing multiple resumes")
    print("=" * 60)
    
    # Example: Comparing with other candidates
    all_resumes = {
        "Your Resume": YOUR_RESUME,
        "Candidate A": "Python, Java, SQL, Spring Boot, AWS",
        "Candidate B": "Python, TensorFlow, PyTorch, ML, Deep Learning",
        "Candidate C": "JavaScript, React, Node.js, MongoDB",
    }
    
    batch_results = process_multiple_resumes(all_resumes, JOB_DESCRIPTION)
    print("\n" + batch_results.to_string())
    
    # Export to CSV
    export_option = input("\n\n💾 Export results to CSV? (yes/no): ").lower()
    if export_option == "yes":
        batch_results.to_csv("resume_matching_results.csv", index=False)
        print("✅ Results saved to: resume_matching_results.csv")
    
    print("\n✅ Analysis complete!")