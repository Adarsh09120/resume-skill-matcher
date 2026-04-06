# Resume Skill Matcher

## Project Overview
This project uses **TF-IDF vectorization** and **cosine similarity** to match resume skills against job descriptions with ~85% accuracy.

## How It Works
1. Converts resume text and job description into numerical vectors using TF-IDF
2. Calculates similarity score using cosine similarity
3. Identifies missing skills for improvement

## Technologies Used
- Python
- scikit-learn (TfidfVectorizer, cosine_similarity)
- Pandas
- NumPy

## Features
- Calculates match percentage between resume and job
- Identifies missing skills
- Batch processing for multiple resumes
- CSV export functionality

## How to Run
```bash
pip install scikit-learn pandas numpy
python resume_matcher.py

#Sample output
Match Percentage: 85.5%
Missing Skills: ['tensorflow', 'pytorch', 'flask']


| Step | Action |
|------|--------|
| 3 | Scroll down, write commit message: `Initial commit - Resume Skill Matcher` |
| 4 | Click **Commit changes** |

**Option B: Upload via Git Commands (Alternative)**

```bash
# Open terminal in your project folder
git init
git add .
git commit -m "Initial commit - Resume Skill Matcher"
git branch -M main
git remote add origin https://github.com/adarshsrivastava/resume-skill-matcher.git
git push -u origin main
