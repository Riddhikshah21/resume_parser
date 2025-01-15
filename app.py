from typing import Dict, List
import spacy
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from transformers import pipeline

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        doc = docx.Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)

    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {
            'education': '',
            'work_experience': '',
            'skills': ''
        }
        
        # Simple section extraction based on common headers
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ['education', 'academic']):
                current_section = 'education'
            elif any(keyword in lower_line for keyword in ['experience', 'employment']):
                current_section = 'work_experience'
            elif any(keyword in lower_line for keyword in ['skills', 'technologies']):
                current_section = 'skills'
            elif current_section:
                sections[current_section] += line + '\n'
                
        return sections

    def extract_skills(self, text: str) -> List[str]:
        doc = self.nlp(text)
        skills = []
        
        # Custom skill patterns (expand this list based on domain)
        skill_patterns = ['python', 'java', 'sql', 'machine learning', 'data analysis']
        
        for token in doc:
            if token.text.lower() in skill_patterns:
                skills.append(token.text)
                
        return list(set(skills))

    def match_job_requirements(self, resume_text: str, job_description: str) -> Dict:
        # Extract skills from both resume and job description
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_description))
        
        matching_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'match_percentage': len(matching_skills) / len(job_skills) * 100 if job_skills else 0
        }

    def improve_bullet_points(self, experience_text: str, job_description: str) -> List[str]:
        # Extract current bullet points
        bullet_points = re.findall(r'•(.*?)(?=•|\n|$)', experience_text)
        
        improved_points = []
        for point in bullet_points:
            # Calculate relevance to job description
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([point, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            if similarity > 0.1:  # Threshold for relevance
                # Summarize and improve the bullet point
                improved = self.summarizer(point.strip(), 
                                        max_length=50, 
                                        min_length=20, 
                                        do_sample=False)[0]['summary_text']
                improved_points.append(improved)
                
        return improved_points

def main():
    parser = ResumeParser()
    
    # Example usage
    resume_text = parser.extract_text_from_pdf("Riddhi_Shah.pdf")
    job_description = """
    We are looking for a Python developer with experience in:
    - Machine Learning
    - SQL databases
    - Data Analysis
    Must have strong problem-solving skills and knowledge of software development practices.
    """
    
    # Extract sections
    sections = parser.extract_sections(resume_text)
    
    # Match requirements
    matches = parser.match_job_requirements(resume_text, job_description)
    
    # Improve bullet points
    improved_points = parser.improve_bullet_points(
        sections['work_experience'], 
        job_description
    )
    
    print(f"Match Percentage: {matches['match_percentage']}%")
    print(f"Missing Skills: {', '.join(matches['missing_skills'])}")
    print("\nImproved Bullet Points:")
    for point in improved_points:
        print(f"• {point}")

if __name__ == "__main__":
    main()