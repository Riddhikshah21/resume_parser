import re
import os
import string
import nltk
from nltk.corpus import stopwords
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# def load_text(file_path):
#     """Loads text from a plain text file."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return f.read()
# class ResumeParser:
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    """Cleans the text by removing special characters, extra spaces, and stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\s+", " ", text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Example Usage
resume_text = extract_text_from_pdf("/Users/riddhishah/Documents/GitHub/resume_parser/Riddhi_Shah.pdf")  # Replace with actual resume file path
# job_description_text = extract_text_from_pdf("job_description.txt")  # Replace with job description file path
job_description_text = """
    Responsibilities

    Collaborate effectively with your squad of 3-5 members, including designers, developers and architects, in short sprints or kanban to achieve regular business and customer outcomes.
    Design and implement automations to reduce operational toil and increase team velocity.
    Understand and address the needs of fellow Trenders by designing both short-term and long-term solutions, delivering practical tools and processes.
    Develop enhanced tooling and metrics to improve our understanding of system health for both our team and our users along with enhancing team tools and practices for tailored outcomes.
    Participate in on-call rotations to support and debug both customer-reported issues and internal service incidents.
    Contribute to regular improvements, compliance audits, disaster recovery simulations, operational incidents, and blameless post-incident reviews.
    Dedicate time to continuous learning and improvement, sharing skills across teams.

    Key Qualifications

    Demonstrated experience in platform development, with a strong focus on backend development including debugging, analysis, and optimization.
    Demonstrated experience working with AWS cloud technologies. Azure and GCP experience also highly desired.
    Proven experience in developing Infrastructure as code using technologies like CloudFormation, Terraform and Serverless Framework.
    Experience with data analytics and visualization tools such as Splunk and ADX and usage of these tools to monitor system quality and performance.
    Hands on infrastructure, systems and application architecture experience in large scale, web-based applications.
    Willingness to learn new tools and technologies and ability to get up to speed quickly.
    Proactive approach to problem-solving, managing uncertainties, and continuous improvement.
    Excellent communication, interpersonal and influencing skills in a cross functional role.

    Relevant Skills & Experience

    Experience with AWS cloud native services including AWS Lambda, API Gateway, DynamoDB, etc.
    Experience with configuration management tools such as Chef.
    Experience with automated Jenkins CI/CD pipelines and Docker.
    Great scripting skills with languages such as Python, Ruby, Bash and TypeScript and a preference for automation over manual toil.
    Deep knowledge of Linux environments and proven experience managing Linux systems and containers.
    Familiarity with secure coding practices and compliance adherence within a cloud environment including PCI, ISO27001 and SOC2.
    Knowledge or experience developing secure code and/or working in a security space.

    """
    
# Clean the extracted text
cleaned_resume = clean_text(resume_text)
cleaned_job_description = clean_text(job_description_text)

# Print cleaned text
print("Cleaned Resume:", cleaned_resume[:500])  # Show first 500 characters
print("Cleaned Job Description:", cleaned_job_description[:500])

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
resume_embedding = model.encode(cleaned_resume, convert_to_tensor=True)
job_description_embedding = model.encode(cleaned_job_description, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.pytorch_cos_sim(resume_embedding, job_description_embedding).item()

print("Semantic Similarity Score:", similarity_score)

def compute_similarity(resume_texts, job_description_text):
    """Computes similarity scores for multiple resumes against a job description."""
    cleaned_job_description = clean_text(job_description_text)
    job_description_embedding = model.encode(cleaned_job_description, convert_to_tensor=True)
    
    scores = []
    for resume_text in resume_texts:
        cleaned_resume = clean_text(resume_text)
        resume_embedding = model.encode(cleaned_resume, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_description_embedding).item()
        scores.append((resume_text[:50], similarity_score))  # Store snippet and score
    
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by highest similarity
    return scores

# Compute and rank resumes by similarity
# ranked_resumes = compute_similarity(cleaned_resume, job_description_text)

# Print ranked results
# for i, (snippet, score) in enumerate(ranked_resumes):
#     print(f"Rank {i+1}: Score: {score:.4f} | Resume Snippet: {snippet}...")

def plot_similarity_score(score):
    """Plots the similarity score of a single resume."""
    plt.figure(figsize=(5, 4))
    plt.barh(["Resume"], [score], color='skyblue')
    plt.xlabel("Similarity Score")
    plt.xlim(0, 1)
    plt.title("Resume Similarity Score")
    plt.show()

def recommend_improvements(resume_text, job_description_text):
    """Provides basic recommendations to enhance a resume based on missing keywords."""
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description_text).split())
    missing_keywords = job_words - resume_words
    
    if missing_keywords:
        print("Consider adding these keywords to align better with the job description:")
        print(", ".join(missing_keywords))
    else:
        print("Your resume already aligns well with the job description!")

# Plot similarity score
plot_similarity_score(similarity_score)

# Recommend resume improvements
recommend_improvements(resume_text, job_description_text)

def recommend_experience_improvements(resume_text, job_description_text):
    """Provides recommendations on specific work experience to enhance resume alignment with job description."""
    resume_sentences = resume_text.split(".\n")
    job_description_sentences = job_description_text.split(".\n")
    
    recommendations = []
    
    for job_sentence in job_description_sentences:
        max_similarity = 0
        best_match = ""
        job_embedding = model.encode(job_sentence, convert_to_tensor=True)
        
        for resume_sentence in resume_sentences:
            resume_embedding = model.encode(resume_sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = resume_sentence
        
        if max_similarity < 0.7:  # Threshold to suggest improvement
            recommendations.append(f"Consider elaborating on experience related to: '{job_sentence.strip()}'")
    
    if recommendations:
        print("Suggested experience improvements:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("Your experience section aligns well with the job description!")

def recommend_experience_improvements(resume_text, job_description_text):
    """Provides recommendations on specific work experience to enhance resume alignment with job description."""
    resume_sentences = resume_text.split(".\n")
    job_description_sentences = job_description_text.split(".\n")
    
    recommendations = []
    
    for job_sentence in job_description_sentences:
        max_similarity = 0
        best_match = ""
        job_embedding = model.encode(job_sentence, convert_to_tensor=True)
        
        for resume_sentence in resume_sentences:
            resume_embedding = model.encode(resume_sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = resume_sentence
        
        if max_similarity < 0.7:  # Threshold to suggest improvement
            recommendations.append(f"Consider elaborating on experience related to: '{job_sentence.strip()}'")
    
    if recommendations:
        print("Suggested experience improvements:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("Your experience section aligns well with the job description!")


# Recommend work experience improvements
recommend_experience_improvements(resume_text, job_description_text)