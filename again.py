import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Load the datasets
resume_df = pd.read_excel('C:\\Users\\ashna\\Downloads\\third semester\\final hackathon\\updated_resume_dataset.xlsx')
internship_df = pd.read_excel('C:\\Users\\ashna\\Downloads\\third semester\\final hackathon\\updated_internship_dataset.xlsx')

# Display the first few rows of each dataset
print(resume_df.head())
print(internship_df.head())

# Check for missing values
resume_df.fillna('', inplace=True)
internship_df.fillna('', inplace=True)

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess skills function
def preprocess_skills(skills):
    tokens = word_tokenize(skills.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Apply the preprocessing function to skills
resume_df['processed_Skills'] = resume_df['Skills'].apply(preprocess_skills)
internship_df['processed_Required_Skills'] = internship_df['Required Skills'].apply(preprocess_skills)

# Create unique skill set and skill vectors
all_Skills = resume_df['processed_Skills'].sum() + internship_df['processed_Required_Skills'].sum()
unique_Skills = set(all_Skills)

# Create a dictionary for skill indices
Skill_to_index = {skill: idx for idx, skill in enumerate(unique_Skills)}

# Function to convert skills to vectors
def skills_to_vector(skills):
    vector = [0] * len(Skill_to_index)
    for skill in skills:
        if skill in Skill_to_index:
            vector[Skill_to_index[skill]] += 1
    return vector

# Convert skills to vectors
resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector)
internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector)

# Prepare features and labels for model training
X = []
y = []

for index, internship in internship_df.iterrows():
    for index, resume in resume_df.iterrows():
        X.append(resume['Skill_vector'] + internship['Required_Skill_vector'])
        y.append(1 if set(resume['processed_Skills']) & set(internship['processed_Required_Skills']) else 0)

# Convert to DataFrame
X = pd.DataFrame(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Train Naive Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")

# Evaluate both models
evaluate_model(logistic_model, X_test, y_test)
evaluate_model(naive_bayes_model, X_test, y_test)

# Function to calculate similarity using Jaccard index
def calculate_similarity(resume_skills, internship_skills):
    set_resume_skills = set(resume_skills)
    set_internship_skills = set(internship_skills)
    intersection = set_resume_skills.intersection(set_internship_skills)
    union = set_resume_skills.union(set_internship_skills)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)  # Jaccard Similarity

# Function to match internships
def match_internships(resume):
    resume_vector = resume['Skill_vector']
    results = []
    for index, internship in internship_df.iterrows():
        similarity_score = calculate_similarity(resume['processed_Skills'], internship['processed_Required_Skills'])
        
        if similarity_score > 0.5:  # Use a threshold for matching
            results.append({
                'internship_title': internship['Title'],
                'company': internship['Company'],
                'location': internship['Location'],
                'description': internship['Description'],
                'similarity_score': similarity_score
            })
    
    # Sort results by similarity score (highest first)
    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    return results

# User interface to find matching internships by name
def find_applicant_and_match_internships():
    applicant_name = input("Enter the name of the applicant: ")
    matching_resume = resume_df[resume_df['Name'].str.contains(applicant_name, case=False)]
    
    if matching_resume.empty:
        print(f"No resume found for applicant: {applicant_name}")
    else:
        resume = matching_resume.iloc[0]
        print(f"Found resume for {resume['Name']}!")
        
        matched_internships = match_internships(resume)
        
        if not matched_internships:
            print("No internships matched for this applicant.")
        else:
            print(f"Top matching internships for {resume['Name']}:")
            top_match = matched_internships[0]
            print(f"Internship Title: {top_match['internship_title']}")
            print(f"Company: {top_match['company']}")
            print(f"Location: {top_match['location']}")
            print(f"Description: {top_match['description']}")

# Run the interface
find_applicant_and_match_internships()
import time

# Function to measure inference time
def measure_inference_time(model, X):
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    return end_time - start_time

# Measure inference time for Logistic Regression
logistic_inference_time = measure_inference_time(logistic_model, X_test)
print(f"Logistic Regression Inference Time: {logistic_inference_time:.4f} seconds")

# Measure inference time for Naive Bayes
naive_bayes_inference_time = measure_inference_time(naive_bayes_model, X_test)
print(f"Naive Bayes Inference Time: {naive_bayes_inference_time:.4f} seconds")
