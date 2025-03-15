from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

def load_raw_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

template = (
    "You are an AI that extracts structured information from unstructured text. "
    "The given text contains course details from a university website. Extract the "
    "relevant course information and return it in JSON format. The JSON should include: "
    "course_name, course_code, description, instructor, schedule, and prerequisites, deadlines for course registration "
    "If any information is missing, exclude that key. The response must be a valid JSON.You must always return valid JSON fenced by a markdown code block. Do not return any additional text. to your input may help steer the model to returning the expected format"
)

model = OllamaLLM(model="llama3.2")

def parse_course_data(raw_text):
    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser()  # Ensures valid JSON output
    chain = prompt | model | parser  # Enforce JSON formatting
    
    response = chain.invoke({"text": raw_text})
    
    return response

if __name__ == "__main__":
    file_path = r"Scrapper\raw_data.txt"
    raw_text = load_raw_data(file_path)
    parsed_json = parse_course_data(raw_text)
    
    print(json.dumps(parsed_json, indent=4))
