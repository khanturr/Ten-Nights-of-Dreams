import re
from typing import List, Dict

def load_text(file_path: str) -> str:
    with open("/Users/work/Desktop/Projects/Eerie/tendreams.txt", "r", encoding="utf-8") as file:
        return file.read()

def split_into_dreams(text: str) -> Dict[str, str]:
    # pattern = r"(\d+(?:st|nd|rd|th) Night -.+?)\n"
    sections = text.split('Night')  # Split the text at every "Night"
    dreams = {}

    for i, section in enumerate(sections[1:], start=1):  # Start from 1st Night
        # Add back the "Nth Night" header manually
        header = f"{i}{['st', 'nd', 'rd', 'th'][min(i, 4)-1]} Night"
        content = section.strip()
        dreams[header] = content

    return dreams

def clean_dream_content(content: str) -> str:
    content = content.lstrip("\n – -").strip()
    content = ' '.join(content.split())
    return content.strip()

def preprocess_dreams(dreams: Dict[str, str]) -> Dict[str, str]:
    return {header: clean_dream_content(content) for header, content in dreams.items()}

if __name__ == "__main__":
    file_path = "ten_nights_of_dreams.txt"
    raw_text = load_text(file_path)
    dreams = split_into_dreams(raw_text)
    cleaned_dreams = preprocess_dreams(dreams)
    
    output_file = "cleaned_ten_nights_of_dreams.json"
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_dreams, f, indent=4, ensure_ascii=False)
    
    for header, content in cleaned_dreams.items():
        print(f"{header}\n{content[:200]}...\n")
        break
