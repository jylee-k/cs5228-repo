import pandas as pd
import re

def clean_text(text):
    """
    Cleans text by fixing encoding artifacts and removing common web-scraping junk.
    """
    if not isinstance(text, str):
        return text
    
    # 1. Fix encoding artifacts (UTF-8 bytes misinterpreted as Latin-1 characters)
    try:
        # Many 'latin-1' files are actually UTF-8 where bytes were read literally.
        # This converts 'â€™' back to '’', 'â€“' to '–', etc.
        text = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If it's already correct or doesn't match the pattern, keep it
        pass
    
    # 2. General Cleanup
    # Remove HTML tags if any (common in scraping)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace, newlines, and tabs
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove specific artifacts like "Source: ..." or "The photo shows ..." if they appear in certain formats
    # (Though we keep the content unless it's clearly non-narrative junk)
    
    # 3. Manual fixes for persistent garbled characters not caught by encode/decode
    replacements = {
        'â€™': "'",
        'â€“': "-",
        'â€”': "—",
        'Â©': "©",
        'â€œ': '"',
        'â€?': '"',
        'â€˜': "'",
        'Ã©': "é",
        'Ã¡': "á",
        'Ã³': "ó",
        'Ã±': "ñ",
        'â€¢': "•",
        'â€¦': "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

def main():
    input_file = 'sps_filtered_qa.csv'
    output_file = 'sps_filtered_qa_utf8.csv'
    
    print(f"Loading {input_file} with 'latin-1' encoding...")
    try:
        # Load with latin-1 to capture the 'garbled' characters for cleaning
        df = pd.read_csv(input_file, encoding='latin-1')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Cleaning dataset columns...")
    # Apply cleaning to text-heavy columns
    target_columns = ['text_chunk', 'question', 'answer']
    for col in target_columns:
        if col in df.columns:
            print(f"  Cleaning column: {col}")
            df[col] = df[col].apply(clean_text)
            
    print(f"Saving cleaned dataset to {output_file} in UTF-8...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print("Success! Dataset is ready for TTS.")

if __name__ == "__main__":
    main()
