import os
import json
import logging
import asyncio
import re
import warnings
from typing import List, Optional
from glob import glob

# Import transformers for loading Qwen-7B model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration
API_PROVIDER = "QWEN"  # Set to QWEN to indicate we're using the Qwen model
QWEN_MODEL_NAME = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"  # Name of the Qwen-7B model on Hugging Face
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

MAX_TOKENS = 1024  # Adjust based on your hardware capabilities
TOKEN_BUFFER = 100
TOKEN_CUSHION = 50

INPUT_DIR = "./english_ocr_text_new"  # Directory containing OCR text files
OUTPUT_DIR = "./english_ocr_text_new_corrected"  # Directory to save processed files

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the Qwen-7B model and tokenizer
logging.info("Loading Qwen-7B model...")

model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME, device_map='auto', trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
model.eval()
logging.info("Model loaded successfully.")

def get_tokenizer(model_name: str):
    return tokenizer  # Since we're using the same tokenizer throughout

def estimate_tokens(text: str, model_name: str) -> int:
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))

def approximate_tokens(text: str) -> int:
    text = re.sub(r'\s+', ' ', text.strip())
    tokens = re.findall(r'\b\w+\b|\S', text)
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)
        elif re.match(r'^[A-Z]{2,}$', token):
            count += len(token)
        elif re.search(r'[^\w\s]', token):
            count += 1
        elif len(token) > 10:
            count += len(token) // 4 + 1
        else:
            count += 1
    return int(count * 1.1)

def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = []
    current_chunk_length = 0
    chunk_size = max_chunk_tokens  # Tokens per chunk

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph, model_name)
        if current_chunk_length + paragraph_tokens <= chunk_size:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_tokens
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_length = paragraph_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Add overlap between chunks
    overlap = 50  # Number of tokens to overlap
    for i in range(1, len(chunks)):
        overlap_text = chunks[i - 1].split()[-overlap:]
        chunks[i] = " ".join(overlap_text) + " " + chunks[i]

    return chunks

def generate_completion_qwen(prompt: str, max_tokens: int = MAX_TOKENS - TOKEN_BUFFER) -> Optional[str]:
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
        prompt_length = input_ids.shape[1]
        adjusted_max_tokens = min(max_tokens, MAX_TOKENS - prompt_length - TOKEN_BUFFER)

        if adjusted_max_tokens <= 0:
            logging.warning("Prompt is too long for the model's context window.")
            return None
        else:
            output_ids = model.generate(
                input_ids,
                max_length=prompt_length + adjusted_max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            return completion.strip()
    except Exception as e:
        logging.error(f"Error generating completion: {e}")
        return None

async def generate_completion(prompt: str, max_tokens: int = 500) -> Optional[str]:
    return generate_completion_qwen(prompt, max_tokens)

async def process_chunk(
    chunk: str,
    prev_context: str,
    chunk_index: int,
    total_chunks: int,
    reformat_as_markdown: bool,
    suppress_headers_and_page_numbers: bool
) -> tuple[str, str]:
    logging.info(f"Processing chunk {chunk_index + 1}/{total_chunks} (length: {len(chunk):,} characters)")

    correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the original text
   - Do not add any new information
   - Remove unnecessary line breaks within sentences
   - Maintain paragraph breaks
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

Previous context:
{prev_context[-500:]}

Current chunk to process:
{chunk}

Corrected text:
"""

    corrected_chunk = await generate_completion(correction_prompt, max_tokens=MAX_TOKENS)
    processed_chunk = corrected_chunk

    if reformat_as_markdown:
        markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Convert headings to markdown heading levels (# for main titles, ## for subtitles, etc.)
2. Maintain paragraph structure and remove incorrect line breaks
3. Format lists properly if they exist
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate
5. Preserve all original content and meaning
6. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly." if not suppress_headers_and_page_numbers else "Remove headers, footers, and page numbers while preserving all other content."}

Text to reformat:

{corrected_chunk}

Reformatted markdown:
"""
        processed_chunk = await generate_completion(markdown_prompt, max_tokens=MAX_TOKENS)

    new_context = processed_chunk[-1000:]
    return processed_chunk, new_context

async def process_chunks(
    chunks: List[str],
    reformat_as_markdown: bool,
    suppress_headers_and_page_numbers: bool
) -> List[str]:
    total_chunks = len(chunks)
    context = ""
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        processed_chunk, context = await process_chunk(
            chunk,
            context,
            i,
            total_chunks,
            reformat_as_markdown,
            suppress_headers_and_page_numbers
        )
        processed_chunks.append(processed_chunk)

    return processed_chunks

async def process_text(
    input_text: str,
    reformat_as_markdown: bool = True,
    suppress_headers_and_page_numbers: bool = True
) -> str:
    """
    Process OCR text for correction and optional markdown formatting.

    Args:
        input_text (str): The OCR text to process
        reformat_as_markdown (bool): Whether to format the output as markdown
        suppress_headers_and_page_numbers (bool): Whether to remove headers and page numbers

    Returns:
        str: The processed text
    """
    logging.info("Starting text processing...")
    max_chunk_tokens = MAX_TOKENS - TOKEN_BUFFER

    # Split text into chunks
    chunks = chunk_text(input_text, max_chunk_tokens, QWEN_MODEL_NAME)
    logging.info(f"Text split into {len(chunks)} chunks. Processing...")

    # Process chunks
    processed_chunks = await process_chunks(
        chunks,
        reformat_as_markdown,
        suppress_headers_and_page_numbers
    )

    # Combine processed chunks
    final_text = "\n\n".join(processed_chunks)

    # Remove any "Corrected text:" headers that might have been added
    final_text = final_text.replace("# Corrected text\n", "").replace("# Corrected text:", "")
    final_text = final_text.replace("\nCorrected text", "").replace("Corrected text:", "")

    logging.info(f"Text processing complete. Final length: {len(final_text):,} characters")
    return final_text

def process_files(input_dir: str, output_dir: str):
    # Get list of OCR text files
    ocr_files = glob(os.path.join(input_dir, "ocr_text_*.txt"))
    total_files = len(ocr_files)
    logging.info(f"Found {total_files} OCR text files to process.")

    for idx, input_file in enumerate(ocr_files, 1):
        filename = os.path.basename(input_file)
        logging.info(f"Processing file {idx}/{total_files}: {filename}")

        output_file = os.path.join(output_dir, f"corrected_{filename}")

        try:
            # Read input text
            with open(input_file, 'r', encoding='utf-8') as f:
                input_text = f.read()

            # Process the text
            corrected_text = asyncio.run(process_text(
                input_text,
                reformat_as_markdown=True,
                suppress_headers_and_page_numbers=True
            ))

            # Save the output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corrected_text)

            logging.info(f"File processed successfully. Output saved to: {output_file}")

        except Exception as e:
            logging.error(f"An error occurred while processing {filename}: {e}")

def main():
    process_files(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()