import os
import torch
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from PIL import Image


# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")
num_layers = config.num_hidden_layers
num_gpus = torch.cuda.device_count()

# Create a device map

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", torch_dtype=torch.float16, device_map='auto'
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")

# Directory configuration
root_directory = './copied_folders_hindi'
output_directory = './'

output_file_name = 'output_responses_qwen2vl_hindi_image_ground_truth.txt'
output_file_path = os.path.join(output_directory, output_file_name)

#model = model.half()

ocr_directory = './hindi_ocr_text_new'

def get_image_paths_and_qa(subdir_path):
    image_paths = []
    qa_file = None
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        if file.endswith(('.jpg', '.jpeg', '.png')) and '_answer_' not in file:
            image_paths.append(file_path)
        elif file.endswith('_qa.txt'):
            qa_file = file_path
    return image_paths, qa_file

def read_questions_from_file(file_path):
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Question:"):
                question = line.split("Question:", 1)[1].strip()
                questions.append(question)
    return questions

def get_processed_qa_files(output_file_path):
    processed_qa_files = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            for line in f:
                if line.startswith("qa_file:"):
                    # Extract the qa_file path and strip any leading/trailing whitespace
                    qa_file_path = line[len("qa_file:"):].strip()
                    # Normalize the path to handle any relative path issues
                    qa_file_path = os.path.normpath(qa_file_path)
                    processed_qa_files.add(qa_file_path)
    return processed_qa_files

processed_qa_files = get_processed_qa_files(output_file_path)


def save_responses_to_file(output_file, qa_file, image_paths, question_prompt, response):
    with open(output_file, 'a') as f:
        f.write(f"qa_file: {qa_file}\nimage_paths: {image_paths}\n\n")
        f.write(f"User: {question_prompt}\nAssistant: {response}\n\n")

def load_image(image_path):
    # Placeholder function to load image. Replace with actual loading logic.
    return torch.rand((1, 3, 224, 224))  # Example tensor shape

def load_and_resize_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return image

def get_ocr_text_for_subdir(subdir_name, ocr_directory):
    # Extract the identifier from the subdirectory name
    # Assuming subdir_name is like 'context_XXXX'
    parts = subdir_name.split('_')
    if len(parts) >= 2:
        identifier = parts[1]  # e.g., '1005'
    else:
        identifier = subdir_name  # If no underscore, use the whole name
    ocr_file_name = f"original_text_{identifier}.txt"
    ocr_file_path = os.path.join(ocr_directory, ocr_file_name)
    if os.path.exists(ocr_file_path):
        with open(ocr_file_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read().strip()
        return ocr_text
    else:
        print(f"OCR text file {ocr_file_path} not found.")
        return ""

# Process each folder in the root directory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    if os.path.isdir(subdir_path):
        image_paths, qa_file = get_image_paths_and_qa(subdir_path)

        if qa_file:
            # Normalize the qa_file path
            qa_file_normalized = os.path.normpath(qa_file)
            # Check if the qa_file has been processed
            if qa_file_normalized in processed_qa_files:
                print(f"Skipping {qa_file} as it has already been processed.")
                continue  # Skip processing this folder

            # Get the OCR text for this subdirectory
            ocr_text = get_ocr_text_for_subdir(subdir, ocr_directory)

            print('qa_file-----', qa_file)
            questions = read_questions_from_file(qa_file)
            selected_image_paths = image_paths[:2] if len(image_paths) > 3 else image_paths
            print('image_paths-----', image_paths)
            print('ocr_text---', ocr_text)

            for question in questions:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path} for img_path in selected_image_paths
                        ] + [
                            {
                                "type": "text",
                                "text": (
                                    f"{question}\n\n"
                                    f"The text extracted from the images is:\n{ocr_text}\n\n"
                                    "Provide the answer in the language of the context given and in a single word or phrase."
                                )
                            }
                        ],
                    }
                ]

                # Prepare inputs for the model
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                #image_inputs, video_inputs = process_vision_info(messages)
                image_inputs = [load_and_resize_image(img_path, target_size=(512, 512)) for img_path in selected_image_paths]
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to("cuda")

                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(f"Response for question '{question}': {output_text}")

                # Save response
                save_responses_to_file(output_file_path, qa_file, image_paths, question, output_text)

                # Clear GPU cache and collect garbage
                del generated_ids, generated_ids_trimmed, inputs, image_inputs, text
                torch.cuda.empty_cache()
                gc.collect()

            del questions, selected_image_paths
            gc.collect()
            torch.cuda.empty_cache()

del model
del processor
torch.cuda.empty_cache()
gc.collect()