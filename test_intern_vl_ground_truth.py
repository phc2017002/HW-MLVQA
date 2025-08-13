import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import io
import math
import gc
import json
import ast
import re
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Image.MAX_IMAGE_PIXELS = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


root_directory = './copied_folders_hindi'
output_directory = './'

ocr_directory = './hindi_ocr_text_new'

output_file_name = 'test_responses_for_hindi_directories_internvl_ground_truth.txt'
output_file_path = os.path.join(output_directory, output_file_name)

text_file_path = output_file_path


def compress_image(image, quality=85):
    """Compress the image using JPEG compression."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.9))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.1)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 2
    device_map['mlp1'] = 2
    device_map['language_model.model.tok_embeddings'] = 2
    device_map['language_model.model.embed_tokens'] = 2
    device_map['language_model.output'] = 2
    device_map['language_model.model.norm'] = 2
    device_map['language_model.lm_head'] = 2
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 2

    return device_map

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, compression_quality=85):
    image = Image.open(image_file).convert('RGB')

    image = compress_image(image, quality=compression_quality)

    new_size = (512, 512)  # Adjust the size as needed for your model
    image = image.resize(new_size)
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

#OpenGVLab/InternVL2-Llama3-76B
#OpenGVLab/InternVL2-4B

path = 'OpenGVLab/InternVL2-8B'
device_map = split_model('InternVL2-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True, device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
#pixel_values = load_image('./test_llm_images/image_1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=2048, do_sample=False)



#question = '<image>\nPlease describe the image shortly.'
#response = model.chat(tokenizer, pixel_values, question, generation_config)
#print(f'User: {question}\nAssistant: {response}')


def get_image_paths_and_qa(directory):
    """Get all main image paths and the corresponding QA file from the directory."""
    image_paths = []
    qa_file = None
    for filename in os.listdir(directory):
        # Check if the file is a main image (not ending with '_answer_<number>.jpg')
        if filename.endswith('.jpg') and not any(filename.endswith(f'_answer_{i}.jpg') for i in range(20)):
            image_paths.append(os.path.join(directory, filename))
        elif filename.endswith('_qa.txt'):
            qa_file = os.path.join(directory, filename)

    print('image_paths----', image_paths)
    return image_paths, qa_file



def save_responses_to_file(output_file, image_paths, question, response):
    """Save the question and response to a text file."""
    with open(output_file, 'a') as file:
        file.write(f"image: {image_paths}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n\n")
        


def read_questions_from_file(qa_file):
    """Read questions from the QA file."""
    with open(qa_file, 'r') as file:
        questions = file.readlines()
    return [q.strip() for q in questions]


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

def get_skip_folders_from_text_file(text_file_path):
    skip_folders = set()
    
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into separate sections by "image:" keyword
        entries = content.split("image:")
        
        for entry in entries[1:]:  # Skip the first split element if empty
            # Extract the list of image paths using regex
            image_match = re.search(r"\['(.+?)'\]", entry.strip(), re.MULTILINE)
            
            if image_match:
                image_paths_str = image_match.group(0)
                try:
                    # Safely evaluate the image paths string to get the list
                    image_paths = ast.literal_eval(image_paths_str)
                    
                    for image_path in image_paths:
                        # Extract folder name from the image path
                        dir_path = os.path.dirname(image_path)
                        folder_name = os.path.basename(dir_path)
                        skip_folders.add(folder_name)
                
                except Exception as e:
                    print(f"Error parsing image paths: {e}")
                    continue
    
    return skip_folders


skip_folders = get_skip_folders_from_text_file(text_file_path) 


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
        print(f"Original text file {ocr_file_path} not found.")
        return ""

for subdir in os.listdir(root_directory):
    if subdir in skip_folders:
        print(f"Skipping {subdir} as it is mentioned in the text file.")
        continue  # Skip processing this folder
    subdir_path = os.path.join(root_directory, subdir)
    if os.path.isdir(subdir_path):
        # Get image paths and QA file for the current directory
        image_paths, qa_file = get_image_paths_and_qa(subdir_path)

        print('len(image_paths)----',len(image_paths))
        if qa_file:
            # Normalize the qa_file path
            #qa_file_normalized = os.path.normpath(qa_file)
            # Check if the qa_file has been processed
            #if qa_file_normalized in processed_qa_files:
            #    print(f"Skipping {qa_file} as it has already been processed.")
            #    continue  # Skip processing this folder
        
            # Read questions from the QA file
            questions = read_questions_from_file(qa_file)
            original_text = get_ocr_text_for_subdir(subdir, ocr_directory)

            #subdir_path_output = '/home/ubuntu'
            #output_file = os.path.join(subdir_path_output, 'test_responses_for_hindi_directories_internvl_final.txt')

            #selected_image_paths = image_paths[:1] if len(image_paths) >= 2 else image_paths

            #print('selected_image_paths', selected_image_paths)

            # Process each image individually
            for idx, image_path in enumerate(image_paths):

                #image = Image.open(image_path)
                
                #new_size = (1024, 1024)  # Adjust the size as needed for your model
                #image = image.resize(new_size)

                #transform = transforms.Compose([
                #transforms.ToTensor(),
                # Include any additional transformations required by your model here
                #])

                
                
                pixel_values = load_image(
                    image_path, max_num=12, compression_quality=85
                ).to(torch.bfloat16).cuda()
                #pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
                num_patches = pixel_values.size(0)
                num_patches_list = [num_patches]

                # Build the prompt with the image and the single OCR text

                # Process each question
                for question in questions:
                    question_prompt = '\n'.join([f'Image-{i+1}: <image>' for i in range(len(image_paths))]) + f'Original_text:{original_text}' + f'\n{question} Provide the answer in the language of the context given and in a single word or phrase. Please provide the bounding box coordinates of the region this sentence describes: <ref>{{}}</ref>'

                    with torch.no_grad():
                        response, history = model.chat(
                            tokenizer, pixel_values, question_prompt, generation_config,
                            num_patches_list=num_patches_list, history=None, return_history=True
                        )

                    # Save the response
                    save_responses_to_file(output_file_path, image_paths, question_prompt, response)
                    print(f'User: {question_prompt}\nAssistant: {response}')

                # Clean up
                del pixel_values, num_patches_list
                torch.cuda.empty_cache()
                gc.collect()

            # After processing the directory
            del questions, image_paths
            gc.collect()
            torch.cuda.empty_cache()

'''
# Iterate over each subdirectory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    if os.path.isdir(subdir_path):
        # Get image paths and QA file for the current directory
        image_paths, qa_file = get_image_paths_and_qa(subdir_path)
        
        if not qa_file:
            print(f"No QA file found in {subdir_path}")
            continue
        
        # Read questions from the QA file
        questions = read_questions_from_file(qa_file)

        # Load and process each image
        pixel_values_list = []
        num_patches_list = []

        for image_path in image_paths:
            pixel_values = load_image(image_path, max_num=12, compression_quality=85).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))

        # Concatenate all pixel values
        pixel_values = torch.cat(pixel_values_list, dim=0)

        if pixel_values_list:
            print(pixel_values_list[0].shape)

        #del pixel_values_list
        #gc.collect()
        #torch.cuda.empty_cache()

        # Print the shape of the first image's pixel values for verification

        # Determine the sequence length from the first image
        seq_length = pixel_values_list[0].size(0) if pixel_values_list else 0
        print('seq_length----', seq_length)

        del pixel_values_list
        gc.collect()
        torch.cuda.empty_cache()

        subdir_path_1 = '/home/ubuntu'

        output_file = os.path.join(subdir_path_1, 'test_responses_for_english_directories.txt')

        # Iterate over questions and process each one
        with torch.no_grad():
            for question in questions:
                # Construct the question dynamically based on the number of images
                question_prompt = '\n'.join([f'Image-{i+1}: <image>' for i in range(len(image_paths))]) + \
                              f'\n{question} Answer the questions in one word/phrase. Please provide the bounding box coordinates of the region this sentence describes: <ref>{{}}</ref>'

                # Get the response from the model
                response, history = model.chat(tokenizer, pixel_values, question_prompt, generation_config,
                                           num_patches_list=num_patches_list,
                                           history=None, return_history=True)

                save_responses_to_file(output_file, question_prompt, response)

                print(f'User: {question_prompt}\nAssistant: {response}')

        del pixel_values, num_patches_list, questions, image_paths
        gc.collect()
        torch.cuda.empty_cache()
'''

'''
#The images are concatenated horizontally.
question = '<image>\n What was Frédérics nationalities? Answer the questions in one word/phrase. \n Please provide the bounding box coordinates of the region this sentence describes: <ref>{}</ref>'

response, history = model.chat(tokenizer, pixel_values1, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')


question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
'''