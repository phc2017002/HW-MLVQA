import transformers
import torch
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

def format_messages(context, question):
    return f"Context: {context}\n\nQuestion: {question}\n\nProvide the answer in the language of the context given in a single word or phrase.\n\nAnswer:"

# Load the JSON file
with open('sorted_easy_mlqa_ocr_hindi_new_test_data.json', 'r') as f:
    data = json.load(f)

results = []

for entry in data:
    context = entry['context']
    for qa in entry['qas']:
        question = qa['question']
        ground_truth = qa['answers'][0]['text']
        
        prompt = format_messages(context, question)
        
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            return_full_text=False
        )
        
        model_answer = outputs[0]['generated_text'].strip()

        print('question----', question)
        print('ground_truth---', ground_truth)
        print('model_answer----', model_answer)
        
        results.append({
            'id': qa['id'],
            'question': question,
            'ground_truth': ground_truth,
            'model_answer': model_answer
        })

# Save results to JSON file
with open('evaluation_results_sorted_easy_mlqa_ocr_hindi_new_test_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Results saved to evaluation_results.json")