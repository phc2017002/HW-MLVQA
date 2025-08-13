from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_args = {"train_batch_size": 16, 
"n_gpu":3, "eval_batch_size": 64, 
'max_answer_length': 50,  
'num_train_epochs': 6, 
'output_dir': "./output_english_1_goolge_ocr_sorted/", 'best_model_dir': 
'./output_english_1_google_ocr_sorted/output_best_english_ocr_sorted/', 'evaluate_during_training': 
True, 'fp16': False, 'overwrite_output_dir':True,
'use_cached_eval_features':False, 
'save_eval_checkpoints': False, 
'save_model_every_epoch': False, 
'max_seq_length': 384, 
'doc_stride': 128, 'do_lower_case': True, 
'gradient_accumulation_steps': 1, 
'learning_rate': 2e-05, 'multiprocessing_chunksize':10}

model_args_3 = {"train_batch_size": 4, 
"n_gpu":3, "eval_batch_size": 64, 
'max_answer_length': 50,  
'num_train_epochs': 6, 
'output_dir': "./output/", 'best_model_dir': 
'./output/best_model_deberta_bs_4/', 'evaluate_during_training': 
True, 'fp16': False, 'overwrite_output_dir':True,
'use_cached_eval_features':False, 
'save_eval_checkpoints': False, 
'save_model_every_epoch': False, 
'max_seq_length': 384, 
'doc_stride': 128, 'do_lower_case': True, 
'gradient_accumulation_steps': 1, 
'learning_rate': 3e-05, 'optimizer': 'Adam', 'adam_epsilon': 1e-08, 'multiprocessing_chunksize':10}



# if you want to fine tune a model locally saved or say you want to continue training a model previously saved give location of the dir where the model is
#model = QuestionAnsweringModel('bert', './models/bert-large-squad-docvqa-finetuned/', args=model_args)


# if you want to fine tune a pretrained model from pytorch trasnformers model zoo (https://huggingface.co/transformers/pretrained_models.html), you can directly give the model name ..the pretrained model will be downloadef first to a cache dir 
# here the model we are fine tuning is bert-large-cased-whole-word-masking-finetuned-squad
#model = QuestionAnsweringModel('bert', 'bert-base-multilingual-uncased', args=model_args)

model = QuestionAnsweringModel('bert', 'oceanpty/mbert-squad', args=model_args)

#print (model.args)

#with open('./sorted_mlqa_ocr_hindi_new_train_data.json') as f:
#    train_data = json.load(f)

#with open('./sorted_mlqa_ocr_hindi_new_val_data.json') as f:
#    dev_data = json.load(f)


#model.train_model(train_data, show_running_loss= False, eval_data=dev_data)
#dev_data, test_data = train_test_split(dev_data, test_size=0.66, random_state=42)


with open('./MLQA_SQuAD_top_4000_test_data.json') as f:
    test_data = json.load(f)


result, predictions = model.eval_model(test_data)
print(result)

#print(predictions)


with open('./all_mlqa_google_ground_truth_test_data_english.txt','w') as f:
    f.write(str(result))
    f.write(str(predictions))


