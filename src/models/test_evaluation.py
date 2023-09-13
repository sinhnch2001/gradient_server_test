import argparse
import sys
import torch
sys.path.insert(0,r'./')
from src.data.dataloader_GradRes import ResDataLoader
from src.data.dataloader_GradSearch import StateDataLoader
from evaluation import Evaluation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from accelerate.logging import get_logger
import json
logger = get_logger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help="Type module")
    parser.add_argument('--test_files', nargs='+', help= "Directory to test file (can be multiple files)")
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help="ignore_pad_token_for_loss")
    parser.add_argument('--model_name', type=str, default="google/flan-t5-small", help="model name")
    parser.add_argument('--path_to_save_dir', type=str, help="Path to the save directory json file")
    parser.add_argument('--log_input_label_predict', type=str, help="Path to the save directory json file")
    parser.add_argument('--max_target_length', type=int, default=80, help="Max target length")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for the dataloader")
    parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument('--num_beams', type=int, default=4, help="number of beams")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--with_tracking', action='store_true',
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument('--train_files',default=None)
    parser.add_argument('--val_file',default=None)

    args = parser.parse_args(args)

    if args.test_files is None:
        raise ValueError(
            "You are running training script without input data. "
            "Make sure you set all input data to --test_files")


    return args


def main(args):
    args = parse_args(args)

    # Load the tokenizer from the tokenizer.json file
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load the model from the model.bin file
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.load_state_dict(torch.load(args.path_to_save_dir))

    accelerator = Accelerator()
    accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps

    device = accelerator.device
    with accelerator.main_process_first():
        model = model.to(device)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    dataloader_args = {
        "model_name": args.model_name,
        "train_file": None,
        "val_file": None,
        "test_file": args.test_files,
        "batch_size": args.batch_size,
        "seed": args.seed
    }
    if args.module == "res":
        dataloaders = ResDataLoader(**dataloader_args).__call__()
        metrics_name = ['rouge', "bleu"]
    elif args.module == "dst_tod":
        dataloaders = StateDataLoader(**dataloader_args).__call__()
        metrics_name = ["rsa", "jga", "sa"]
    elif args.module == "dst_odd":
        dataloaders = StateDataLoader(**dataloader_args).__call__()
        metrics_name = ["f1"]

    model, dataloaders['test'], tokenizer = accelerator.prepare(model, dataloaders['test'], tokenizer)

    evaluator = Evaluation(eval_dataloaders=dataloaders['test'],
                           ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
                           metrics_name=metrics_name,
                           with_tracking=args.with_tracking,
                           num_beams=args.num_beams,
                           max_target_length=args.max_target_length)


    if args.with_tracking:
        result, total_loss_eval, label, predict, jga = evaluator.eval(accelerator=accelerator,
                                                 tokenizer=tokenizer, model=model, log_label_predict=True)
    else:
        result, label, predict, jga = evaluator.eval(accelerator=accelerator,
                                tokenizer=tokenizer, model=model, log_label_predict=True)
    test = json.load(open(args.test_files[0]))

    ildm_list = []
    for i in range(len(label)):
        if args.module == "res":
            item = test[i]['instruction'] \
                    .replace('{context}', test[i]['context'].strip()) \
                    .replace('{ontology}', test[i]['ontology'].strip()) \
                    .replace('{system_action}', test[i]['system_action'].strip()) \
                    .replace('{documents}', test[i]['documents'].strip()) \
                    .replace('\s+', ' ') \
                    .replace(' |  | .', '.') \
                    .replace(' | .', '.') \
                    .replace(' || ', ' | ') \
                    .replace(' |  | ', '')
        else:
            item = test[i]['instruction'] \
                    .replace('{list_user_action}', test[i]['list_user_action'].strip()) \
                    .replace('{history}', test[i]['history'].strip()) \
                    .replace('{current}', test[i]['current'].strip()) \
                    .replace('{ontology}', test[i]['ontology'].strip())
        ildm = {
            "input": item,
            "label": label[i],
            "predict": predict[i],
            "JGA" : jga[i]
        }
        ildm_list.append(ildm)
    with open(args.log_input_label_predict, 'w') as f:
        json.dump(ildm_list, f, indent=4)

    for k,v in result.items():
        print(k,":",v)
    print("\n")

if __name__ == "__main__":
    main(sys.argv[1:])

#
#
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import argparse
# import sys
# sys.path.insert(0,r'./')
# from src.data.dataloader_GradRes import ResDataLoader
# from src.data.dataloader_GradSearch import StateDataLoader
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from tqdm.auto import tqdm
# from accelerate.utils import DistributedType
# import numpy as np
# import nltk
# nltk.download('punkt', quiet=True)
# import json
#
# logger = get_logger(__name__)
#
#
# def parse_args(args):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--module', type=str, help="Type module")
#     parser.add_argument('--test_files', nargs='+', help="Directory to test file (can be multiple files)")
#     parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help="ignore_pad_token_for_loss")
#     parser.add_argument('--model_name', type=str, default="google/flan-t5-small", help="model name")
#     parser.add_argument('--path_to_save_dir', type=str, help="Path to the save directory json file")
#     parser.add_argument('--log_input_label_predict', type=str, help="Path to the save directory json file")
#     parser.add_argument('--max_target_length', type=int, default=80, help="Max target length")
#     parser.add_argument('--batch_size', type=int, default=8, help="Batch size for the dataloader")
#     parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")
#     parser.add_argument('--num_beams', type=int, default=4, help="number of beams")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument('--with_tracking', action='store_true',
#                         help="Whether to enable experiment trackers for logging.")
#     parser.add_argument('--train_files', default=None)
#     parser.add_argument('--val_file', default=None)
#
#     args = parser.parse_args(args)
#
#     if args.test_files is None:
#         raise ValueError(
#             "You are running training script without input data. "
#             "Make sure you set all input data to --test_files")
#
#     return args
#
# def main(args):
#     label = []
#     predict = []
#     args = parse_args(args)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
#     model.load_state_dict(torch.load(args.path_to_save_dir))
#
#     accelerator = Accelerator()
#     accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
#
#     device = accelerator.device
#     with accelerator.main_process_first():
#         model = model.to(device)
#
#     embedding_size = model.get_input_embeddings().weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         model.resize_token_embeddings(len(tokenizer))
#     if model.config.decoder_start_token_id is None:
#         raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
#
#     dataloader_args = {
#         "model_name": args.model_name,
#         "train_file": None,
#         "val_file": None,
#         "test_file": args.test_files,
#         "batch_size": args.batch_size,
#         "seed": args.seed
#     }
#     if args.module == "res":
#         dataloaders = ResDataLoader(**dataloader_args).__call__()
#         metrics_name = ['rouge', "bleu"]
#     elif args.module == "dst_tod":
#         dataloaders = StateDataLoader(**dataloader_args).__call__()
#         metrics_name = ["rsa", "jga", "sa"]
#     elif args.module == "dst_odd":
#         dataloaders = StateDataLoader(**dataloader_args).__call__()
#         metrics_name = ["f1"]
#
#     model, dataloaders['test'], tokenizer = accelerator.prepare(model, dataloaders['test'], tokenizer)
#
#     accelerator.wait_for_everyone()
#     gen_kwargs = {
#         "max_length": args.max_target_length,
#         "num_beams": args.num_beams
#     }
#     for step, batch in enumerate(tqdm(dataloaders['test'],
#                                       desc="Eval on process: " + str(accelerator.process_index),
#                                       colour="blue", position=accelerator.process_index)):
#
#         # Pass dummy batch to avoid caffe error
#         if step == 0 and accelerator.distributed_type == DistributedType.FSDP:
#             model(**batch)
#         with torch.no_grad():
#             # synced_gpus was necessary else resulted into indefinite hang
#             generated_tokens = accelerator.unwrap_model(model).generate(
#                 batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 synced_gpus=True if accelerator.distributed_type != DistributedType.NO else False,
#                 **gen_kwargs
#             )
#
#             generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
#             labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
#             generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
#             generated_tokens = generated_tokens.cpu().numpy()
#             labels = labels.cpu().numpy()
#
#             if args.ignore_pad_token_for_loss:
#                 # Replace -100 in the labels as we can't decode them.
#                 labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#             if isinstance(generated_tokens, tuple):
#                 generated_tokens = generated_tokens[0]
#
#             decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#             decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#             decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
#
#             label = label + decoded_labels
#             predict = predict + decoded_preds
#             del decoded_preds
#             del decoded_labels
#     ld_list = []
#     for i in range(len(label)):
#         ld = {
#             "label": label[i],
#             "predict": predict[i]
#         }
#         ld_list.append(ld)
#     with open(args.log_input_label_predict, 'w') as f:
#         json.dump(ld_list, f, indent=4)
#
# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [label.strip() for label in labels]
#
#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
#     return preds, labels
#
# if __name__ == "__main__":
#      main(sys.argv[1:])