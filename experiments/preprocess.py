import os
import torch
import argparse

from utils import *
from def_sent_utils import get_all

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler,TensorDataset

PWD = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path", default="bert-base-uncased", type=str, help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type", default="bert", type=str, help="choose from ['bert','roberta','albert']",
)

parser.add_argument(
    '--cuda', type=int, default=1, help="run cuda core number"
)

parser.add_argument(
    "--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n" "Sequences longer than this will be truncated, and sequences shorter \n" "than this will be padded."
)

parser.add_argument(
    "--output", default=None, type=str, help="The output file where the preprocessed embedding saved."
)

parser.add_argument(
    "--no_cuda", default=False, type=str, help="whether to tune the pooling layer with the auxiliary loss",
)

def load_model_tokenizer(version, args):

    if version == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif version == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif version == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    else:
        raise NotImplementedError("not implemented!")
    return model, tokenizer

if __name__ == "__main__":

    args = parser.parse_args()
    if (args.output == None):
        args.output = os.path.join(PWD, args.model_name_or_path + "_" + args.max_seq_length + ".bin")

    print("output: {}".format(args.output))

    model, tokenizer = load_model_tokenizer(args.model_type, args)

    # get external data
    all_data = get_all()

    data_list = []

    for key in all_data.keys():
        for inkey in all_data[key].keys():
            for temp in all_data[key][inkey]:
                data_list.append(temp)
        print("total senteces: {}".format(len(data_list)))

    tokens = {'input_ids': [], 'attention_mask': []}
    
    for data in data_list:
        new_tokens = tokenizer.encode_plus(data, max_length=args.max_seq_length, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    preprocess_data = TensorDataset(tokens['input_ids'],tokens['attention_mask'])
    preprocess_sampler = RandomSampler(preprocess_data)
    preprocess_dataloader = DataLoader(preprocess_data, sampler=preprocess_sampler, batch_size=8)

    if args.cuda == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.cuda)
        device = torch.device("cuda", args.cuda)

    model.to(device)
    model.eval()

    external_embedding_list = []
    external_data_dic = { "ids" : [] , "embedding" : [] }

    for batch in preprocess_dataloader:

        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)

        with torch.no_grad():
            output = model(input_ids, input_mask)
            hidden_states = output["hidden_states"]
            last = hidden_states[-1].transpose(1, 2)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            mean_pooled = torch.avg_pool1d(last_avg.transpose(1, 2), kernel_size=2).squeeze(-1)
            mean_pooled_array = mean_pooled.detach().cpu().numpy()

            for temp_mean in mean_pooled_array:
                external_data_dic["embedding"].append(temp_mean)

            ids_list = input_ids.detach().cpu().tolist()

            for temp_id in ids_list:
                external_data_dic['ids'].append(temp_id)

            assert len(external_data_dic["ids"]) == len(external_data_dic["embedding"])

    torch.save(external_data_dic, args.output)
