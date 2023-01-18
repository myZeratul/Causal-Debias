import csv
import logging
import sys
import argparse
import torch
import os
import random
import time 
import numpy as np
import torch.nn.functional as F


from scipy.stats import wasserstein_distance
from utils import *
from functools import reduce
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler,TensorDataset

from transformers import AdamW
from transformers import BertTokenizer,BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer,AlbertForSequenceClassification

PWD = os.getcwd()

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender','race'],
    help="Choose from ['gender','race']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--task_name",
    default="SST-2",
    type=str,
    help="The name of the task to train.[CoLA, QNLI, SST-2]"
)

parser.add_argument(
    "--data_dir",
    default="/data/glue_data/SST-2",
    type=str,
    help="data path to put dataset",
)

parser.add_argument(
    '--k', 
    default=2,
    type=int,  
    help='top k similar sentence'
)

parser.add_argument(
    '--cuda',
    type=int,
    default=1,
    help="run cuda core number"
)

parser.add_argument(
    "--num_train_epochs",
    default=5,
    type=int,
    help="Total number of training epochs to perform."
)

parser.add_argument(
    "--train_batch_size",
    default=32,
    type=int,
    help="Total batch size for training."
)

parser.add_argument(
    "--eval_batch_size",
    default=32, 
    type=int,
    help="Total batch size for eval."
)

parser.add_argument(
    "--lr",
    default=1e-5,
    type=float,
    help="learning rate in auto-debias fine-tuning",
)

parser.add_argument(
    "--no_cuda",
    default=False,
    type=str,
    help="whether to tune the pooling layer with the auxiliary loss",
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help="random seed for initialization"
)

parser.add_argument(
    "--max_seq_length",
    default=128,
    # default=512, # roberta
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded."
)

parser.add_argument(
    "--do_train",
    default=True,
    action='store_true',
    help="Whether to run training."
)

parser.add_argument(
    "--do_eval",
    default=True,
    action='store_true',
    help="Whether to run eval on the dev set."
)


parser.add_argument(
    '--gradient_accumulation_steps',
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass."
)

parser.add_argument(
    "--no_save",
    default=False,
    # action='store_true',
    help="Set this flag if you don't want to save any results."
)

parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    help="The output directory where the model predictions and checkpoints will be written."
)

parser.add_argument(
    "--data_path",
    default="data/gender_data/",
    type=str,
    help="Data path to put the target/attribute word list",
)

parser.add_argument(
    '--external_data', 
    default="",
    type=str,  
    help='Dir that saved external data'
)

parser.add_argument(
    '--tau', 
    default=0.5,
    type=float,  
    help='Hyper parameters'
)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, segment_ids, label_id):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DualInputFeatures(object):
    """A single set of dual features of data."""

    def __init__(self, input_ids_a, input_ids_b, mask_a, mask_b, segments_a, segments_b):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.mask_a = mask_a
        self.mask_b = mask_b
        self.segments_a = segments_a
        self.segments_b = segments_b

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

#features prepare
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of input features."""
    '''
    output_mode: classification or regression
    '''	
    if (label_list != None):
        label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    for tuple_example in enumerate(examples):

        example = tuple_example[1]

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        if args.model_type == 'roberta':
            tokens = ["<s>"] + tokens_a + ["</s>"]
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            if args.model_type == 'roberta':
                tokens += tokens_b + ["</s>"]
            else:
                tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        if args.model_type == 'roberta':
            padding = [1] * (max_seq_length - len(input_ids))
        else:
            padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert(len(input_ids) == max_seq_length)
        assert(len(input_mask) == max_seq_length)
        assert(len(segment_ids) == max_seq_length)

        if (label_list != None):
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label_id = None

        features.append(
                InputFeatures(tokens=tokens,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
    return features


def convert_examples_to_augfeatures(examples, label_list, max_seq_length, tokenizer, output_mode, male, female):

    if (label_list != None):
        label_map = {label : i for i, label in enumerate(label_list)}

    assert len(male) == len(female), "two list length is not equal"
    aug_sentence_list = []
    features = []
    for tuple_example in enumerate(examples):
        example = tuple_example[1]
        tokens_a = tokenizer.tokenize(example.text_a)
        # print(tokens_a)
        it_male = iter(male)
        it_female = iter(female)

        flag_a = False

        for count in range(len(male)):
            count += 1

            text_male = next(it_male)
            text_female = next(it_female)  
            # print(text_male, text_female)
            if text_female in tokens_a:
                tokens_a[tokens_a.index(text_female)] = text_male
                flag_a = True
            elif text_male in tokens_a:
                tokens_a[tokens_a.index(text_male)] = text_female
                flag_a = True
            else:
                continue
        
        if flag_a:
            aug_sentence_a = " ".join(tokens_a).replace(" ##", "").strip()
            # print(aug_sentence_a)
            aug_sentence_list.append(aug_sentence_a)
            # print(tokens_a)
            None 
        else:
            continue

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

            # print(tokens_b)
            it1_male = iter(male)
            it1_female = iter(female)
        
            flag_b = False

            for count in range(len(male)):
                count += 1

                text_male = next(it1_male)
                text_female = next(it1_female)  

                # print(text_male, text_female)
                if text_female in tokens_b:
                    tokens_b[tokens_b.index(text_female)] = text_male
                    flag_b = True
                elif text_male in tokens_b:
                    tokens_b[tokens_b.index(text_male)] = text_female
                    flag_b = True
                else:
                    continue
            
            if flag_b:
                aug_sentence_b = " ".join(tokens_b).replace(" ##", "").strip()
                # print(aug_sentence_b)
                aug_sentence_list.append(aug_sentence_b)
            else:
                continue
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if args.model_type == 'roberta':
            tokens = ["<s>"] + tokens_a + ["</s>"]
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            if args.model_type == 'roberta':
                tokens += tokens_b + ["</s>"]
            else:
                tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        if args.model_type == 'roberta':
            padding = [1] * (max_seq_length - len(input_ids))
        else:
            padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert(len(input_ids) == max_seq_length)
        assert(len(input_mask) == max_seq_length)
        assert(len(segment_ids) == max_seq_length)

        if (label_list != None):
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label_id = None

        features.append(
                InputFeatures(tokens=tokens,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
    return features

# bert roberta albert
def load_model_tokenizer(version, args):

    if version == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif version == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif version == 'albert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    else:
        raise NotImplementedError("Not implemented!")
    return model, tokenizer

def get_expand_festure_list(args, features, tokenizer, model, device, word_list):

    print("Start loading external data")
    starttime = time.time()
    model.eval()
    external_dic = torch.load(args.external_data)
    external_ids_list = external_dic['ids']
    embedding = torch.from_numpy(np.array(external_dic["embedding"])).to(device)
    expand_feature_list = []
    for feature in features:
        if set(word_list).intersection(set(feature.tokens)) == set():
            continue
        temp_list = get_top_k_features(args, feature, tokenizer, model, device, external_ids_list, embedding)
        expand_feature_list += temp_list

    result = reduce(lambda y,x:y if (x.tokens in [i.tokens for i in y]) else (lambda z ,u:(z.append(u),z))(y,x)[1],expand_feature_list,[])

    endtime = time.time()
    dtime = endtime - starttime
    print("Expanding time: %.8s s" % dtime) 
    print(len(result))
    print("Finish loading external data")
    return result

def get_top_k_features(args, feature, tokenizer, model, device, external_ids_list, embedding):

    top_k_feature = []
    ori_sent = tokenizer.decode(feature.input_ids,skip_special_tokens=True)
    ori_tokens = tokenizer.encode(ori_sent)
    segment_idxs = [0] * len(ori_tokens)
    input_mask = [1] * len(ori_tokens)
    ori_tokens_tensor = torch.tensor([ori_tokens]).to(device)
    segments_tensor = torch.tensor([segment_idxs]).to(device)
    input_mask_tensor = torch.tensor([input_mask]).to(device)
    with torch.no_grad():
        output = model(ori_tokens_tensor,segments_tensor)
        hidden_states = output["hidden_states"]
        first = hidden_states[1].transpose(1, 2)
        last = hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
        mean_pooled = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

        result = F.cosine_similarity(mean_pooled,embedding)
        _, index = torch.sort(result, descending=True)
        top_k_index = index[:(args.k)]
        for index in top_k_index:
            max_seq_length = len(external_ids_list[index])
            temp_sent = tokenizer.decode(external_ids_list[index],skip_special_tokens=True)
            input_ids = external_ids_list[index]
            input_mask = [1] * (len(tokenizer.tokenize(temp_sent)))
            if args.model_type == 'roberta':
                padding = [1] * (max_seq_length - len(tokenizer.tokenize(temp_sent)))
            else:
                padding = [0] * (max_seq_length - len(tokenizer.tokenize(temp_sent)))
            input_mask += padding
            segment_ids = [0] * max_seq_length

            assert(len(input_ids) == max_seq_length)
            assert(len(input_mask) == max_seq_length)
            assert(len(segment_ids) == max_seq_length)

            top_k_feature.append(InputFeatures(tokens=tokenizer.convert_ids_to_tokens(external_ids_list[index]), 
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=feature.label_id))

    return top_k_feature


if __name__ == "__main__":

    args = parser.parse_args()

    if (args.output_dir == None):
        print("No output directory provided.")    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Output directory: {}".format(args.output_dir))

    print("Loading the model: {}".format(args.model_name_or_path))

    model, tokenizer = load_model_tokenizer(args.model_type, args)
    
    print("Finish loading the model: {}".format(args.model_name_or_path))

    processors = {
        "cola": ColaProcessor,
        "sst-2": Sst2Processor,
        "qnli": QnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "sst-2": "classification",
        "qnli": "classification",
    }


    if args.cuda == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.cuda)
        device = torch.device("cuda", args.cuda)

    model.to(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (not args.no_save):
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    nb_tr_steps = 0    
    tr_loss = 0



    if args.do_train:
        logger.info("Prepare training features")

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        ori_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        ori_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        ori_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            ori_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


        if args.debias_type == 'gender':
            male_words_ = load_word_list(args.data_path+"male.txt")
            female_words_ = load_word_list(args.data_path+"female.txt")
            tar1_words, tar2_words = clean_word_list2(male_words_, female_words_,tokenizer)
            ster_words_ = clean_word_list(load_word_list(args.data_path+"stereotype.txt"),tokenizer)
            aug_features_ori = convert_examples_to_augfeatures(
                        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, tar1_words, tar2_words)
            word_list = male_words_ + female_words_ + ster_words_
            expand_aug_features = get_expand_festure_list(args, train_features + aug_features_ori , tokenizer, model, device, word_list)

        elif args.debias_type=='race':
            race1_words_ = load_word_list(args.data_path+"race1.txt")
            race2_words_ = load_word_list(args.data_path+"race2.txt")

            tar1_words, tar2_words = clean_word_list2(race1_words_, race2_words_,tokenizer)
            ster_words_ = clean_word_list(load_word_list(args.data_path+"stereotype.txt"),tokenizer)
            aug_features_ori = convert_examples_to_augfeatures(
                        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, tar1_words, tar2_words)
            word_list = race1_words_ + race2_words_ + ster_words_
            expand_aug_features = get_expand_festure_list(args, train_features + aug_features_ori , tokenizer, model, device, word_list)

        all_features = train_features + expand_aug_features

        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in all_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in all_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


        jsd_model = JSD()

        model.zero_grad()
        
        model.train()

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                invariant_mask = input_mask.clone()

                tar_list = tar1_words + tar2_words
                for tar_token in tokenizer.convert_tokens_to_ids(tar_list):
                    tar_index = torch.nonzero(torch.where(input_ids == tar_token, torch.tensor(tar_token).to(device), torch.tensor(0).to(device)))
                    invariant_mask.index_put_(tuple(tar_index.t()),torch.tensor([0]).to(device))
                    

                # define a new function to compute loss values for both output_modes
                if args.model_type == 'roberta':
                    output_ori = model(input_ids, input_mask)
                    logits_ori = output_ori["logits"]
                else:
                    output_ori = model(input_ids, segment_ids, input_mask)
                    logits_ori = output_ori["logits"]

                with torch.no_grad():
                    if args.model_type == 'roberta':
                        output_invariant = model(input_ids, invariant_mask)
                        logits_invariant = output_invariant["logits"]
                    else:
                        output_invariant = model(input_ids, segment_ids, invariant_mask)
                        logits_invariant = output_invariant["logits"]


                w_dist = 0

                for i in range(logits_invariant.shape[0]):
                    w_dist += wasserstein_distance(logits_invariant[i].detach().cpu(),logits_ori[i].detach().cpu())
                avg_w_dist = w_dist/(logits_ori.shape[0])


                loss_fct = CrossEntropyLoss()
                

                loss = loss_fct(logits_ori.view(-1, num_labels), label_ids.view(-1)) + args.tau * avg_w_dist

                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                
                tr_loss += loss.item()
                epoch_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss /= len(train_dataloader)
                print("Epoch {}: loss={}".format(epoch, epoch_loss))


        if not args.no_save:
            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/", args.output_dir, "pytorch_model.bin")
            output_config_file = os.path.join(args.output_dir, "config.json")

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)


    if args.do_eval:
        if (not args.do_train):
            # Load a trained model and vocabulary that you have fine-tuned
            if args.model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.output_dir)
                model = BertForSequenceClassification.from_pretrained(args.output_dir)
            elif args.model_type == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained(args.output_dir)
                model = RobertaForSequenceClassification.from_pretrained(args.output_dir)
            elif args.model_type == 'albert':
                tokenizer = AlbertTokenizer.from_pretrained(args.output_dir)
                model = AlbertForSequenceClassification.from_pretrained(args.output_dir)
            else:
                raise NotImplementedError("not implemented!")

            model.to(device)

        if args.debias_type == "gender":
            male_words_ = load_word_list(args.data_path+"male.txt")
            female_words_ = load_word_list(args.data_path+"female.txt")
            tar1_words, tar2_words = clean_word_list2(male_words_, female_words_,tokenizer)
        
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        all_sample_ids = torch.arange(len(eval_features), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sample_ids)
        # Run prediction for full data
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []



        for input_ids, input_mask, segment_ids, label_ids, sample_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            invariant_mask = input_mask.clone()

            tar_list = tar1_words + tar2_words
            for tar_token in tokenizer.convert_tokens_to_ids(tar_list):
                tar_index = torch.nonzero(torch.where(input_ids == tar_token, torch.tensor(tar_token).to(device), torch.tensor(0).to(device)))
                invariant_mask.index_put_(tuple(tar_index.t()),torch.tensor([0]).to(device))

            with torch.no_grad():
                if args.model_type == 'roberta':
                    output_ori = model(input_ids, input_mask)
                    logits_ori = output_ori["logits"]
                else:
                    output_ori = model(input_ids, segment_ids, input_mask)
                    logits_ori = output_ori["logits"]

                if args.model_type == 'roberta':
                    output_invariant = model(input_ids, invariant_mask)
                    logits_invariant = output_invariant["logits"]
                else:
                    output_invariant = model(input_ids, segment_ids, invariant_mask)
                    logits_invariant = output_invariant["logits"]

            w_dist = 0

            for i in range(logits_invariant.shape[0]):
                w_dist += wasserstein_distance(logits_invariant[i].detach().cpu(),logits_ori[i].detach().cpu())
            avg_w_dist = w_dist/(logits_ori.shape[0])


            loss_fct = CrossEntropyLoss()
            
            tmp_eval_loss = loss_fct(logits_ori.view(-1, num_labels), label_ids.view(-1)) + args.tau * avg_w_dist

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits_invariant.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits_invariant.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        preds = np.argmax(preds, axis=1)

        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/global_step if (args.do_train and global_step > 0) else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                if (not args.no_save):
                    writer.write("%s = %s\n" % (key, str(result[key])))