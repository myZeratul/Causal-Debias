from sklearn.metrics import matthews_corrcoef, f1_score

def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst

# tradition metric 
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def clean_word_list2(tar1_words_,tar2_words_,tokenizer):
    tar1_words = []
    tar2_words = []
    for i in range(len(tar1_words_)):
        if tokenizer.convert_tokens_to_ids(tar1_words_[i])!=tokenizer.unk_token_id and tokenizer.convert_tokens_to_ids(tar2_words_[i])!=tokenizer.unk_token_id:
            tar1_words.append(tar1_words_[i])
            tar2_words.append(tar2_words_[i])
    return tar1_words, tar2_words

def clean_word_list(vocabs,tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        if tokenizer.convert_tokens_to_ids(vocabs[i])!=tokenizer.unk_token_id:
            vocab_list.append(vocabs[i])
    return vocab_list
