import datasets
import torchtext
import os

def get_data(dataset, vocab, batch_size):
    '''
    Implementing the Dataloader
    given a dataset gives a way to iterate over batches of it (In a batch, all examples are processed in parallel)
    '''
    data = []
    for example in dataset:
        if example['tokens']:
            # appends each sequence of tokenized text with an <eos> token to mark its end
            tokens = example['tokens'].append('<eos>')
            # encodes each token to a numerical value equal to its index in the vocabulary; rare words match to unknown token
            tokens = [vocab[token] for token in example['tokens']]
            data.extend(tokens)
    # combines all the numerical sequences into a list (1D Tensor)
    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    # reshapes it into a 2D tensor of dimensions [batch_size, num_batches]
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data

def get_vocab_tokenizer():
    #  load data
    files = []
    path = "data"
    for file in os.listdir(f"./{path}"):
        if "rabbit" in file or "red" in file: continue
        files.append(f"{path}/{file}")
    # dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')

    dataset = datasets.load_dataset('text', data_files={'train': files,
                                                        #'test': "/content/drive/MyDrive/AI/pschloss-CSE30124-Fall2023-submissions /Project/test/red_fairybook_parsed.txt",
                                                        #'validation': "/content/drive/MyDrive/AI/pschloss-CSE30124-Fall2023-submissions /Project/validation/peter_rabbit_parsed.txt"
                                                        })

    # tokenize data, basically breaks into words and punctuation here
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'],fn_kwargs={'tokenizer': tokenizer})
    
    # create vocab of any word that occurs at least 3 times
    # length will be the number of neurons in the output classification layer
    vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'],min_freq=3)
    # manually add an <unk> token and set is as the default index so that whenever we request from the vocabulary the index of a word that it doesn’t have we get <unk>
    vocab.insert_token('<unk>', 0)
    # add <eos> token; We will later insert it at the end of each sequence so model will learn to do so as well
    vocab.insert_token('<eos>', 1)
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tokenizer