import os

from transformers import BertTokenizer
from utils import get_rank, mkdir, synchronize


class CustomBertTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(CustomBertTokenizer, self).__init__(*args, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True,
               clean_up_tokenization_spaces=True, end_flags=[]):
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            end_flags=end_flags)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(" " + token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ''.join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False, end_flags=[]):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in end_flags:
                tokens.append('.')
                break
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

def get_tokenizer(config):
    if get_rank() != 0:
        synchronize()

    pretrained_cache_dir = '.cache_uncased/'
    bert_base_path = 'bert-base-uncased'  # 30522

    if not os.path.exists(pretrained_cache_dir):
        mkdir(pretrained_cache_dir)
        tokenizer = CustomBertTokenizer.from_pretrained(bert_base_path)
        tokenizer.save_pretrained(save_directory=pretrained_cache_dir)
    else:
        tokenizer = CustomBertTokenizer.from_pretrained(pretrained_cache_dir)

    if get_rank() == 0:
        synchronize()

    SEP = tokenizer.sep_token_id
    PAD = tokenizer.pad_token_id
    MASK = tokenizer.mask_token_id
    EOS = tokenizer.convert_tokens_to_ids('.')
    num_tokens = tokenizer.vocab_size

    return tokenizer, SEP, EOS, MASK, PAD, num_tokens
