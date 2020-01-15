""" Handles the personachat data.

"""
import os
import glob
import tqdm
import pickle

import numpy as np
import tensorflow as tf

from itertools import chain
from utils import InputFeatures

DEFAULT_PATH_TO_DATA = "../datasets/personachat"

SPECIAL_TOKENS = ["<SST>", "<END>", "<PAD>", "<SPK:S>", "<SPK:O>", "<PERS>"]
ATTR_TO_SPECIAL_TOKENS = {"bos_token": "<SST>", "eos_token": "<END>", "pad_token": "<PAD>",
                          "additional_special_tokens": ("<SPK:S>", "<SPK:O>", "<PERS>")}


class Personachat:
    def __init__(self, tokenizer, path_to_data=None):
        if path_to_data is not None:
            self.path_to_data = path_to_data
        else:
            self.path_to_data = DEFAULT_PATH_TO_DATA
        pass
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKENS)
        self.spk_s, self.spk_o, self.pers = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[3:])
        self.dataset = None

        self.KEYS = ["utterances", "persona_1_original", "persona_1_revised", "persona_2_original",
                     "persona_2_revised", "wrong_utterances"]
        self.KEYS_SAMPLE = ["label_utterance", "dialogue_history", "personas_original",
                            "personas_revised", "wrong_utterances"]

    def load_txt_dataset(self):
        """ Loads the datafiles from the original dataset and decodes it for further processing. """
        # --- load the original files ---------------------------------------------------------------------------------
        datafiles_original = glob.glob(os.path.join(self.path_to_data, "raw/*both_original.txt"))
        datafiles_revised = glob.glob(os.path.join(self.path_to_data, "raw/*both_revised.txt"))
        corpus_raw_input = {
            "train_original": [],
            "train_revised": [],
            "valid_original": [],
            "valid_revised": [],
            "test_original": [],
            "test_revised": []
        }
        for datafile, datafile_revised in zip(datafiles_original, datafiles_revised):
            with open(datafile, 'r') as f, open(datafile_revised, 'r') as f2:
                if "train" in datafile:
                    corpus_raw_input['train_original'] = f.readlines()
                    corpus_raw_input['train_revised'] = f2.readlines()
                elif "valid" in datafile:
                    corpus_raw_input['valid_original'] = f.readlines()
                    corpus_raw_input['valid_revised'] = f2.readlines()
                elif "test" in datafile:
                    corpus_raw_input['test_original'] = f.readlines()
                    corpus_raw_input['test_revised'] = f2.readlines()

        # --- processes the dataset into structured dialogues ---------------------------------------------------------
        corpus_structured_dialogues = {
            "train": [],
            "valid": [],
            "test": []
        }
        for split in ["train", "valid", "test"]:
            dialogue = None
            for line_original, line_revised in zip(corpus_raw_input[split + "_original"],
                                                   corpus_raw_input[split + "_revised"]):
                num, ctype, content = Personachat._parse_raw_data_line(line_original)
                _, _, content_revised = Personachat._parse_raw_data_line(line_revised)  # num, ctype is the same

                if num == 1:  # begin of a new dialogue (every dialogue is enumerated from 1 to X)
                    if dialogue is not None:  # if a dialogue is already processed, add to the list
                        corpus_structured_dialogues[split].append(dialogue)
                    dialogue = {
                        "utterances": [],
                        "persona_1_original": [],
                        "persona_1_revised": [],
                        "persona_2_original": [],
                        "persona_2_revised": [],
                        "wrong_utterances": []
                    }

                if ctype is not "dialogue":
                    dialogue[ctype + "_original"].append(content)
                    dialogue[ctype + "_revised"].append(content_revised)
                else:
                    dialogue["utterances"] += content["dial_utts"]
                    dialogue["wrong_utterances"] += content["wrong_utts"]
            corpus_structured_dialogues[split].append(dialogue)  # add the last dialogue to the list
        self.dataset = corpus_structured_dialogues

    def tokenize_dataset(self, path_to_save):
        """ This function tokenizes the data and saves the dataset.
        Only for saving some time, when starting the training scripts multiple times (e.g. for debugging).
        """
        assert self.dataset is not None  # Dataset need to be loaded first!

        for split, dialogues in self.dataset.items():
            print("Processing {} data ...".format(split))
            for dialogue in tqdm.tqdm(dialogues):
                for key in self.KEYS:
                    dialogue[key + "_binarized"] = [self.tokenizer.encode(d) for d in dialogue[key]]

        # --- saving data ---
        path = os.path.join(self.path_to_data, path_to_save)
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(self.dataset, f)

    def get_tensorflow_features(self, split, current_dataset=True):
        """ Creates a tensorflow dataset. """
        if not current_dataset:
            raise NotImplementedError("Currently not implemented! :D")

        samples = Personachat._convert_dialogue_to_samples(self.dataset[split][0])

        # check if data is binarized
        # We assume that if at least one binarized key is available, the others are available as well.
        if "dialogue_history_binarized" not in samples[0]:
            for sample in samples:
                for key in self.KEYS_SAMPLE:
                    sample[key + "_binarized"] = [self.tokenizer.encode(d) for d in sample[key]]

        # create features
        features = []
        padding_length = 256

        def pad(x, padding):
            return x + [padding] * (padding_length - len(x))

        for sample in samples:
            seqs = self._convert_sample_to_sequences(persona=sample["personas_original_binarized"],
                                                     history=sample["dialogue_history_binarized"],
                                                     reply=sample["label_utterance_binarized"][0],
                                                     lm_labels=True)
            seqs_distractor = self._convert_sample_to_sequences(persona=sample["personas_original_binarized"],
                                                                history=sample["dialogue_history_binarized"],
                                                                reply=sample["wrong_utterances_binarized"][0],
                                                                lm_labels=False)

            # padding
            (input_ids, input_ids_distractor,
             token_type_ids, token_type_ids_distractor) = [pad(x, self.tokenizer.pad_token_id)
                                                           for x in (seqs["input_ids"], seqs_distractor["input_ids"],
                                                                     seqs["token_type_ids"],
                                                                     seqs_distractor["token_type_ids"])]
            (lm_labels, lm_labels_distractor) = [pad(x, -1) for x in (seqs["lm_labels"],
                                                                      seqs_distractor["lm_labels"])]

            # features.append(InputFeatures(input_ids=input_ids,
            #                               token_type_ids=token_type_ids,
            #                               mc_token_ids=seqs["mc_token_ids"],
            #                               label=lm_labels))
            features.append(InputFeatures(input_ids=[input_ids, input_ids_distractor],
                                          token_type_ids=[token_type_ids, token_type_ids_distractor],
                                          mc_token_ids=[seqs["mc_token_ids"], seqs_distractor["mc_token_ids"]],
                                          label=[lm_labels, lm_labels_distractor]))

        # create tensorflow dataset
        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "token_type_ids": ex.token_type_ids,
                        "mc_token_ids": ex.mc_token_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "token_type_ids": tf.int32, "mc_token_ids": tf.int32}, tf.int32),
            (
                {
                    "input_ids": tf.TensorShape([2, None]),
                    "token_type_ids": tf.TensorShape([2, None]),
                    "mc_token_ids": tf.TensorShape([2]),
                },
                tf.TensorShape([2, None])
            )
        )

    def load_dataset(self, path):
        """ Loads a (already binarized) dataset into the object. """
        with open(path, "rb") as f:
            ds = pickle.load(f)
            for split in ["train", "valid", "test"]:
                if split not in ds:
                    raise Exception("Dataset does not contain {} data.".format(split))
                if "utterances_binarized" not in ds[split][0]:
                    raise Exception("Data is not binarized!")
            self.dataset = ds

    def _convert_sample_to_sequences(self, persona, history, reply, lm_labels=True):
        """ This takes as input one sample and generates sequences for the GPT models. """

        sequence = [[self.tokenizer.bos_token_id] + list(chain(*persona))] + history + \
                   [reply + [self.tokenizer.eos_token_id]]
        sequence = [sequence[0]] + [[self.spk_s if (len(sequence)-i) % 2 else self.spk_o] + s
                                    for i, s in enumerate(sequence[1:])]
        seqs = {
            "input_ids": list(chain(*sequence))
        }

        def cond(i):
            if i == 0:
                # personality tokens for the first list (concatenated personalities)
                return self.pers
            if i % 2:
                # system speaker tokens for even entries
                return self.spk_s
            else:
                # other speaker tokens for odd entries
                return self.spk_o
        seqs["token_type_ids"] = [cond(i) for i, s in enumerate(sequence) for _ in s]

        seqs["mc_token_ids"] = len(seqs["input_ids"]) - 1
        seqs["lm_labels"] = [-1] * len(seqs["input_ids"])
        if lm_labels:
            seqs["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return seqs

    @staticmethod
    def _convert_dialogue_to_samples(dialogue, binarized=False, num_wrong_utts=1):
        """ Converts one dialogue in all possible samples, given some settings.
        """
        if binarized:
            bin_key = "_binarized"
        else:
            bin_key = ""
        samples = []
        for num in range(len(dialogue["utterances" + bin_key]) - 1):
            if num % 2 == 0:
                persona_num = "2"
            else:
                persona_num = "1"
            sst = num_wrong_utts * num
            end = sst + num_wrong_utts
            samples.append({
                "label_utterance": [dialogue["utterances" + bin_key][num + 1]],
                "dialogue_history": dialogue["utterances" + bin_key][0:num + 1],
                "personas_original": dialogue["persona_" + persona_num + "_original" + bin_key],
                "personas_revised": dialogue["persona_" + persona_num + "_revised" + bin_key],
                "wrong_utterances": dialogue["wrong_utterances" + bin_key][sst:end]
            })
        return samples

    @staticmethod
    def _parse_raw_data_line(line):
        """ Extracts all information from a line of the personachat dataset. """
        num = int(line.split(" ")[0])
        if "your persona: " in line:
            sentence = line.split("your persona: ")[-1].rstrip()
            return num, "persona_2", sentence
        elif "partner\'s persona: " in line:
            sentence = line.split("partner\'s persona: ")[-1].rstrip()
            return num, "persona_1", sentence
        else:
            utterances = line.split("\t")
            dial_utterances = [
                utterances[0].split(" ", 1)[-1].rstrip(),
                utterances[1].rstrip()
            ]
            wrong_utterances = utterances[-1].split("|")
            wrong_utterances[-1] = wrong_utterances[-1].rstrip()
            return num, "dialogue", {"dial_utts": dial_utterances, "wrong_utts": wrong_utterances}


# --- test class ------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #from sources import tokenizer_gpt2
    from transformers import tokenization_gpt2 as tokenizer_gpt2

    tokenizerGPT2 = tokenizer_gpt2.GPT2Tokenizer(vocab_file="../data/gpt2-vocab.json",
                                                 merges_file="../data/gpt2-merges.txt")
    ds = Personachat(tokenizerGPT2)
    ds.load_txt_dataset()

    # ds.tokenize_dataset(tokenizer=tokenizerGPT2, path_to_save="personachat_gpt2_tokenized.pkl")
    ds.load_dataset("../datasets/personachat/personachat_gpt2_tokenized.pkl")
    ds.get_tensorflow_features("train")
