import logging
import random

import torch
from tokenizers import Tokenizer

from components.layers.decoder import causal_mask_without_pads


class PreTrainedTokenizer:
    _COL = ":"
    _SEMICOL = ";"
    _PAD = "<pad>"
    _SOS = "<sos>"
    _EOS = "<eos>"
    _UNK = "<unk>"

    _BIRADS_START = "<birads>"
    _BIRADS_END = "</birads>"
    _MODALITY_START = "<modality>"
    _MODALITY_END = "</modality>"
    _FINDING_START = "<finding>"
    _FINDING_END = "</finding>"

    SPECIAL_TOKENS = [
        _PAD,
        _SOS,
        _EOS,
        _UNK,
        _COL,
        _SEMICOL,
        _BIRADS_START,
        _BIRADS_END,
        _MODALITY_START,
        _MODALITY_END,
        _FINDING_START,
        _FINDING_END,
    ]

    ADDED_SPECIAL_TOKENS = [
        _BIRADS_START,
        _BIRADS_END,
        _MODALITY_START,
        _MODALITY_END,
        _FINDING_START,
        _FINDING_END,
    ]

    def __init__(
        self, block_size: int, pre_trained_tokenizer: str, shuffle_findings: bool
    ):
        super(PreTrainedTokenizer, self).__init__()
        self.block_size = block_size
        self.tokenizer = Tokenizer.from_pretrained(identifier=pre_trained_tokenizer)
        self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        self.ADDED_SPECIAL_TOKENS_IDS = [
            self.tokenizer.token_to_id(t) for t in self.ADDED_SPECIAL_TOKENS
        ]
        self.shuffle_findings = shuffle_findings
        self.training = shuffle_findings

    def train(self):
        self.training = self.shuffle_findings

    def eval(self):
        self.training = False

    def encode_findings(self, findings: str) -> list[int]:
        """Encode the findings part of the report."""
        result = []
        split_findings = findings.split(self._SEMICOL)
        if self.training:
            random.shuffle(split_findings)
        for finding in split_findings:
            finding = finding.strip()
            if finding:
                encoded_finding = self.tokenizer.encode(
                    finding, add_special_tokens=False
                ).ids
                result.extend(
                    [self.get_finding_start_token_id()]
                    + encoded_finding
                    + [self.get_finding_end_token_id()]
                )
        return result

    def encode(
        self,
        encoded_findings: list[int],
        birad: str,
        modality: str,
        include_sos: bool = True,
        include_eos: bool = True,
    ):
        sos_token_id = self.get_sos_token_id()
        eos_token_id = self.get_eos_token_id()

        encoded_findings = encoded_findings[-self.block_size :]

        # <modality> US </modality> <finding>mass shape : oval </finding> <finding> mass size: 2.5 cm </finding> <birads> 4 </birads>
        encoded_text = (
            [self.get_modality_start_token_id()]
            + self.tokenizer.encode(modality, add_special_tokens=False).ids
            + [self.get_modality_end_token_id()]
            + encoded_findings
            + [self.get_birads_start_token_id()]
            + self.tokenizer.encode(birad, add_special_tokens=False).ids
            + [self.get_birads_end_token_id()]
        )

        if include_sos:
            encoded_text = [sos_token_id] + encoded_text

        if include_eos:
            encoded_text = encoded_text + [eos_token_id]

        encoded_text = encoded_text + [
            self.get_pad_token_id() for _ in range(self.block_size - len(encoded_text))
        ]
        return encoded_text

    def encode_batch(self, reports, birads, modalities):
        input_tokens = []
        input_masks = []
        target_tokens = []

        for i in range(len(reports)):
            findings_toks = self.encode_findings(findings=reports[i])

            # encoder input
            curr_tokens = self.encode(
                encoded_findings=findings_toks,
                birad=birads[i],
                modality=modalities[i],
                include_sos=True,
                include_eos=False,
            )
            curr_tokens = torch.tensor(data=curr_tokens)
            curr_mask = causal_mask_without_pads(
                tokens=curr_tokens, pad_token_index=self.get_pad_token_id()
            )
            input_tokens.append(curr_tokens)
            input_masks.append(curr_mask)

            # encoder target
            curr_target = self.encode(
                encoded_findings=findings_toks,
                birad=birads[i],
                modality=modalities[i],
                include_sos=False,
                include_eos=True,
            )
            curr_target = torch.tensor(data=curr_target)
            target_tokens.append(curr_target)

        return (
            torch.stack(input_tokens),
            torch.stack(input_masks),
            torch.stack(target_tokens),
        )

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def decode(self, ids: list, skip_special_tokens=True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def pretty_decode(
        self, ids: list, return_structured_predictions: bool = False
    ) -> str | dict:
        toks = self.tokenizer.decode(ids, skip_special_tokens=False)

        try:
            mod_s = toks.index(self._MODALITY_START)
            mod_e = toks.index(self._MODALITY_END)
            modality = toks[mod_s + len(self._MODALITY_START) : mod_e].strip()

            birads_s = toks.index(self._BIRADS_START)
            birads_e = toks.index(self._BIRADS_END)
            birads = toks[birads_s + len(self._BIRADS_START) : birads_e].strip()

            find_s = toks.find(self._FINDING_START)
            if find_s != -1:
                findings = toks[find_s + len(self._FINDING_START) : birads_s].strip()
                findings = (
                    findings.replace(self._FINDING_START, "")
                    .replace(self._FINDING_END, f" {self._SEMICOL} ")
                    .strip()
                )
            else:
                findings = ""

            pretty_report = f"Modality:{modality}\nBirads:{birads}\nFindings:{findings}"
        except Exception as e:
            logging.error(
                f"Could not find the core structured tokens in the generated report. Returning un-prettified report.\nError:{e}"
            )
            pretty_report = self.decode(ids)
            birads = ""
            modality = ""
            findings = ""

        if return_structured_predictions:
            return {
                "pretty_report": pretty_report,
                "birads": birads,
                "modality": modality,
                "findings": findings,
            }

        return pretty_report

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def load(self, fp: str) -> None:
        self.tokenizer = Tokenizer.from_file(fp)

    def get_pad_token_id(self):
        return self.tokenizer.token_to_id(self._PAD)

    def get_sos_token_id(self):
        return self.tokenizer.token_to_id(self._SOS)

    def get_eos_token_id(self):
        return self.tokenizer.token_to_id(self._EOS)

    def get_col_token_id(self):
        return self.tokenizer.token_to_id(self._COL)

    def get_semicol_token_id(self):
        return self.tokenizer.token_to_id(self._SEMICOL)

    def get_unk_token_id(self):
        return self.tokenizer.token_to_id(self._UNK)

    # GET ADDITIONAL TOKEN IDS
    def get_modality_start_token_id(self):
        return self.tokenizer.token_to_id(self._MODALITY_START)

    def get_modality_end_token_id(self):
        return self.tokenizer.token_to_id(self._MODALITY_END)

    def get_birads_start_token_id(self):
        return self.tokenizer.token_to_id(self._BIRADS_START)

    def get_birads_end_token_id(self):
        return self.tokenizer.token_to_id(self._BIRADS_END)

    def get_finding_start_token_id(self):
        return self.tokenizer.token_to_id(self._FINDING_START)

    def get_finding_end_token_id(self):
        return self.tokenizer.token_to_id(self._FINDING_END)
