import logging

from models.amber.pre_trained_tokenizer import PreTrainedTokenizer


class PreTrainedTokenizerNoModality(PreTrainedTokenizer):
    def __init__(
        self, block_size: int, pre_trained_tokenizer: str, shuffle_findings: bool
    ):
        super().__init__(
            block_size=block_size,
            pre_trained_tokenizer=pre_trained_tokenizer,
            shuffle_findings=shuffle_findings,
        )

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

        # <finding>mass shape : oval </finding> <finding> mass size: 2.5 cm </finding> <birads> 4 </birads>
        encoded_text = (
            encoded_findings
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

    def pretty_decode(
        self, ids: list, return_structured_predictions: bool = False
    ) -> str | dict:
        toks = self.tokenizer.decode(ids, skip_special_tokens=False)

        try:
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

            pretty_report = f"Birads:{birads}\nFindings:{findings}"
        except Exception as e:
            logging.error(
                f"Could not find the core structured tokens in the generated report. Returning un-prettified report.\nError:{e}"
            )
            pretty_report = self.decode(ids)
            birads = ""
            findings = ""

        if return_structured_predictions:
            return {
                "pretty_report": pretty_report,
                "birads": birads,
                "modality": "",  # No modality prediction in this tokenizer
                "findings": findings,
            }

        return pretty_report
