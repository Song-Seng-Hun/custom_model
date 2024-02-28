# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import List, Optional

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: Optional[str]):
        # Reload tokenizer.
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        ## using special tokens with <unused~>
        self.unused_ids = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 255999]
        self.mask_id: int = 7 # masking token
        self.sep_id: int = 8 # split order token
        self.cls_id: int = 9 # token for if it is needed
        self.ref_id: int = 10 # token for referencing ("You are ~")
        self.bod_id: int = 11 # begin of data (data1(text, image, audio), data2(text, image, audio)...)
        self.bot_id: int = 11 # begin of text
        self.boi_id: int = 12 # begin of image
        self.boa_id: int = 13 # begin of audio
        self.bov_id: int = 14 # begin of video
        self.bop_id: int = 15 # begin of point cloud
        self.bom_id: int = 16 # begin of mesh
        self.bog_id: int = 17 # begin of graph
        self.bohp_id: int = 18 # begin of human pose / animal pose / robot pose
        self.bod_id: int = 19 # begin of digit
        self.bops_id: int = 20 # begin of pose (position + orientation)
        self.boc_id: int = 21 # begin of control
        self.bon_id: int = 22 # begin of noise (timestamp)
        self.bopy_id: int = 23 # begin of python
        self.bocp_id: int = 24 # begin of cpp
        self.bocs_id: int = 25 # begin of csharp
        self.boh_id: int = 26 # begin of http
        self.bodf_id: int = 27 # begin of csv, dataframe
        self.box_id: int = 28 # begin of xml / urdf / xacro 
        self.boj_id: int = 29 # begin of json / topic
        self.bol_id: int = 30 # begin of list
        self.sit_id: int = 31 # description of situation (what is this image, what is this sound..., who is talking...)
        self.lan_id: int = 32 # description of language data (text in image, meaning of talking...)
        self.mel_id: int = 33 # melody of sound
        self.loss_id: int = 34 # loss for rl
        self.gen_id: int = 35 # generation Q&A
        self.inst_id: int = 36 # for chat format (request)
        self.think_id: int = 37 # for logics
        self.ans_id: int = 38 # final answer to Question (response)
        self.sil_id: int = 39 # silence answer
        self.bbox_id: int = 40 # bbox
        self.seg_id: int = 41 # segment
        self.pose6d_id: int = 42 # 6d pose
        self.tool_id: int = 43 # tool / function
        self.search_id: int = 44 # search
        self.label_id: int = 45 # label
        self.watching_id: int = 46 # eyetracking
        self.bobw_id: int = 47 # begin of brainwave

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.decode(t)


if __name__ == "__main__":
    tokenizer = Tokenizer('tokenizer/tokenizer.model')
    unused_prompt = ""
    for i in range(0, 100):
        unused_prompt += f"<unused{i}>" 
    
    # 토큰화 및 토큰 ID 확인
    token_ids = tokenizer.encode(unused_prompt)[1:]
    tokens = [tokenizer.decode(t) for t in token_ids]

    print(f"token ID: {token_ids}, total {len(token_ids)}")
    print(f"token: {tokens}, total {len(tokens)}")
