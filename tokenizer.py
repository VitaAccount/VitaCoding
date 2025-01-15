from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from typing import List, Optional
import os

class LLMTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.bos_token = "[BOS]"
        self.unk_token = "[UNK]"
        
    def train(self, files: List[str], min_frequency: int = 2):
        """Train the tokenizer on a list of files"""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=[
                self.pad_token,
                self.eos_token,
                self.bos_token,
                self.unk_token
            ]
        )
        self.tokenizer.train(files, trainer)
        
        # Add post-processor for special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} $B:1 {self.eos_token}:1",
            special_tokens=[
                (self.bos_token, self.tokenizer.token_to_id(self.bos_token)),
                (self.eos_token, self.tokenizer.token_to_id(self.eos_token)),
            ],
        )
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token ids"""
        encoded = self.tokenizer.encode(text)
        if max_length:
            encoded.truncate(max_length)
        return encoded.ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text"""
        return self.tokenizer.decode(ids)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'LLMTokenizer':
        """Load tokenizer from file"""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance 