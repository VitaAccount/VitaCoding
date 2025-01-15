import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
from typing import Optional, List
from model import LLM, LLMConfig
from tokenizer import LLMTokenizer
import json

class TextDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: LLMTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = f.readlines()
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        text = self.examples[i].strip()
        encodings = self.tokenizer.encode(
            text,
            max_length=self.max_length,
        )
        
        input_ids = torch.tensor(encodings[:-1], dtype=torch.long)
        labels = torch.tensor(encodings[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

def train(
    config: LLMConfig,
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = 'output',
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    warmup_steps: int = 1000,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    local_rank: int = -1,
):
    # Initialize distributed training
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    if local_rank in [-1, 0]:
        wandb.init(project="llm-training", config=vars(config))
    
    # Create model
    model = LLM(config)
    model.to(device)
    
    if local_rank != -1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    
    # Create tokenizer and datasets
    tokenizer = LLMTokenizer(vocab_size=config.vocab_size)
    train_dataset = TextDataset(
        train_file,
        tokenizer,
        max_length=config.max_position_embeddings,
    )
    
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if local_rank != -1
        else None
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * num_epochs,
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        if local_rank != -1:
            train_sampler.set_epoch(epoch)
            
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=local_rank not in [-1, 0],
        )
        
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, _ = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, config.vocab_size),
                labels.view(-1),
            )
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if local_rank in [-1, 0]:
                    wandb.log({
                        'loss': loss.item() * gradient_accumulation_steps,
                        'lr': scheduler.get_last_lr()[0],
                    })
                    
                    if global_step % 1000 == 0:
                        # Save model
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            
                        model_to_save = (
                            model.module if hasattr(model, 'module') else model
                        )
                        
                        torch.save(
                            model_to_save.state_dict(),
                            os.path.join(output_dir, f'model_{global_step}.pt'),
                        )
                        
    if local_rank in [-1, 0]:
        wandb.finish()

if __name__ == "__main__":
    config = LLMConfig()
    train(config, "train.txt") 