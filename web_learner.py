import wikipediaapi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import random
from tqdm import tqdm
import time

class WebLearner:
    def __init__(self, model_name="facebook/opt-350m"):
        self.wiki = wikipediaapi.Wikipedia('en')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=True if torch.cuda.is_available() else False,
        )
    
    def get_random_wikipedia_articles(self, num_articles=10):
        """Fetch random Wikipedia articles"""
        articles = []
        for _ in tqdm(range(num_articles), desc="Fetching articles"):
            while True:
                page = self.wiki.page(self.wiki.random())
                if page.exists() and len(page.text) > 100:
                    articles.append({
                        'text': page.text[:2048]  # Limit length to prevent memory issues
                    })
                    break
                time.sleep(0.1)  # Be nice to Wikipedia's servers
        return articles
    
    def prepare_dataset(self, articles):
        """Convert articles to a HuggingFace dataset"""
        dataset = Dataset.from_list(articles)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True,
        )
        return tokenized_dataset
    
    def train(self, num_articles=100):
        """Continuously train on Wikipedia articles"""
        while True:
            print("\nFetching new articles...")
            articles = self.get_random_wikipedia_articles(num_articles)
            dataset = self.prepare_dataset(articles)
            
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=dataset,
            )
            
            trainer.train()
            
            # Save the model
            self.model.save_pretrained("./continuous_learner")
            self.tokenizer.save_pretrained("./continuous_learner")
            print("\nModel saved. Starting next iteration...")
            
            # Optional: Sleep to prevent overwhelming Wikipedia's servers
            time.sleep(60)

if __name__ == "__main__":
    learner = WebLearner()
    try:
        learner.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        learner.model.save_pretrained("./continuous_learner")
        learner.tokenizer.save_pretrained("./continuous_learner") 