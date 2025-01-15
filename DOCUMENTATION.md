# Self-Learning LLM Chatbot Documentation

## Project Overview

This project implements a self-learning language model chatbot that continuously improves by learning from Wikipedia articles. The system consists of three main components:

1. **Continuous Learning System** (`web_learner.py`)
2. **Chat Server** (`server.py`)
3. **Web Interface** (`templates/index.html`)

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure
```
├── web_learner.py        # Continuous learning system
├── server.py             # FastAPI server
├── templates/
│   └── index.html        # Web interface
└── requirements.txt      # Project dependencies
```

## Running the System

### 1. Start the Continuous Learning Process

```bash
# In terminal 1
python web_learner.py
```

This will:
- Download a pre-trained model (facebook/opt-350m)
- Fetch random Wikipedia articles
- Continuously fine-tune the model
- Save improvements to ./continuous_learner/

### 2. Start the Chat Server

```bash
# In terminal 2
python server.py
```

This will:
- Load the latest model from ./continuous_learner/
- Start the FastAPI server on http://localhost:8000

### 3. Access the Web Interface
Open `templates/index.html` in a web browser to interact with the chatbot.

## Component Details

### Continuous Learning System (web_learner.py)

The system uses the following parameters which you can modify:

```python
training_args = TrainingArguments(
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
```

Key parameters to adjust:
- `num_train_epochs`: Number of training epochs per batch of articles
- `per_device_train_batch_size`: Batch size for training
- `learning_rate`: Learning rate for fine-tuning
- `num_articles`: Number of articles to fetch per iteration (in train() method)

### Chat Server (server.py)

The server provides a REST API with the following endpoint:

```
POST /chat
{
    "message": "Your message here",
    "temperature": 0.7,        # Controls randomness (0.0 - 1.0)
    "max_length": 512,         # Maximum response length
    "top_p": 0.9,             # Nucleus sampling parameter
    "top_k": 50               # Top-k sampling parameter
}
```

### Model Architecture

The project uses the OPT-350M model which has:
- 350 million parameters
- 24 transformer layers
- 1024 hidden dimensions
- 16 attention heads

## Customization Options

### 1. Change Base Model

To use a different pre-trained model:

```python
# In web_learner.py
class WebLearner:
    def __init__(self, model_name="facebook/opt-1.3b"):  # Change model here
```

Available options:
- facebook/opt-125m (smaller, faster)
- facebook/opt-1.3b (better quality)
- facebook/opt-2.7b (even better quality)

### 2. Modify Learning Sources

The system currently learns from Wikipedia. To add more sources:

1. Create new data fetcher methods in `WebLearner`:
```python
def get_custom_data(self):
    # Implement custom data fetching
    pass
```

2. Modify the train() method to use multiple sources:
```python
def train(self):
    articles = self.get_random_wikipedia_articles()
    custom_data = self.get_custom_data()
    combined_data = articles + custom_data
    dataset = self.prepare_dataset(combined_data)
```

### 3. Adjust Training Parameters

For faster learning but potentially lower quality:
```python
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
)
```

For slower learning but potentially better quality:
```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
)
```

## Hardware Requirements

Minimum:
- 8GB RAM
- CUDA-capable GPU with 6GB VRAM
- 20GB disk space

Recommended:
- 16GB RAM
- CUDA-capable GPU with 8GB+ VRAM
- 50GB+ disk space

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce `per_device_train_batch_size`
   - Reduce `max_length` in tokenizer
   - Use a smaller base model

2. **Slow Learning**:
   - Increase `learning_rate`
   - Decrease `gradient_accumulation_steps`
   - Reduce `num_articles`

3. **Poor Response Quality**:
   - Decrease `temperature` in generation
   - Increase `num_train_epochs`
   - Use a larger base model

## Future Improvements

1. **Data Sources**:
   - Add support for custom datasets
   - Implement web scraping for specific domains
   - Add support for PDF/document learning

2. **Model Improvements**:
   - Implement RLHF (Reinforcement Learning from Human Feedback)
   - Add model quantization for faster inference
   - Implement model pruning for efficiency

3. **System Features**:
   - Add conversation memory
   - Implement user feedback loop
   - Add support for multiple languages

## Security Considerations

1. **API Security**:
   - Add authentication to the API
   - Implement rate limiting
   - Add input validation and sanitization

2. **Model Safety**:
   - Implement content filtering
   - Add toxic language detection
   - Implement prompt injection protection

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Follow these guidelines:
- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow PEP 8 style guide 