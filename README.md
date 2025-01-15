# Custom LLM Chatbot

A complete implementation of a transformer-based language model chatbot built from scratch using PyTorch. This project includes the model architecture, training pipeline, and inference server with a web interface.

## Features

- Custom transformer-based language model implementation
- Byte-Pair Encoding (BPE) tokenizer
- Distributed training support
- FastAPI-based inference server
- Clean web interface for chat interactions
- Temperature and top-k/top-p sampling for response generation

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-chatbot.git
cd llm-chatbot
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

1. Prepare your training data in a text file format.

2. Train the tokenizer and model:
```bash
# First, train the tokenizer on your data
python train_tokenizer.py --input_file path/to/your/data.txt --vocab_size 32000

# Then train the model
python train.py --train_file path/to/your/data.txt --output_dir ./output
```

For distributed training on multiple GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py --train_file path/to/your/data.txt
```

## Running the Server

1. Start the FastAPI server:
```bash
python server.py
```

2. Open `templates/index.html` in your web browser or serve it using a static file server.

## Model Architecture

- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 2048 sequence length
- Learned positional embeddings

## Customization

You can modify the model architecture and training parameters by editing the `LLMConfig` class in `model.py`. Key parameters include:

- `vocab_size`: Size of the tokenizer vocabulary
- `hidden_size`: Dimension of the model's hidden states
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads per layer
- `intermediate_size`: Dimension of the feed-forward network
- `max_position_embeddings`: Maximum sequence length

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 