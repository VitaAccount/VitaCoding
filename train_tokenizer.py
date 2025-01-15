from tokenizer import LLMTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output_file", type=str, default="tokenizer.json")
    args = parser.parse_args()
    
    # Create and train tokenizer
    tokenizer = LLMTokenizer(vocab_size=args.vocab_size)
    tokenizer.train([args.input_file])
    
    # Save tokenizer
    tokenizer.save(args.output_file)
    print(f"Tokenizer saved to {args.output_file}")

if __name__ == "__main__":
    main() 