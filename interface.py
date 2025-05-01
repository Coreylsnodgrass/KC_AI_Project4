# interface.py

import argparse
from generate import generate_text

def main():
   
    parser = argparse.ArgumentParser(
        description="Jesterbot CLI (distilgpt2-based)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default= 80,
        help="Number of tokens to generate (beyond prompt)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K filtering"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus (top-p) threshold"
    )
    args = parser.parse_args()
    print("\n====================================================")
    print("This is Jesterbot (distilgpt2). Type 'quit' to exit.")
    print("====================================================== \n")
    #print("Extra Commands -> --top_k , --top_p, --temperature, --length")
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.strip().lower() == "quit":
            break

        # pass args.length into the max_length param
        joke = generate_text(
            prompt,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\n⟶", joke)

if __name__ == "__main__":
    main()
