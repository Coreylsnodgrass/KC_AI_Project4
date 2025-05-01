import argparse
from generate import generate_text  # Import text generation function

def main():
    # Set up argument parser for optional generation controls
    parser = argparse.ArgumentParser(
        description="Jesterbot CLI (distilgpt2-based)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=80,
        help="Number of tokens to generate (beyond prompt)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more conservative generation)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K sampling: restricts selection to top K tokens"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold"
    )

    args = parser.parse_args()

    # Begin interactive prompt session
    print("\n====================================================")
    print("This is Jesterbot (distilgpt2). Type 'quit' to exit.")
    print("====================================================\n")

    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.strip().lower() == "quit":
            break

        # Generate and print model output
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
