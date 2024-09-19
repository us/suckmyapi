import asyncio
from openai_ducker import main as openai_main
from anthropic_ducker import main as anthropic_main

def main():
    print("Choose an option:")
    print("1. OpenAI")
    print("2. Anthropic")
    print("3. Both 😈")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        run_openai()
    elif choice == '2':
        run_anthropic()
    elif choice == '3':
        run_both()
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")

def run_openai():
    # Placeholder for OpenAI main function
    print("Running OpenAI main function...")
    asyncio.run(openai_main())

def run_anthropic():
    # Placeholder for Anthropic main function
    print("Running Anthropic main function...")
    asyncio.run(anthropic_main())

def run_both():
    # Run both OpenAI and Anthropic main functions
    print("Running both OpenAI and Anthropic main functions...")
    asyncio.run(run_all())

async def run_all():
    await asyncio.gather(openai_main(), anthropic_main())

if __name__ == "__main__":
    main()
