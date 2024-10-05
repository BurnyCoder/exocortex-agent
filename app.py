from exocortex_agent import ExocortexAgent

exocortex_agent = ExocortexAgent()

def run_terminal():
    print("Welcome to the Exocortex Agent. Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        output = exocortex_agent.run(user_input)
        print("\nAssistant:", output)

if __name__ == "__main__":
    run_terminal()
    