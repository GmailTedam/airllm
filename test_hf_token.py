import os
from huggingface_hub import model_info, whoami
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, LocalEntryNotFoundError

def check_token_and_access():
    print("--- Hugging Face Token & Access Checker ---")
    # Fetch the tokens
    token = os.environ.get('HF_TOKEN') or os.environ.get('HF_ACCESS_TOKEN')
    
    if not token:
        print("❌ Error: No token found. Please ensure HF_TOKEN or HF_ACCESS_TOKEN are set in your environment.")
        return

    # 1. Validate the token itself
    try:
        user = whoami(token=token)
        print(f"✅ Token is valid! Authenticated as Hugging Face user: {user.get('name')}")
    except Exception as e:
        print(f"❌ Token validation failed. The token might be invalid or expired.\nDetails: {str(e)}")
        return

    # 2. Check access to Llama 3
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    print(f"\nChecking access to the gated model: {model_id}...")
    try:
        info = model_info(repo_id=model_id, token=token)
        print(f"✅ Success! You have authorized access to {model_id}.")
        print("You are all set to run airllm_example.py!")
    except GatedRepoError as e:
        print(f"❌ Access Denied.")
        print(f"You need to accept the user agreement for {model_id} on the Hugging Face website.")
        print(f"Go to: https://huggingface.co/{model_id}")
        print("Once you accept the terms and are granted access, this script will pass.")
    except RepositoryNotFoundError as e:
        print(f"❌ Error: Model not found or you lack read access.\nDetails: {str(e)}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    check_token_and_access()
