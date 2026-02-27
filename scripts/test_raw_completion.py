import urllib.request
import json
import sys

def main():
    url = "http://localhost:8080/completion"
    
    # DeepSeek-R1 raw prompt format
    # <｜User｜>Message<｜Assistant｜><think>
    # Note: We put <think> at the end to force the model to continue reasoning
    
    prompt = "<｜User｜>Welche Voraussetzungen müssen erfüllt sein, damit die AIT als akkreditierte Prüfstelle gilt?<｜Assistant｜><think>"
    
    print(f"Sending raw prompt to {url}...")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    data = json.dumps({
        "prompt": prompt,
        "n_predict": 500,
        "temperature": 0.6,
        "stop": ["<｜end▁of▁sentence｜>"]
    }).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result.get('content', '')
            
            print("\nModel Output:")
            print("-" * 50)
            # We manually added <think>, so prepend it to output to see full flow
            print(f"<think>{content}")
            print("-" * 50)
            
            if "</think>" in content:
                print("\nSUCCESS: Model generated a closing </think> tag!")
            else:
                print("\nWARNING: No </think> tag found (chunk might be too short or model failed to reason).")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
