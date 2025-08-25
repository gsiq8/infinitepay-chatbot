# StopIteration Reproduction – Hugging Face Inference

This branch reproduces the `StopIteration` error when using Hugging Face Inference in place of sentence transformers.

## Setup
1. Clone repo and checkout this branch:

      git clone <[https://github.com/gsiq8/infinitepay-chatbot.git]>

      git checkout bug/stopiteration-hf-inference

3. Create `.env` from `.env.example` and add your keys.

4. Install deps:
   pip install -r requirements.txt

5. Run:
   uvicorn backend.main:app --reload
   * or python backend/repro.py (if you isolate into a single script)

## Logs
Logs showing the error: [https://gist.github.com/gsiq8/1718280a5b0495ac68a8a4f990279c2e]
