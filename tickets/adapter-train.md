# Ticket: adapter-train

Branch: `feature/adapter-train`
Issue: #5

## Request

An MPS trainer for the self-denial adapter, new file src/finetune\_adapter.py. Serves\
MVP Step 3. Port third\_party/Pain-axis/scripts/4.3\_selfmed/01\_finetune\_self\_report.py\
(MIT, keep the notice, never edit third\_party). Theirs is CUDA-only, interactive and\
sets HF\_HOME=/root/hf\_cache; ours takes flags, asks nothing, uses MPS else CPU. Same\
1,684 pairs and the authors' default hyperparameters, cited by line, none changed.\
Thinking off. 8B first. Memory check, no quantization. Output to runs/adapters/.\
Add a `probe` subcommand: the authors' state probes, base versus base plus adapter.\
If MPS cannot train in bf16, report it as a blocking question; do not switch dtype.
