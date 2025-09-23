# Fine-Tuning LLM with mlx_lm

This document walks through how to fine-tune a llm using mlx_lm framework, and how to convert the resulting fine-tuned models into GGUF format for use with llama.cpp.

## 1. Install mlx_lm :

If not already installed :

    pip install mlx_lm

**Note** : Requires Python 3.10+ and macOS with Apple Silicon ( M1/M2/M3).

## 2. Convert Hugging Face model to MLX format :

You can start with a Hugging Face transformer model and convert it to MLX for training. Quantization helps save VRAM during fine-tuning.

     mlx_lm.convert —hf-path < Hugging Face Model path > -q

Flags explained :

`- -hf-path —> Path to the Hugging Face model`

`-q —> Generate a quantized model.`

This produces a quantized mlx_model ready for fine-tuning.

## 3. Prepare the Dataset :

Organize your dataset into training, validation (optional), and test (optional) splits. For fine-tuning, only train and validation are required. Or use open source dataset from [Hugging Face](https://huggingface.co/datasets)

## 4. Fine-tune with LoRA : 

Run LoRA fine-tuning with mlx_lm.lora

    mlx_lm.lora \
  	--model ./mlx_model \
    --train \
  	--data ./data \
  	--fine-tune-type lora \
  	--optimizer adamw \
  	--batch-size 8 \
  	--iters 600 \
    --learning-rate 5e-05 \
  	--val-batches -1 \
  	--steps-per-report 10 \
    --steps-per-eval 100 \
  	--save-every 100 \
  	--mask-prompt \
  	--grad-checkpoint \
  	--max-seq-length 1024 \
  	--seed 42

Flags explained :

`--model —> Path (or Hugging Face repo) of the base model you want to fine-tune.`

`--train —> Enables training mode. Without this, the script may only load/evaluate the model.`

`--data —> Directory (or Hugging Face dataset) containing train.jsonl, valid.jsonl, and optionally.`

`--fine-tune-type —> Chooses the fine-tuning strategy. Options are:`

 `— lora : parameter efficient`

 `— dora : a bit more expressive than LoRA.`

 `— full : update all model weights.`

`--optimizer —> Optimization algorithm, adamw is widely used for transformers (better weight decay handling).`

`--batch-size —> Number of samples per step.`

`--iters —> Number of training iterations (steps).`

`--learning-rate —> Step size for updating model weights.`

`--val-batches —> Number of validation batches, -1 uses the entire validation set.`

`--steps-per-report —> Frequency (in steps) for printing training loss/logs to console.`

`--steps-per-eval —> Frequency for running validation (on the valid.jsonl set).`

`--save-every —> Save checkpoints every N steps.`

`--mask-prompt —> Mask the prompt in the loss when training.`

`--grad-checkpoint —> Activates gradient checkpointing.`

`--max-seq-length —> Maximum number of tokens per input sequence.`

`--seed —> The PRNG seed.`

This produces an [./adapters/](https://github.com/Sudeep5663/Running-llama.cpp-Locally-on-macOS/tree/main/Fine_Tune/adapters) directory containing the LoRA weights.

## 5. Generate responses with adapters (optional test) :

You can test the fine-tuned model before fusing:

    mlx_lm.generate \
	--model ./mlx_model \
	--adapter-path ./adapters/ \
	--prompt “ Your prompt here ”

Flags explained :

`--adapter-path —> Optional path for the trained adapter weights and config.`

`--prompt —> Message to be processed by the model`

## 6. Fusing Adapters :

Naively fusing adapters into the quantized mlx_model and exporting fails. You may encounter :

* NotImplementedError : Conversion of quantized models is not yet supported.
* ValueError : [save_gguf] can only serialize row-major arrays.

Instead of the MLX model, download the Hugging Face base model (non-quantized):

    huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./hf_base_mode

Fuse the adapters into this base model :

    mlx_lm.fuse \
	--model ./hf_base_model \
	--adapter-path ./adapters/ \
	--save-path ./hf_fused_model

## 7. Convert fused Hugging Face model to GGUF :

Use the llama.cpp conversion script, Clone llama.cpp repository to your local system if not :

    git clone https://github.com/ggerganov/llama.cpp.git

    python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
  	./hf_fused_model \
  	--outfile ./fused_model.ggu

## 8. Qunatize the GGUF model :

Quantization makes the GGUF model lighter and faster for CPU/GPU inference :

    /path/to/llama.cpp/quantize \
  	./fused_model.gguf \
  	./fused_model_Q4_K_M.gguf \
  	Q4_K_M

## 9. Run inference with llama.cpp :

Finally, serve the fine-tuned model :

    llama-sever \
    --model ./qunatized_fine-tuned_gguf_model_path \
    --host 0.0.0.0 \
	--port 8080 \
    --api-key <Your api key> \
    -ngl -1 \
    -t 8 \
    -kvu

## 10. Query via Python :

Create a Python Client ( client.py ) :

-- [client](https://github.com/Sudeep5663/llama.cpp/blob/main/test4.py)

Run :

    python3 client.py
