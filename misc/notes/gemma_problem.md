docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/models/google:/models \
  vllm/vllm-openai:v0.22.1 --model /models/gemma-4-31B-it-qat-w4a16-ct \
  --tensor-parallel-size 1 \
  --max-model-len auto \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-dtype-skip-layers sliding_window \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --reasoning-parser gemma4 \
  --speculative-config '{"model":"/models/gemma-4-31B-it-assistant","num_speculative_tokens":4}'



(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1570, in codegen_and_compile                                                                                                                                 [25/1343]
(EngineCore pid=333)     compiled_module = graph.compile_to_module()
(EngineCore pid=333)                       ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2499, in compile_to_module
(EngineCore pid=333)     return self._compile_to_module()
(EngineCore pid=333)            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2505, in _compile_to_module
(EngineCore pid=333)     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
(EngineCore pid=333)                                                              ^^^^^^^^^^^^^^
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2448, in codegen
(EngineCore pid=333)     result = self.wrapper_code.generate(self.is_inference)
(EngineCore pid=333)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/wrapper.py", line 1787, in generate
(EngineCore pid=333)     return self._generate(is_inference)
(EngineCore pid=333)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/wrapper.py", line 1854, in _generate
(EngineCore pid=333)     self.generate_and_run_autotune_block()
(EngineCore pid=333)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/wrapper.py", line 1942, in generate_and_run_autotune_block
(EngineCore pid=333)     raise RuntimeError(f"Failed to run autotuning code block: {e}") from e
(EngineCore pid=333) torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block: at 1:0:
(EngineCore pid=333) def triton_red_fused__to_copy_add_cat_clamp_index_select_mul_reciprocal_rms_norm_split_split_with_sizes_sub_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
(EngineCore pid=333) ^
(EngineCore pid=333) ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
[rank0]:[W610 10:12:02.613147412 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
