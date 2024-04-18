lm_backbone = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
sft_model = orbax_checkpointer.restore('sftmodels/')['policy_model']['params']['lm_backbone_params']['params']
lm_backbone.params = sft_model
