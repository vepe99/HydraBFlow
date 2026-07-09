uv run python scripts/train.py \
  simulator=stream_agama_rnbody_huang \
  model=stream_fusion_model5 \
  composition=global \
  adapter=stream \
  preprocessing=stream_global \
  augmentation=stream_global \
  data.data_dir=data_jarvis/data_agama_rnbody_huang_hydrabflow \
  data.n_simulations=60000 \
  training.n_epochs=1000 \
  seed=2026 \
  augmentation.params.resources_dir=assets/gaia

# simulated test set (needs test_multistream_333.npz, generated after training set)
uv run python scripts/evaluate.py simulator=stream_agama_rnbody_huang model=stream_fusion_model5 \
  composition=global eval=stream_compositional adapter=stream \
  preprocessing=stream_global augmentation=stream_global \
  data.data_dir=data_jarvis/data_agama_rnbody_huang_hydrabflow data.n_simulations=333 \
  model_dir=<TRAIN_RUN_DIR> augmentation.params.resources_dir=assets/gaia

# real Gaia streams
uv run python scripts/evaluate_real.py simulator=stream_agama_rnbody_huang model=stream_fusion_model5 \
  composition=global adapter=stream \
  preprocessing=stream_real_global augmentation=stream_real_global \
  model_dir=<TRAIN_RUN_DIR> augmentation.params.resources_dir=assets/gaia