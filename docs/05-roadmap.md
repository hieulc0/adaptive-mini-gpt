# Roadmap and Milestone Gates

| Milestone | Scope | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **M0** | Baseline Implementation | Pre-norm Transformer, CharacterTokenizer, causal mask invariance tests, tiny-batch memorization | **PASSED** |
| **M1** | Learning & Resumption | Lawful Shakespeare corpus, atomic `CheckpointManager`, RNG/sampler restoration, corruption recovery | **PASSED** |
| **M2** | Colab T4 Qualification | FP16 `autocast` + `GradScaler`, >500k tok/s on Tesla T4, Google Drive backup sync, 20/20 test pass | **PASSED** |
| **M3** | Modular Experts & Continual Tasks | Multi-skill datasets (Text + Procedural Math), `AdaptiveMoELayer`, load-balancing loss, forgetting telemetry | **READY** |
| **M4** | Function-Preserving Growth | Zero-init expert/layer expansion CLI (`amg grow`), checkpoint migration, matched-FLOP baseline comparison | **READY** |
| **M5** | Dynamic Compute Allocation | Token-level early-exit / Mixture-of-Depths (skipping blocks on easy tokens) | **PLANNED** |
