        # Phase 5: Skill Router Benchmark & Interference Elimination
        logger.info("--- Phase 5: Benchmarking Task-Conditioned Skill Router ---")
        target_eval_env = create_dev_deep_chain_environment(seed=seed + 1)
        composite_for_router = ResidualSkillAdapter(
            source_adapter=source_adapter,
            d_model=base_model.config.d_model,
            residual_bottleneck_dim=16,
        ).to(device)

        # Train target residual briefly to induce target specialization