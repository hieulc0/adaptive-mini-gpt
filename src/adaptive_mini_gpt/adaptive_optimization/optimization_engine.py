        elif is_expansion:
            adapter_candidate = EvolvableResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=16,
            ).to(self.device)
        else:
            # All other arms use standard compact ResidualSkillAdapter (d=16, 2,371 params)
            adapter_candidate = ResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=self.base_model.config.d_model,
                residual_bottleneck_dim=16,
            ).to(self.device)

        trainable_params = sum(p.numel() for p in adapter_candidate.parameters() if p.requires_grad)