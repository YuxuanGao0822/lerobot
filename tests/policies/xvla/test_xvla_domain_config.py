#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.processor_xvla import XVLAAddDomainIdProcessorStep


def test_xvla_domain_id_is_forwarded_to_processor_step() -> None:
    config = XVLAConfig(domain_id=6)
    step = XVLAAddDomainIdProcessorStep(domain_id=config.domain_id)

    assert step.domain_id == 6
    assert step.get_config()["domain_id"] == 6


@pytest.mark.parametrize("domain_id", [-1, 30])
def test_xvla_domain_id_must_address_a_pretrained_domain(domain_id: int) -> None:
    with pytest.raises(ValueError, match="domain_id"):
        XVLAConfig(domain_id=domain_id, num_domains=30)
