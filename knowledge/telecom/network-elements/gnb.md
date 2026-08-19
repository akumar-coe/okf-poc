---
type: Network Element
title: 5G gNB
description: A 5G radio access network node providing connectivity between user equipment and the 5G core.
tags:
 - 5g
 - ran
 - gnb
domain: telecom
lifecycle: active

generated:
 by: human:okf-poc-author
 at: 2026-08-19T12:00:00+05:30
verified:
 by: process:okf-poc-validator
 at: 2026-08-19T12:15:00+05:30
status: stable
stale_after: 2026-12-31
sources:
 - id: synthetic-telecom-knowledge
   resource: synthetic://okf-poc/telecom
   title: Synthetic telecom knowledge created for the OKF PoC
   author: human:okf-poc-author

---
# 5G gNB
The gNB is the primary radio access network node in a 5G network.
## Responsibilities
- Provides radio connectivity to user equipment.
- Communicates with the 5G Core.
- Handles radio resource management.
- Supports user-plane and control-plane communication.
## Relationships
The gNB provides connectivity toward the [5G UPF](./upf.md).
## Operational relevance
High radio utilization or radio congestion can contribute to degraded user experience.
## Synthetic knowledge
This concept is synthetic knowledge created specifically for the OKF PoC.
