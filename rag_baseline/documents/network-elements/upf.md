---
type: Network Element
title: 5G UPF
description: A 5G core network function responsible for user-plane packet processing and forwarding.
tags:
 - 5g
 - core
 - upf
 - user-plane
domain: telecom
lifecycle: active

generated:
 by: human:okf-poc-author
 at: 2026-08-19T12:00:00+05:30
status: stable
stale_after: 2026-12-31
sources:
 - id: synthetic-telecom-knowledge
   resource: synthetic://okf-poc/telecom
   title: Synthetic telecom knowledge created for the OKF PoC
   author: human:okf-poc-author

---
# 5G UPF
The User Plane Function handles user-plane traffic in the 5G Core.
## Responsibilities
- User-plane packet forwarding
- Traffic steering
- Connectivity toward external data networks
- Application traffic handling
## Relationships
The UPF receives user traffic from the [5G gNB](./gnb.md).
The UPF provides connectivity for the [Mobile Broadband Service](../services/mobile-broadband.md).
## Operational relevance
UPF overload can contribute to increased packet latency and packet loss.
Because the UPF provides user-plane connectivity for the Mobile Broadband Service, 
UPF overload can affect the performance of the Mobile Broadband Service.
## Synthetic knowledge
This concept is synthetic knowledge created specifically for the OKF PoC.
