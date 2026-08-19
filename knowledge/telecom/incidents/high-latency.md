---
type: Network Incident
title: High Network Latency
description: A network condition where observed packet latency exceeds the expected operational threshold.
tags:
 - incident
 - latency
 - network-performance
domain: telecom
lifecycle: active

generated:
 by: human:okf-poc-author
 at: 2026-08-19T12:00:00+05:30
verified:
 by: process:okf-poc-validator
 at: 2026-08-19T12:25:00+05:30
status: stable
stale_after: 2026-10-31
sources:
 - id: synthetic-telecom-knowledge
   resource: synthetic://okf-poc/telecom
   title: Synthetic telecom knowledge created for the OKF PoC
   author: human:okf-poc-author

---
# High Network Latency
High network latency occurs when packets take longer than expected to traverse the network.
## Potential contributing factors
- Network congestion
- Router overload
- Radio congestion
- UPF overload
- Transport network degradation
## Related network elements
Potentially affected components include:
- [IP Router](../network-elements/router.md)
- [5G gNB](../network-elements/gnb.md)
- [5G UPF](../network-elements/upf.md)
## Investigation
The investigation should identify where latency is introduced in the end-to-end traffic path.
## Related service
The incident can affect the [Mobile Broadband Service](../services/mobile-broadband.md).
## Synthetic knowledge
This incident definition is synthetic knowledge created specifically for the OKF PoC.
