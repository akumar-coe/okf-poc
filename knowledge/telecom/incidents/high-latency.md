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
