---
type: Network Service
title: Mobile Broadband Service
description: A telecom service providing IP connectivity to mobile subscribers.
tags:
 - service
 - mobile
 - broadband
 - 5g
domain: telecom
lifecycle: active
---
# Mobile Broadband Service
The Mobile Broadband Service provides IP connectivity to mobile subscribers.
## Network path
A simplified service path is:
User Equipment
→ [5G gNB](../network-elements/gnb.md)
→ [5G UPF](../network-elements/upf.md)
→ [IP Router](../network-elements/router.md)
→ External data network
## Operational dependencies
The service depends on the availability and performance of:
- Radio access network
- 5G Core
- User-plane connectivity
- IP transport
## Related incidents
See [High Network Latency](../incidents/high-latency.md).
## Synthetic knowledge
This concept is synthetic knowledge created specifically for the OKF PoC.
