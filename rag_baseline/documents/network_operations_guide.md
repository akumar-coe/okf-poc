# Network Operations Guide
The 5G radio access network uses gNB nodes to provide connectivity to
mobile devices.
User-plane traffic is processed by the 5G UPF. The UPF is responsible
for forwarding user traffic toward external networks.
IP routers provide connectivity between different network segments and
can be involved in transporting traffic between the radio access and
core network components.
When a mobile broadband service experiences degraded performance,
operations teams should consider the health of the gNB, UPF and routers.
For latency-related incidents, engineers should examine traffic load,
packet forwarding behaviour, routing paths and the performance of the
network elements involved.
A high-latency condition does not necessarily indicate a failure of a
single network element. Multiple components may contribute to the
observed behaviour.
UPF overload can contribute to increased packet latency and packet loss.
Because the UPF provides user-plane connectivity for the Mobile Broadband
Service, UPF overload can affect the performance of the Mobile Broadband
Service.
