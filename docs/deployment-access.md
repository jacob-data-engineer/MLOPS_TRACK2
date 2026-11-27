# Track2 deployment and recovery

Verified 27 November 2026.

Separate Elastic IP addresses are now attached for development and production. Both applications remain on the shared host and retain their existing ports. This does not create network isolation. A secondary private address was configured persistently alongside DHCP without restarting networking. Both public health endpoints passed verification. Exact infrastructure identifiers are retained in the local deployment record.

SQL polling fixes af6306e and 4e771ea were activated in the development app and worker. No production application code was changed. After the controlled restart, database connection verification took 0.30 seconds and three lightweight queue reads took 0.48–0.88 seconds.

Run 83 Epoch1 was reconciled from its successful audit to model v6; the worker reconciled Epoch2 to model v7. The experiment advanced to Comparing. No training retry was submitted. Continued monitoring is needed; these checks establish recovery at deployment time, not a guarantee against future database timeouts.
