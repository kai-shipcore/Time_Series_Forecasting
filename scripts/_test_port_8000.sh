#!/usr/bin/env bash
# Is port 8000 reachable from this Mac? Run BEFORE and AFTER the OCI rule.
#
#     bash scripts/_test_port_8000.sh
#
# The host firewall is not blocking: iptables INPUT policy is ACCEPT with one
# redundant rule for port 3000 and no REJECT, and firewalld and ufw are both
# inactive. So the only remaining layer is the Oracle Cloud VCN security list.
#
# The distinction that matters when reading the result:
#   TIMEOUT / hang     the VCN security list is dropping the packet
#   CONNECTION REFUSED something on the host said no, or nothing is listening
#   JSON               reachable
#
# Dropped and refused are different failures. A cloud security list drops
# silently, which is why this hangs rather than erroring immediately.

set -uo pipefail
API="http://144.24.40.252:8000"

echo "=== /health, unauthenticated by design ==="
if out=$(curl -s --max-time 8 "$API/health" 2>&1); then
    echo "REACHABLE"
    echo "$out" | head -c 200; echo
else
    code=$?
    case $code in
        28) echo "TIMEOUT after 8s -> the VCN security list is dropping it. Add the ingress rule." ;;
        7)  echo "CONNECTION REFUSED -> reached the host, but nothing accepted. Check the service is up." ;;
        *)  echo "curl exit $code"; echo "$out" ;;
    esac
fi

echo
echo "=== /segmentation, should be 401 if the token is enforced ==="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$API/segmentation" 2>/dev/null)
case "$code" in
    401) echo "401 - correct. Port open AND auth enforced." ;;
    200) echo "200 - REACHABLE BUT UNAUTHENTICATED. The token is not being applied."
         echo "      POST /chat and POST /run-forecast are exposed. Close the VCN rule"
         echo "      and fix FORECAST_API_TOKEN before leaving this open." ;;
    000) echo "no response - not reachable yet" ;;
    *)   echo "unexpected status: $code" ;;
esac
