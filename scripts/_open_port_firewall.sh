#!/usr/bin/env bash
# Step 2 of opening port 8000: the host firewall. Plus the two loose ends.
#
#     bash scripts/_open_port_firewall.sh
#
# Prerequisite, already done on 2026-08-07: the service binds to 0.0.0.0 and the
# systemd unit owns the port (pid 3263790, active). Until that was true no
# firewall change could have had any effect.
#
# There are still TWO firewalls. This handles the host one. The Oracle Cloud VCN
# security list is web-console only and is printed at the end. Both must allow
# the port, and the VCN is the one people miss because the host looks correct
# and a blocked connection simply hangs rather than reporting anything.

set -uo pipefail
HOST="coverland@144.24.40.252"

echo "=== 1. which host firewall is in use? ==="
ssh -t "$HOST" '
  echo "firewalld: $(systemctl is-active firewalld 2>/dev/null || echo inactive)"
  echo "ufw:       $(systemctl is-active ufw 2>/dev/null || echo inactive)"
  echo "--- iptables INPUT chain:"
  sudo iptables -L INPUT -n --line-numbers | head -20
'

cat <<'EOF'

=== 2. apply ONE of these, matching what you just saw ===

  firewalld active:
    ssh -t coverland@144.24.40.252 'sudo firewall-cmd --permanent --add-port=8000/tcp && sudo firewall-cmd --reload'

  ufw active:
    ssh -t coverland@144.24.40.252 'sudo ufw allow 8000/tcp'

  neither, plain iptables (usual on Oracle Linux / Ubuntu OCI images):
    Insert ABOVE the catch-all REJECT rule, or it never matches. Use the line
    number of the REJECT from the listing above, e.g. if REJECT is rule 6:

    ssh -t coverland@144.24.40.252 'sudo iptables -I INPUT 6 -p tcp --dport 8000 -j ACCEPT'

    Then persist it, or it is lost at the next reboot:
      Ubuntu/Debian:  ssh -t coverland@144.24.40.252 'sudo netfilter-persistent save'
      Oracle/RHEL:    ssh -t coverland@144.24.40.252 'sudo service iptables save'

=== 3. Oracle Cloud VCN security list (web console, no CLI) ===

  Console > Networking > Virtual Cloud Networks > your VCN > Security Lists
    > Default Security List > Add Ingress Rule
        Stateless:        No
        Source Type:      CIDR
        Source CIDR:      0.0.0.0/0     <- or your office IP as x.x.x.x/32
        IP Protocol:      TCP
        Destination Port: 8000

  A /32 is the same convenience for your team and removes the public internet
  from the threat model. The API token is set, so an open port is not wide open,
  but POST /chat and POST /run-forecast sit behind that one token.

=== 4. verify FROM YOUR MAC, not from the server ===

  curl -s --max-time 5 http://144.24.40.252:8000/health
  curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 http://144.24.40.252:8000/segmentation

  Expect JSON from the first and 401 from the second. A 401 is the good outcome:
  it proves the port is reachable AND that auth is enforced.

  If it hangs rather than refusing, that is the VCN rule missing. A refusal is
  the host firewall. A hang and a refusal are different symptoms; do not treat
  them the same.

=== 5. loose end: what started the second uvicorn at 00:01:11? ===

  Not yet identified. It ran with a relative venv path and no --workers 1, from
  /opt/coverland-forecast-api, five seconds after the deploy rsync. The Next.js
  app's on-demand start is the likely source (DEPLOYMENT.md: leave
  FORECAST_SERVER_DIR unset in production). Find where that app lives and check:

    ssh -t coverland@144.24.40.252 'sudo find /opt /srv /home /var/www -maxdepth 3 -name ".env*" 2>/dev/null | head -20'
    ssh -t coverland@144.24.40.252 'sudo grep -rn "FORECAST_SERVER_DIR" /opt /srv /home /var/www --include=".env*" 2>/dev/null'

  If it is set anywhere, remove it and restart that app. Otherwise this repeats
  at the next deploy and the unit goes back to crash-looping unnoticed.
EOF
