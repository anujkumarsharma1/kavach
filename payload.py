# The EICAR test string: the industry-standard antivirus test signature.
# It is NOT malware — it does nothing when run — but every AV engine is
# built to flag it, which is exactly why it's the standard proof that a
# signature is real and detectable. Source: https://www.eicar.org/download-anti-malware-testfile/
PAYLOAD = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

# A second payload that is deliberately NOT in scan.py's KNOWN_SIGNATURES.
# This is the generalization proof: it shows Kavach flags and extracts a
# payload it has never catalogued, on statistical grounds alone — not
# because it's pattern-matching one hardcoded string. Modeled on what a
# realistic backdoor config/exfil string looks like.
# 198.51.100.0/24 is IANA's reserved documentation range (RFC 5737) —
# guaranteed fake, never a real routable address.
UNKNOWN_PAYLOAD = b"C2-CONFIG exfil=hxxp://198.51.100.7/drop auth_token=b4d4c70ffee00"