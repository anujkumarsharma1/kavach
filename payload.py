# The EICAR test string: the industry-standard antivirus test signature.
# It is NOT malware — it does nothing when run — but every AV engine is
# built to flag it, which is exactly why it's the standard proof that a
# signature is real and detectable. Source: https://www.eicar.org/download-anti-malware-testfile/
PAYLOAD = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"