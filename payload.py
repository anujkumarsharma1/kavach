# The EICAR test string: the industry-standard antivirus test signature.
# It is NOT malware -- it does nothing when run -- but every AV engine is
# built to flag it, which is exactly why it's the standard proof that a
# signature is real and detectable. Source: https://www.eicar.org/download-anti-malware-testfile/
PAYLOAD = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

# ONLY NEW LINE IN THIS FILE as of the first update.
# Deliberately NOT zero. The old scan.py only ever detected payloads
# starting at float-index 0 because both tamper.py and scan.py agreed
# on that one hardcoded position. That's not a detector, it's matching
# your own test fixture. Embedding here at a non-zero, non-byte-aligned
# offset (87531 % 8 == 3, i.e. NOT aligned to a byte boundary) and having
# scan.py find it with zero foreknowledge of this number is the actual
# proof the detector generalizes. Change this constant any time and the
# new scan.py still finds it without being told where to look.
EMBED_OFFSET = 87531

# --- Added for multi-location tampering (the "10-20 hits" demo) ---
# A second, DIFFERENT payload that is deliberately NOT in scan.py's
# KNOWN_SIGNATURES list. Embedding this alongside EICAR at other layers
# proves the scanner isn't just doing string-matching against one known
# signature -- unmatched but structured payloads still get flagged
# SUSPICIOUS. This string is inert; it is not executable and does
# nothing if it were ever run.
MYSTERY_PAYLOAD = b"DEMO-UNKNOWN-PAYLOAD-NOT-EICAR-SIGNATURE-XYZ123-STRUCTURED-SAMPLE"

# How many separate layers get tampered with each payload. 15 + 3 = 18,
# inside the "10-20" range you asked for. These are counts, not a
# guarantee -- if the loaded model has fewer than this many layers large
# enough to hold a payload, multi_tamper() uses as many as actually fit
# and prints a warning rather than failing.
NUM_EICAR_LOCATIONS = 15
NUM_MYSTERY_LOCATIONS = 3