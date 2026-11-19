"""
Created by Patrick Callaghan (2523109)

COSC4371 Security Analytics - Final Checkpoint
Project: #15 Variant (Safe-Guard only)

Defenses:

Two defenses, each tested against every attack in attacks.py. Each defended
model gets its own threshold, chosen with the same rule as the undefended one
(lowest threshold with at most 5% of benign validation prompts flagged).

    Defense         Where it acts     What it does
    normalization   input             cleans every prompt before scoring: drops invisible
                                      format characters, applies Unicode NFKC, maps
                                      Cyrillic and Greek look-alikes to Latin, decodes
                                      embedded Base64; the trained model is unchanged
    augmentation    training data     retrains a fresh copy of each model on the training
                                      set plus one obfuscated copy of every injection and
                                      of an equal number of benign prompts
                                      (augment() in final_checkpoint.py)

What each defense can and cannot undo:

    Attack family   normalization                    augmentation
    encoding        undone exactly (text restored)   learned from examples
    structure       not undone (typos and splits     learned from examples
                    are not encoding)
    context         not undone (nothing to decode)   learned from examples

Normalization only reverses changes that have a single correct original. A
typo or an inserted hyphen has no unique correction, and a role-play frame is
meaning, not encoding, so those are left to augmentation.

Augmentation uses variant="train" templates for base64, payload_split and
role_framing, so test-time wrappers stay unseen. The four character-level
attacks use the same generator in training and test (different random draws),
so augmentation results for those are a best case.

Limitations: the look-alike table is hand-written and covers common Cyrillic
and Greek letters only; normalization would also convert genuine Cyrillic or
Greek text into Latin letters.    
"""

import base64
import binascii
import re
import unicodedata

# Look-alike letters from other scripts -> the Latin letter they imitate.
# Written as a general confusables list (Cyrillic and Greek), not copied from attacks.py,
# so it also covers look-alikes the attack never uses.
CONFUSABLES = {
    # Cyrillic lowercase
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u0456": "i", "\u0458": "j", "\u043e": "o",
    "\u0440": "p", "\u0455": "s", "\u0445": "x", "\u0443": "y", "\u04bb": "h", "\u0501": "d",
    "\u051b": "q", "\u051d": "w",
    # Cyrillic uppercase
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H", "\u0406": "I",
    "\u0408": "J", "\u041a": "K", "\u041c": "M", "\u041e": "O", "\u0420": "P", "\u0405": "S",
    "\u0422": "T", "\u0425": "X", "\u0423": "Y",
    # Greek
    "\u03bf": "o", "\u03b1": "a", "\u03b9": "i", "\u03bd": "v", "\u03c1": "p",
    "\u039f": "O", "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H", "\u0399": "I",
    "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
}
CONFUSABLE_TABLE = str.maketrans(CONFUSABLES)

# A run of 16+ Base64 characters, optionally ending in = padding.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _decode_if_text(match):
    """Replace a Base64 run with its decoded text, but only if it decodes to readable text."""
    chunk = match.group(0)
    try:
        decoded = base64.b64decode(chunk, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return chunk
    printable = sum(ch.isprintable() or ch.isspace() for ch in decoded) / max(len(decoded), 1)
    return decoded if printable >= 0.95 else chunk

def normalize(text):
    """Undo encoding-level obfuscation before a prompt reaches the model."""
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")   # 1. drop invisible format characters
    text = unicodedata.normalize("NFKC", text)                             # 2. fold compatibility forms (fullwidth etc.)
    text = text.translate(CONFUSABLE_TABLE)                                # 3. map look-alike letters to Latin
    text = BASE64_RUN.sub(_decode_if_text, text)                           # 4. decode embedded Base64
    return text