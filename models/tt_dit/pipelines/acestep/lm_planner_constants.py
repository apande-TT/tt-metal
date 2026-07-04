# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Constants for ACE-Step LM constrained decoding (from ace-step/ACE-Step-1.5)."""

VALID_LANGUAGES = [
    "ar",
    "az",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "la",
    "lt",
    "ms",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sa",
    "sk",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "yue",
    "zh",
    "unknown",
]

KEYSCALE_NOTES = ["A", "B", "C", "D", "E", "F", "G"]
KEYSCALE_ACCIDENTALS = ["", "#", "b", "♯", "♭"]
KEYSCALE_MODES = ["major", "minor"]

VALID_KEYSCALES = set()
for note in KEYSCALE_NOTES:
    for acc in KEYSCALE_ACCIDENTALS:
        for mode in KEYSCALE_MODES:
            VALID_KEYSCALES.add(f"{note}{acc} {mode}")

BPM_MIN = 30
BPM_MAX = 300
DURATION_MIN = 10
DURATION_MAX = 600
VALID_TIME_SIGNATURES = [2, 3, 4, 6]
