import unicodedata

SILENCE = "SP"
BREATH = "AP"
GEMINATE = "cl"
GLOBAL_PHONEMES = {SILENCE, BREATH}
VOWELS = {"a", "i", "u", "e", "o", "N", GEMINATE}  # note onsets in OpenUtau
# first spelling is canonical
IPA = {
    "a": ["a", "ɑ", "ɐ"],
    "i": ["i", "ɪ"],
    "u": ["ɯ", "u", "ɨ", "ʉ", "ʊ"],
    "e": ["e", "ɛ"],
    "o": ["o", "ɔ"],
    "N": ["ɴ", "ŋ", "ɰ̃"],
    "cl": ["ʔ", "t̚", "k̚", "p̚", "tɕ̚"],
    "b": ["b"],
    "by": ["bʲ"],
    "ch": ["tɕ", "tʃ", "tɕʰ", "ʨ"],
    "d": ["d"],
    "dy": ["dʲ"],
    "f": ["ɸ", "f"],
    "g": ["g", "ɡ", "ɣ"],
    "gy": ["gʲ", "ɡʲ", "ɟ"],
    "h": ["h", "ɦ"],
    "hy": ["ç", "hʲ"],
    "j": ["dʑ", "ʑ", "dʒ"],
    "k": ["k", "kʰ"],
    "ky": ["kʲ", "kʰʲ", "kʲʰ", "c"],
    "m": ["m"],
    "my": ["mʲ"],
    "n": ["n"],
    "ny": ["ɲ", "nʲ"],
    "p": ["p", "pʰ"],
    "py": ["pʲ"],
    "r": ["ɾ", "r", "ɽ", "l"],
    "ry": ["ɾʲ", "rʲ", "lʲ"],
    "s": ["s"],
    "sh": ["ɕ", "ʃ"],
    "t": ["t", "tʰ"],
    "ts": ["ts"],
    "ty": ["tʲ"],
    "v": ["v", "β"],
    "w": ["w", "ɰ"],
    "y": ["j"],
    "z": ["z", "dz"],
}
IPA_CHARACTERS = {
    char
    for spellings in IPA.values()
    for spelling in spellings
    for char in spelling
}
SECONDARY_MARKS = set("ːˑ̥̩̪̃˞ˠˤ̴̺̯͡")  # ignored without an exact match
LENGTH_MARK = "ː"
_LOOKUP = {
    spelling: phoneme
    for phoneme, spellings in IPA.items()
    for spelling in spellings
}


def split_phones(characters: list[str]) -> list[tuple[int, str]]:
    phones = []
    for i, char in enumerate(characters):
        if phones and unicodedata.category(char) in ("Mn", "Lm", "Sk"):
            start, phone = phones[-1]
            phones[-1] = (start, phone + char)
        else:
            phones.append((i, char))
    return phones


def from_ipa(phones: list[str]) -> tuple[list[int], list[str], int]:
    starts, phonemes, dropped, i = [], [], 0, 0
    while i < len(phones):
        width = (
            2
            if i + 1 < len(phones) and _match(phones[i] + phones[i + 1])
            else 1
        )
        matched = _match("".join(phones[i : i + width]))
        if matched is None:
            dropped += 1
        else:
            # a long consonant is a geminate followed by the consonant
            if (
                phones[i + width - 1].endswith(LENGTH_MARK)
                and matched not in VOWELS
            ):
                starts.append(i)
                phonemes.append(GEMINATE)
            starts.append(i)
            phonemes.append(matched)
        i += width
    return starts, phonemes, dropped


def to_ipa(phonemes: list[str]) -> list[str]:
    return [IPA[phoneme][0] for phoneme in phonemes]


def word_lengths(phonemes: list[str]) -> list[int]:
    lengths = []
    for phoneme in phonemes:
        if phoneme in VOWELS or phoneme in GLOBAL_PHONEMES or not lengths:
            lengths.append(1)
        else:
            lengths[-1] += 1
    return lengths


def _match(spelling: str) -> str | None:
    return _LOOKUP.get(spelling) or _LOOKUP.get(
        "".join(char for char in spelling if char not in SECONDARY_MARKS)
    )
