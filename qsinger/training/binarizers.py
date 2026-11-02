from basics.base_binarizer import BinarizationError
from preprocessing.acoustic_binarizer import AcousticBinarizer
from preprocessing.variance_binarizer import VarianceBinarizer


# fine tuning keeps base phonemes
class _MissingPhonemesAllowed:
    def check_coverage(self):
        try:
            super().check_coverage()
        except BinarizationError as error:
            print(f"| {error} They keep what the shared base model learned.")


class VoiceAcousticBinarizer(_MissingPhonemesAllowed, AcousticBinarizer):
    pass


class VoiceVarianceBinarizer(_MissingPhonemesAllowed, VarianceBinarizer):
    pass
