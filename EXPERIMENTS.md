# QSinger Experiments

This document reports the results of training the analyzer and the shared models on public datasets and then fine-tuning a per-speaker model on unlabeled recordings of a single test speaker. It covers, in order, the accuracy of the analyzer labels, the analysis of the test speaker recordings, the training behavior, and the synthesis results.

## Experimental Setup

The analyzer and the shared models were trained on singing data from Ofuton P Utagoe DB, PJS, JSUT-song, and GTSinger, and on speech data from JSUT and GTSinger. The test speaker recordings consist of 48 files with a total duration of 28.6 minutes.

All training was performed on a single NVIDIA L40S GPU. The shared acoustic and variance models were each trained for 160,000 steps. The per-speaker models were fine-tuned for 30,000 steps (acoustic), 20,000 steps (variance), and 2,000 steps (vocoder), which took approximately 3 hours and 36 minutes.

## Analyzer Evaluation

On 223 items held out from recognizer training, the analyzer achieved a phoneme error rate of 6.1% and a mean phoneme onset error of 46 ms. Breath detection achieved a precision of 42.7% and a recall of 62.1%, frequently marking unvoiced regions other than breaths as breaths. With the selected threshold of 0.30, the singing detector identified all singing items among 409 evaluation items while classifying 1.3% of speech items as singing.

## Test Speaker Analysis

Of the test speaker recordings, 5.9 minutes of singing (44 segments) and 18.8 minutes of speech (155 segments) were converted into training data. A pitch stability rule applied to the same recordings identified only 0.3 minutes of singing, whereas the singing detector classified 5.9 minutes as singing.

The estimated notes deviated from the actual pitch by -0.05 semitones on average, with no octave errors. However, voicing was detected in 16% of the frames estimated as rests, indicating that note boundaries were in some cases placed earlier than the end of the actual phonation.

## Training Results

The validation losses of the shared models continued to decrease beyond 16,000 steps. By 160,000 steps, the total variance loss decreased from 0.0836 to 0.0322, the duration loss from 0.0487 to 0.0247, and the pitch loss from 0.0141 to 0.0038, while the auxiliary decoder loss of the acoustic model decreased from 0.0181 to 0.0157.

The auxiliary decoder loss of the per-speaker acoustic model reached its minimum of 0.0196 at 3,000 steps and remained nearly constant at around 0.0205 thereafter. The duration loss of the per-speaker variance model reached its minimum of 0.0159 at 1,000 steps and increased steadily thereafter, reaching 0.0191 at 4,000 steps and 0.0228 at 14,000 steps.

## Listening Observations

Synthesis in OpenUTAU functioned correctly, although some phonemes were pronounced unclearly and some long notes were not sustained to their end.

## Samples

The following samples were synthesized with the per-speaker model, using Vaundy's "踊り子" as the sample song and the UST created by エックス ([bowlroll.net/file/350099](https://bowlroll.net/file/350099)).

| <video src="assets/sample1.mp4" controls></video> | <video src="assets/sample2.mp4" controls></video> |
| :---: | :---: |
| Sample 1 | Sample 2 |

## Discussion

The per-speaker variance model, trained on approximately six minutes of singing, overfit early in training, and the acoustic model showed no further improvement after 3,000 steps. These trends, together with the failure to sustain long notes, may stem from the amount and quality of the training data, its predominance of speech, or note boundaries placed too early. Further evaluation with additional speakers and recordings is required to determine the cause.
