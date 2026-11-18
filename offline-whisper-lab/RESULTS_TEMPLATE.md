# Results: Private Whisper.cpp speech-recognition study

Hardware: [CPU / GPU / accelerator and RAM]
OS: [version]
whisper.cpp commit: [hash]
Whisper GGML model SHA-256: [hash]
Recording microphone and environment: [description, permission]
Audio codec/sample rate: 16kHz 16-bit mono PCM

## Model comparison

| Model | Unique clips | Corpus WER | Median batch RTF | Median wall time | Notes |
|---|---:|---:|---:|---:|---|
| tiny.en | | | | | |
| base.en | | | | | |
| small.en | | | | | |

## Error examples

| Clip | Reference span | Hypothesis span | Error type | Context |
|---|---|---|---|---|
| | | | | |

## Timing notes

RTF values include model load + audio decode + inference + output writing for each invocation. Live microphone window latency is a separate measurement and not quantified by this batch lab. Compare only with methodologically matched configurations.

## Privacy note

Record only with informed consent. Note local/cloud-synced storage and any network isolation measures.
