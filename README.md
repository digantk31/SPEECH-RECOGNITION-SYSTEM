# SPEECH-RECOGNITION-SYSTEM

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: DIGANT KATHIRIYA  
**INTERN ID**: CODF51  
**DOMAIN**: Artificial Intelligence Markup Language  
**DURATION**: 4 WEEKS

---

## Description of Task

During this four-week internship at CODTECH IT SOLUTIONS, I was tasked with designing and implementing a basic speech-to-text system capable of accurately transcribing short audio recordings into written text. The primary objective was to leverage existing pre-trained speech recognition models and widely adopted libraries to construct a functional end-to-end transcription pipeline, demonstrate its performance on sample audio clips, and document the process thoroughly for GitHub.

The project workflow involved several key stages:

1. **Requirement Analysis and Environment Setup**  
   I began by surveying the landscape of open-source speech recognition tools and identifying two complementary approaches: the Google Web Speech API, accessed through the `SpeechRecognition` Python package, and Facebook’s Wav2Vec2 model, available in the Hugging Face Transformers library. To ensure consistent development environments, I defined all dependencies in a `requirements.txt` file (including `speechrecognition`, `transformers`, `torch`, `torchaudio`, and `soundfile`), and verified installation on both Windows and Unix-based machines.

2. **Audio Preprocessing**  
   Accurate transcription depends heavily on audio quality. I established guidelines for incoming files: mono-channel WAV format, 16-bit PCM encoding, and a sampling rate of either 16 kHz or 44.1 kHz. This preprocessing step included validating format compliance, normalizing volume levels, and trimming silence segments to minimize background noise and reduce processing time. Whenever necessary, I converted incompatible files using Python bindings for the `soundfile` library or command-line utilities like `ffmpeg`.

3. **Transcription Module Development**  
   The core of the system resides in a single Python script, `speech_to_text.py`, which exposes a command-line interface accepting two parameters: the transcription method (`--method google` or `--method wav2vec2`) and the path to the audio file. Internally, the script defines two key functions:
   - `transcribe_google(audio_path: str) -> str`: Uses `speech_recognition.Recognizer` to record audio and calls the Google Web Speech API over the internet. It handles common exceptions such as `UnknownValueError` (unintelligible speech) and `RequestError` (network issues), providing informative error messages.
   - `transcribe_wav2vec2(audio_path: str) -> str`: Reads audio data via `soundfile`, processes it with `Wav2Vec2Processor`, runs inference through the `Wav2Vec2ForCTC` model, and decodes predicted token IDs into human-readable text. All Hugging Face artifacts are cached upon first download to optimize subsequent runs.

4. **Performance Considerations**  
   - **Runtime Limits**: While the Google API reliably handles clips up to 60 seconds, I recommended splitting longer recordings into ≤30 second segments for local Wav2Vec2 inference to balance memory usage and speed.
   - **Batch Processing**: For scalability, the script can be extended to loop through entire directories of audio files, aggregate transcriptions into a single report, and log processing times for benchmarking.

5. **Testing and Validation**  
   I selected a variety of sample recordings, including clear speech, accented voices, and mild background noise, to evaluate transcription accuracy. Results were compared between the two methods: the cloud-based Google API generally yielded higher precision on noisy inputs, while the local Wav2Vec2 model excelled on clean, studio-quality audio without internet dependencies.

6. **Documentation and GitHub Integration**  
   All source code is thoroughly commented to explain function purpose, parameter choices, and error-handling logic. The repository’s `README.md` provides detailed instructions for cloning, environment setup, and usage examples. By following Git best practices, I committed code in logical increments, wrote meaningful commit messages, and used branching to isolate feature development and bug fixes.

Overall, this internship task not only deepened my understanding of speech recognition technologies and Python tooling but also honed my ability to build, document, and deliver a real-world AI-driven application from conception through deployment.

---

## Requirements Installation

## Repository Structure

```
speech_recognition_system/
├── README.md
├── requirements.txt
└── speech_to_text.py
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/speech_recognition_system.git
   cd speech_recognition_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Run Procedure

Use the `speech_to_text.py` script to transcribe a WAV file. Below are the commands you’ll include in your README (code form):

> ### Google Web Speech API (Online)
```bash
# Transcribe with Google Web Speech API
python speech_to_text.py --method google path/to/short_clip.wav
```

> ### Wav2Vec2 (Offline)
```bash
# Transcribe with Wav2Vec2 (downloads model on first run)
python speech_to_text.py --method wav2vec2 path/to/short_clip.wav
```

## requirements.txt

```text
speechrecognition
transformers
torch
torchaudio
soundfile
```

---

*Include the above commands under a **Run Procedure** section in your GitHub `README.md`. No need to embed the full Python code—just the CLI usage snippets.*


