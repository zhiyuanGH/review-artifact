# Corrected ORT GenAI wrapper source

`OgaGenerate.java` is the Java source used for the independent corrected ORT measurements. The original wrapper's constant `max_length=256` included both prompt and generated tokens. The corrected source encodes the full prompt, adds the requested new-token budget, checks `model.context_length` from `genai_config.json`, and fails when the bound would be exceeded. It never truncates the prompt.

The original command interface is preserved: argument 0 is the model directory, argument 1 is the prompt, and argument 2 is the new-token budget. The standalone default is 8 tokens; the measured executor requests 64. Deterministic generation, stream output, native completion checks, model/generation timing and error exit behavior remain unchanged. Added output fields expose token counts, effective length, stop reason and wrapper revision.

`build_wrapper.py` retains the actual measured build method: verify the original audited phone JAR, remove only its first `OgaGenerate` DEX class definition, retain the original GenAI Java definitions, compile the corrected wrapper, and merge it with SDK D8. It requires the external original JAR and its checksum, a JDK, and an Android SDK. The measured build used build-tools 36.1.0, android-34 platform classes, Java 8 bytecode and minimum Android API 26.

```sh
python3 experiment_support/ort_wrapper/build_wrapper.py \
  --original-jar ORIGINAL_AUDITED_JAR \
  --expected-original-sha256 ORIGINAL_AUDITED_SHA256 \
  --java-home JDK_HOME --android-sdk ANDROID_SDK
```

Host paths and the original JAR checksum are explicit inputs. The original JAR, corrected JAR/DEX/class outputs, GenAI native libraries and model files are not included. The build procedure intentionally depends on the original audited API classes rather than replacing them with an unverified library version.

`compile_only_stubs/` contains only signatures for compiling this wrapper; these stubs are excluded from the runtime output. They are not an inference implementation. `MaxLengthCheck.java` is the original host-only arithmetic checker and loads no model. It is retained as source; no compilation or inference was performed while creating this package.
