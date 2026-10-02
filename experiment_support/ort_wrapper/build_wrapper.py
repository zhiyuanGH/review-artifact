"""Build the corrected wrapper locally, retaining the original GenAI Java classes.

Compile-only stubs are never included in the output JAR. The original DEX's first
class definition is OgaGenerate; remove only that definition, then merge the new
wrapper with all original ai.onnxruntime.genai definitions using SDK D8.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import struct
import subprocess
import zipfile
import zlib

BUILD = Path(__file__).resolve().parent
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--original-jar', type=Path, required=True, help='Original audited phone wrapper JAR, not included in this source package.')
parser.add_argument('--expected-original-sha256', required=True, help='Checksum of the independently retained original audited JAR.')
parser.add_argument('--java-home', type=Path, required=True, help='JDK home containing bin/java and bin/javac.')
parser.add_argument('--android-sdk', type=Path, required=True)
parser.add_argument('--build-tools', default='36.1.0')
parser.add_argument('--android-platform', default='android-34')
args = parser.parse_args()
ORIGINAL = args.original_jar.resolve()
JAVA = args.java_home.resolve() / 'bin'
SDK = args.android_sdk.resolve()
D8 = SDK / 'build-tools' / args.build_tools / 'lib/d8.jar'
ANDROID = SDK / 'platforms' / args.android_platform / 'android.jar'
EXPECTED_SHA = args.expected_original_sha256

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args):
    subprocess.run([str(x) for x in args], cwd=BUILD, check=True)

assert sha(ORIGINAL) == EXPECTED_SHA, 'Original wrapper does not match the audited phone JAR'
with zipfile.ZipFile(ORIGINAL) as archive:
    original_dex = archive.read('classes.dex')
patched = bytearray(original_dex)
class_count, class_offset = struct.unpack_from('<II', patched, 0x60)
first_class_idx = struct.unpack_from('<I', patched, class_offset)[0]
type_count, type_offset = struct.unpack_from('<II', patched, 0x40)
string_count, string_offset = struct.unpack_from('<II', patched, 0x38)
descriptor_idx = struct.unpack_from('<I', patched, type_offset + first_class_idx * 4)[0]
descriptor_offset = struct.unpack_from('<I', patched, string_offset + descriptor_idx * 4)[0]
while patched[descriptor_offset] & 0x80:
    descriptor_offset += 1
descriptor_offset += 1
descriptor = bytes(patched[descriptor_offset:]).split(b'\0', 1)[0].decode()
assert descriptor == 'LOgaGenerate;', descriptor
assert class_count > 1
struct.pack_into('<II', patched, 0x60, class_count - 1, class_offset + 32)
map_offset = struct.unpack_from('<I', patched, 0x34)[0]
map_count = struct.unpack_from('<I', patched, map_offset)[0]
map_updated = False
for i in range(map_count):
    entry = map_offset + 4 + 12 * i
    entry_type = struct.unpack_from('<H', patched, entry)[0]
    if entry_type == 0x0006:
        struct.pack_into('<II', patched, entry + 4, class_count - 1, class_offset + 32)
        map_updated = True
assert map_updated
patched[12:32] = hashlib.sha1(patched[32:]).digest()
struct.pack_into('<I', patched, 8, zlib.adler32(patched[12:]) & 0xffffffff)
library_dex = BUILD / 'original-genai-api.dex'
library_dex.write_bytes(patched)

stubs = {
    'Model': 'public Model(String path) {}',
    'Tokenizer': 'public Tokenizer(Model model) {} public TokenizerStream createStream() { return null; } public Sequences encode(String text) { return null; }',
    'TokenizerStream': 'public String decode(int token) { return null; }',
    'Sequences': 'public int[] getSequence(long index) { return null; }',
    'GeneratorParams': 'public GeneratorParams(Model model) {} public void setSearchOption(String name, double value) {} public void setSearchOption(String name, boolean value) {}',
    'Generator': 'public Generator(Model model, GeneratorParams params) {} public void appendTokenSequences(Sequences input) {} public boolean isDone() { return false; } public void generateNextToken() {} public int getLastTokenInSequence(long index) { return 0; }',
}
stub_source = BUILD / 'compile_only_stubs/ai/onnxruntime/genai'
stub_source.mkdir(parents=True, exist_ok=True)
for name, body in stubs.items():
    (stub_source / (name + '.java')).write_text('package ai.onnxruntime.genai; public final class ' + name
        + ' implements AutoCloseable { ' + body + ' public void close() {} }\n')
stub_classes = BUILD / 'compile_only_classes'
wrapper_classes = BUILD / 'wrapper_classes'
stub_classes.mkdir(exist_ok=True)
wrapper_classes.mkdir(exist_ok=True)
run([JAVA / 'javac', '--release', '8', '-d', stub_classes, *stub_source.glob('*.java')])
run([JAVA / 'javac', '-source', '8', '-target', '8', '-bootclasspath', ANDROID,
    '-classpath', stub_classes, '-d', wrapper_classes, BUILD / 'OgaGenerate.java'])
class_files = list(wrapper_classes.glob('*.class'))
assert [p.name for p in class_files] == ['OgaGenerate.class']
output = BUILD / 'oga-generate-corrected.jar'
run([JAVA / 'java', '-cp', D8, 'com.android.tools.r8.D8', '--min-api', '26',
    '--lib', ANDROID, '--output', output, library_dex, *class_files])
with zipfile.ZipFile(output) as archive:
    assert archive.namelist() == ['classes.dex'], archive.namelist()
    corrected_dex = archive.read('classes.dex')
(BUILD / 'corrected-classes.dex').write_bytes(corrected_dex)
run([SDK / 'build-tools' / args.build_tools / 'dexdump', '-d', BUILD / 'corrected-classes.dex'])
audit = {
    'original_jar': str(ORIGINAL), 'original_jar_sha256': sha(ORIGINAL),
    'corrected_jar': str(output), 'corrected_jar_sha256': sha(output),
    'source_sha256': sha(BUILD / 'OgaGenerate.java'),
    'original_dex_sha256': hashlib.sha256(original_dex).hexdigest(),
    'original_genai_api_only_dex_sha256': sha(library_dex),
    'original_class_count': class_count, 'original_genai_api_classes_retained': class_count - 1,
    'compile_only_stubs_excluded_from_runtime_jar': True,
    'native_libraries_unchanged': True,
    'java_defaults_and_args': 'Same class OgaGenerate, same model and prompt defaults, args[2] new-token budget, default 8',
    'changed_search_option': 'max_length: constant 256 -> encoded prompt token count + requested new-token budget',
    'context_bound_check': 'Read model.context_length from model/genai_config.json and fail if prompt + budget exceeds it; never truncate input',
    'unchanged_behavior': ['do_sample=false', 'generate at most args[2] tokens', 'native generator isDone/EOS guard',
        'appendTokenSequences(input)', 'getLastTokenInSequence(0)', 'TokenizerStream.decode', 'stream print and flush',
        'model_load_ms and generation_ms timing boundaries', 'success JSON original fields and output',
        'Throwable JSON error, stack trace and System.exit(2)', 'try-with-resources close order'],
    'added_audit_fields': ['effective_max_length', 'max_new_tokens', 'context_length', 'last_token_id', 'stop_reason', 'wrapper_revision'],
    'stop_reason_interpretation': 'generator_done_before_budget uses unchanged native isDone guard; new_token_budget reports loop budget exhaustion. last_token_id enables comparison to model EOS.',
    'android_min_api': 26,
    'host_compiler': str(JAVA / 'javac'),
    'dex_compiler': str(D8),
}
(BUILD / 'build_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'jar': str(output), 'sha256': sha(output)}, indent=2))
