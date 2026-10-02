import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import org.json.JSONObject;

/** Restored from the original wrapper's DEX, with only max_length handling corrected. */
public class OgaGenerate {
    public static void main(String[] args) {
        String modelPath = args.length > 0 ? args[0] : "/data/local/tmp/ort_llm_qwen_cpu";
        String prompt = args.length > 1 ? args[1] : "<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n";
        int maxNewTokens = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        long start = System.nanoTime();
        try (Model model = new Model(modelPath)) {
            long loaded = System.nanoTime();
            try (Tokenizer tokenizer = new Tokenizer(model);
                 TokenizerStream stream = tokenizer.createStream();
                 Sequences input = tokenizer.encode(prompt);
                 GeneratorParams params = new GeneratorParams(model)) {
                int promptTokens = input.getSequence(0).length;
                // Original wrapper: params.setSearchOption("max_length", 256.0).
                // The limit counts prompt AND generated tokens; preserve the requested new-token budget.
                long contextLength = contextLength(modelPath);
                long effectiveMaxLength = checkedMaxLength(promptTokens, maxNewTokens, contextLength);
                params.setSearchOption("max_length", (double) effectiveMaxLength);
                params.setSearchOption("do_sample", false);
                try (Generator generator = new Generator(model, params)) {
                    generator.appendTokenSequences(input);
                    StringBuilder output = new StringBuilder();
                    long generationStart = System.nanoTime();
                    int generated = 0;
                    int lastToken = -1;
                    for (int i = 0; i < maxNewTokens && !generator.isDone(); i++) {
                        generator.generateNextToken();
                        lastToken = generator.getLastTokenInSequence(0);
                        String text = stream.decode(lastToken);
                        output.append(text);
                        System.out.print(text);
                        System.out.flush();
                        generated++;
                    }
                    long generationEnd = System.nanoTime();
                    System.out.println();
                    System.out.println("{\"status\":\"ok\",\"model_path\":\"" + esc(modelPath)
                        + "\",\"prompt_tokens\":" + promptTokens
                        + ",\"generated_tokens\":" + generated
                        + ",\"model_load_ms\":" + ((loaded - start) / 1000000.0)
                        + ",\"generation_ms\":" + ((generationEnd - generationStart) / 1000000.0)
                        + ",\"output\":\"" + esc(output.toString())
                        + "\",\"effective_max_length\":" + effectiveMaxLength
                        + ",\"max_new_tokens\":" + maxNewTokens
                        + ",\"context_length\":" + contextLength
                        + ",\"last_token_id\":" + lastToken
                        + ",\"stop_reason\":\"" + (generated < maxNewTokens ? "generator_done_before_budget" : "new_token_budget")
                        + "\",\"wrapper_revision\":\"prompt_plus_new_tokens_v1\"}");
                }
            }
        } catch (Throwable error) {
            long end = System.nanoTime();
            System.out.println("{\"status\":\"error\",\"model_path\":\"" + esc(modelPath)
                + "\",\"elapsed_ms\":" + ((end - start) / 1000000.0)
                + ",\"error\":\"" + esc(error.toString()) + "\"}");
            error.printStackTrace(System.err);
            System.exit(2);
        }
    }

    static long checkedMaxLength(int promptTokens, int maxNewTokens, long contextLength) {
        if (promptTokens < 0 || maxNewTokens < 0 || contextLength <= 0) {
            throw new IllegalArgumentException("invalid prompt/new-token/context length");
        }
        long effective = (long) promptTokens + (long) maxNewTokens;
        if (effective > contextLength) {
            throw new IllegalArgumentException("prompt_tokens + max_new_tokens (" + effective
                + ") exceeds model.context_length (" + contextLength + ")");
        }
        return effective;
    }

    private static long contextLength(String modelPath) throws Exception {
        StringBuilder text = new StringBuilder();
        try (InputStreamReader reader = new InputStreamReader(
                new FileInputStream(new File(modelPath, "genai_config.json")), "UTF-8")) {
            char[] buffer = new char[4096];
            int count;
            while ((count = reader.read(buffer)) != -1) text.append(buffer, 0, count);
        }
        return new JSONObject(text.toString()).getJSONObject("model").getLong("context_length");
    }

    private static String esc(String text) {
        return text.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n").replace("\r", "\\r");
    }
}
