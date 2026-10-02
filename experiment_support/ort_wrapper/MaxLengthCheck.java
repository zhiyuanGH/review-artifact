/** Host-only arithmetic validation: no model or native inference is loaded. */
public class MaxLengthCheck {
    public static void main(String[] args) {
        check(OgaGenerate.checkedMaxLength(8, 64, 131072) == 72, "short input");
        check(OgaGenerate.checkedMaxLength(597, 64, 131072) == 661, "minimum failed input");
        check(OgaGenerate.checkedMaxLength(1888, 64, 131072) == 1952, "maximum failed input");
        check(OgaGenerate.checkedMaxLength(131008, 64, 131072) == 131072, "exact context boundary");
        rejects(131009, 64, 131072);
        rejects(Integer.MAX_VALUE, 64, 131072);
        rejects(8, -1, 131072);
        rejects(8, 64, 0);
        System.out.println("PASS: 8 prompt/budget/context arithmetic checks, no inference.");
    }
    static void check(boolean ok, String message) { if (!ok) throw new AssertionError(message); }
    static void rejects(int prompt, int budget, long context) {
        try { OgaGenerate.checkedMaxLength(prompt, budget, context); }
        catch (IllegalArgumentException expected) { return; }
        throw new AssertionError("Missing context/input rejection");
    }
}
