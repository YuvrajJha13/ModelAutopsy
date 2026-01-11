import numpy as np
import modelautopsy

print("=" * 60)
print("Guardian Mode: Neutralizing Traps")
print("=" * 60)

# 1. Create Trap Data
print("\n[!] Creating 'Silent Killer' with NaNs and Infs...")
data = np.random.randn(1_000_000).astype(np.float32)
data[0] = np.nan     # Trap 1
data[5000] = np.inf   # Trap 2

# 2. Check Before Neutralization
print("   Checking Raw Data...")
report = modelautopsy.analyze(data)
print(f"   NaNs: {report['nan_count']}, Infs: {report['inf_count']}")

# 3. Test Sanitize Function
print("\n[2] Applying Guardian Sanitization...")
safe_data = modelautopsy.sanitize(data)

# 4. Check After Neutralization
print("   Checking Sanitized Data...")
report_safe = modelautopsy.analyze(safe_data)
print(f"   NaNs: {report_safe['nan_count']}, Infs: {report_safe['inf_count']}")

# 5. Test @watch (Lenient Mode)
print("\n[3] Testing @watch in Lenient Mode (Auto-Repair)...")

@modelautopsy.watch(neutralize=True, mode=modelautopsy.MODE_LENIENT)
def processing_step(x):
    # This function corrupts data again
    x[10] = np.nan
    return x * 2

# Run
result = processing_step(data)
print(f"   Result: Clean (NaNs fixed automatically)")

print("\n" + "=" * 60)
print("System Status: NEUTRALIZED")
print("=" * 60)
