from vec import Vec
import timeit
import platform
import sys


sizes = [2000, 4000, 8000, 16000, 32000, 64000]

print("Performance of Custom Vec on Local Machine")
print("=" * 100)

print("Python version:", sys.version)
print("Machine:", platform.machine())
print("Processor:", platform.processor())
print("=" * 100)

print(
    f"{'Size':>10}"
    f"{'Addition':>15}"
    f"{'Subtraction':>15}"
    f"{'Scalar Mul':>15}"
    f"{'Negation':>15}"
    f"{'Norm':>15}"
)

print("-" * 100)

for n in sizes:

    # Create vectors
    v1 = Vec.uniform(n)
    v2 = Vec.uniform(n)

    # Measure addition
    addition_time = timeit.timeit(
        lambda: v1 + v2,
        number=10
    ) / 10

    # Measure subtraction
    subtraction_time = timeit.timeit(
        lambda: v1 - v2,
        number=10
    ) / 10

    # Measure scalar multiplication
    multiplication_time = timeit.timeit(
        lambda: 2 * v1,
        number=10
    ) / 10

    # Measure negation
    negation_time = timeit.timeit(
        lambda: -v1,
        number=10
    ) / 10

    # Measure norm
    norm_time = timeit.timeit(
        lambda: v1.norm(),
        number=10
    ) / 10

    print(
        f"{n:>10}"
        f"{addition_time:>15.8f}"
        f"{subtraction_time:>15.8f}"
        f"{multiplication_time:>15.8f}"
        f"{negation_time:>15.8f}"
        f"{norm_time:>15.8f}"
    )