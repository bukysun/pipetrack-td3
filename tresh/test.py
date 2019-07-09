import math
def test(x, y, theta):
    r = math.sqrt(0.56**2 + 1.8**2)
    x_new = x + r * math.sin(theta)
    y_new = y + r * math.cos(theta)
    return x_new, y_new

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--theta", type=float)
    args = parser.parse_args()
    print test(args.x, args.y, args.theta)
