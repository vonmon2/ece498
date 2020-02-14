def function(a, X, b, y):
    Z = (a(X^t*X) + b^t*X)
    ans = ((Z - y) ** 2)
    return ans