def strtobool(x):
    y = [bool(int(c)) for c in x]
    return(y)
def booltostr(y):
    x = ''.join([str(int(b)) for b in y])
    return(x)