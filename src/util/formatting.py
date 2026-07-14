def lstr(s, width=32, c=' '):
    return (s + width * c)[:width]


def pstr(i, n, c_f="*", c_e=" "):
    return "[" + (i * c_f + n * c_e)[:n] + "]"
