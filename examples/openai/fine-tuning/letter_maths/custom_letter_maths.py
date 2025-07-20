def int_to_letter(n: int) -> str:
    result = ""
    while n > 0:
        n -= 1
        result = chr(ord('a') + (n % 26)) + result
        n //= 26
    return result


def letter_to_int(s: str) -> int:
    result = 0
    for char in s:
        result = result * 26 + (ord(char) - ord('a') + 1)
    return result
