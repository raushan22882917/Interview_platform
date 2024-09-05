def is_palindrome(s: str) -> str:
    # Clean the input string by removing non-alphanumeric characters and converting to lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    # Check if the cleaned string reads the same forward and backward
    if cleaned == cleaned[::-1]:
        return 'The string is a palindrome.'
    else:
        return 'The string is not a palindrome.'

# Example usage
if __name__ == "__main__":
    # Example inputs
    inputs = ['madam', 'hello']
    
    # Processing each input
    for input_str in inputs:
        print(f"Input: {input_str}")
        print(is_palindrome(input_str))
