function isPalindrome(s) {
    // Clean the input string by removing non-alphanumeric characters and converting to lowercase
    const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, '');

    // Check if the cleaned string reads the same forward and backward
    if (cleaned === cleaned.split('').reverse().join('')) {
        return 'The string is a palindrome.';
    } else {
        return 'The string is not a palindrome.';
    }
}
