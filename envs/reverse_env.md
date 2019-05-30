## Bitstring reversal environment

Description of the mechanics of bitstring environment involving reversals of substrings within the bitstring.

# Parameters:

Bistring length: length of bitstring (bitstring is represented as an array of bits)

Number of occluded bits: occluded bits are represented by a -1 in observations to the agent. Occlusion is a property of each bit, and therefore reversals affect occlusions (by moving them) as well.

Substring length: fixed length of each substring area that is reversed

Offset: number of bits between each reversal area.

# Actions

The index of each action corresponds to the reversal area, starting with 0 on the left and ending on the right.

As an example, consider the array of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Normally the values would be 0 and 1 but here we use other integers for ease of reading. 

Let the substring length be 4 and the offset be 1. The reversal ranges would be:

[0, 3], [1, 4], ..., [8, 11]

If the offset were two, the ranges would be:

[0, 3], [2, 5], ..., [8, 11]

Each action a_i therefore corresponds to a reversal of the ith range. For example, if the substring length were 4 and the offset were 1, and action index were 2, the action would result in:

[0, 1, 5, 4, 3, 2, 6, 7, 8, 9, 10, 11]

# Occluded bits

Suppose we had the same array as above, with some elements occluded:

[0, 1, -1, 3, -1, 5, 6, -1, 8, 9, 10, 11]. If we had the same substring length and the same offset as above, and performed the same action, we would get:

[0, 1, 5, -1, 3, -1, 6, -1, 8, 9, 10, 11].


