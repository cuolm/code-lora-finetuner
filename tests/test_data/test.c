uint32_t count_bits(uint32_t value) {
    uint32_t count = 0;
    while(value) {
        count = count + (value & 1);
        value = (value >> 1);
    }
    return count;
}

int add(int a, int b) {
    int result = a + b;
    return result;
}
