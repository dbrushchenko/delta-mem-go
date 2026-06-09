package turbogo

func packBits(codes []byte, bitWidth int) []byte {
	if bitWidth == 2 {
		n := (len(codes) + 3) / 4
		packed := make([]byte, n)
		for i, code := range codes {
			packed[i/4] |= (code & 0x03) << ((3 - (i % 4)) * 2)
		}
		return packed
	}
	n := (len(codes) + 1) / 2
	packed := make([]byte, n)
	for i, code := range codes {
		if i%2 == 0 {
			packed[i/2] = (code & 0x0F) << 4
		} else {
			packed[i/2] |= code & 0x0F
		}
	}
	return packed
}

func unpackBits(packed []byte, bitWidth, dim int) []byte {
	codes := make([]byte, dim)
	if bitWidth == 2 {
		for i := 0; i < dim; i++ {
			codes[i] = (packed[i/4] >> ((3 - (i % 4)) * 2)) & 0x03
		}
	} else {
		for i := 0; i < dim; i++ {
			if i%2 == 0 {
				codes[i] = packed[i/2] >> 4
			} else {
				codes[i] = packed[i/2] & 0x0F
			}
		}
	}
	return codes
}
